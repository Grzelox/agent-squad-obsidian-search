import click
from pathlib import Path

from modules import ObsidianAgent, VaultCopyService, setup_cli_logger, get_config
from modules.cli_config import cli_options


@click.command()
@cli_options
def main(
    vault_path: str,
    destination: str,
    model: str,
    embedding_model: str,
    rebuild: bool,
    chroma_host: str,
    chroma_port: int,
    collection_name: str,
    verbose: bool,
    quiet: bool,
    enable_summarization: bool,
    summarization_min_words: int,
    markdown_mode: str,
    markdown_strategy: str,
):
    """Obsidian AI Agent CLI

    An intelligent search agent that copies your Obsidian vault to a working directory,
    indexes it, and allows you to query your knowledge base using natural language.

    Features include optional document summarization for long documents, enhanced markdown
    parsing with UnstructuredMarkdownLoader, and special commands to view summaries and statistics.

    Configuration can be provided via environment variables or CLI arguments.
    CLI arguments take precedence over environment variables.

    Examples:
        python main.py -v "/path/to/vault"
        python main.py -v "/path/to/vault" -d "./my_working_vault"
        python main.py -v "/path/to/vault" -m llama2 -r
        python main.py -v "/path/to/vault" --chroma-host localhost --chroma-port 8000
        python main.py -v "/path/to/vault" --verbose
        python main.py -v "/path/to/vault" --quiet
        python main.py -v "/path/to/vault" --enable-summarization --summarization-min-words 300
        python main.py -v "/path/to/vault" --markdown-mode elements --markdown-strategy hi_res
        python main.py -v "/path/to/vault" --enable-summarization --markdown-mode elements --verbose
    """

    if verbose and quiet:
        click.echo("Error: --verbose and --quiet cannot be used together", err=True)
        raise click.Abort()

    config = get_config()
    config.update_from_cli(
        model_name=model,
        embedding_model=embedding_model,
        collection_name=collection_name,
        summarization_enabled=enable_summarization,
        summarization_min_words=summarization_min_words,
        markdown_mode=markdown_mode,
        markdown_strategy=markdown_strategy,
    )

    main_logger = setup_cli_logger(verbose=verbose, quiet=quiet)

    main_logger.info(
        f"CLI started with source vault: {vault_path}, destination: {destination}, "
        f"model: {config.model_name}, embedding: {config.embedding_model}"
    )
    main_logger.info(
        f"Markdown processing: mode={config.markdown_mode}, strategy={config.markdown_strategy}"
    )
    if config.summarization_enabled:
        main_logger.info(
            f"Document summarization enabled (min words: {config.summarization_min_words})"
        )
        click.echo(
            f"📝 Document summarization enabled for documents with {config.summarization_min_words}+ words"
        )
    if chroma_host:
        main_logger.info(f"Using remote ChromaDB: {chroma_host}:{chroma_port}")
        main_logger.info(f"Collection name: {config.collection_name}")
    else:
        main_logger.info("Using local ChromaDB storage")

    try:
        vault_copy_service = VaultCopyService(main_logger)

        destination_path = Path(destination)
        click.echo(f"Copying vault from '{vault_path}' to '{destination_path}'...")
        working_vault_path = vault_copy_service.copy_vault(vault_path, destination_path)
        click.echo(f"✓ Vault copied successfully to '{working_vault_path}'")

        agent = ObsidianAgent(
            obsidian_vault_path=working_vault_path,
            model_name=config.model_name,
            embedding_model=config.embedding_model,
            persist_directory=config.persist_directory,
            log_file=config.logs_file,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=config.collection_name,
            verbose=verbose,
            quiet=quiet,
        )

        agent.initialize(force_rebuild=rebuild)

        if chroma_host:
            click.echo(
                f"\n🌐 ChromaDB Explorer available at: http://{chroma_host}:{chroma_port}"
            )
            click.echo(f"📊 Collection name: {config.collection_name}")

        main_logger.info("Starting interactive query session")
        click.echo("\nObsidian AI Agent ready! Type 'quit' to exit.\n")

        if config.summarization_enabled:
            click.echo("💡 Special commands:")
            click.echo("  - Type 'summaries' to view all document summaries")
            click.echo("  - Type 'stats' to see summarization statistics")
            click.echo()

        query_count = 0
        while True:
            question = click.prompt("Ask a question", type=str)

            if question.lower() in ["quit", "exit", "q"]:
                main_logger.info(f"User exited after {query_count} queries")
                break

            # Handle special commands for summarization
            if question.lower() in ["summaries", "list summaries"]:
                try:
                    summaries = agent.get_document_summaries()
                    if not summaries:
                        click.echo("No document summaries found.")
                    else:
                        click.echo(
                            f"\n📚 Document Summaries ({len(summaries)} documents):"
                        )
                        click.echo("=" * 60)
                        for doc_path, info in summaries.items():
                            click.echo(f"\n📄 {doc_path}")
                            click.echo(f"Words: {info.get('word_count', 'unknown')}")
                            click.echo(f"Summary: {info['summary']}")
                            click.echo("-" * 40)
                    click.echo("\n" + "=" * 50 + "\n")
                    continue
                except Exception as e:
                    click.echo(f"Error retrieving summaries: {e}")
                    continue

            if question.lower() in ["stats", "statistics", "summary stats"]:
                try:
                    stats = agent.get_summarized_documents_stats()
                    click.echo(f"\n📊 Summarization Statistics:")
                    click.echo("=" * 40)
                    click.echo(
                        f"Total summarized documents: {stats['total_summaries']}"
                    )
                    click.echo(f"Average word count: {stats['avg_word_count']}")
                    if stats["documents"]:
                        click.echo(f"\nSummarized documents:")
                        for doc in stats["documents"]:
                            click.echo(f"  - {doc}")
                    click.echo("\n" + "=" * 50 + "\n")
                    continue
                except Exception as e:
                    click.echo(f"Error retrieving statistics: {e}")
                    continue

            try:
                query_count += 1
                main_logger.info(f"Processing query #{query_count}")
                result = agent.query(question)

                click.echo(f"\nAnswer:")
                click.echo(f"{result['answer']}")

                if result["sources"]:
                    source_details = result.get("source_details", {})
                    click.echo(f"\nSources:")

                    # Show original sources
                    original_sources = source_details.get("original_sources", [])
                    summary_sources = source_details.get("summary_sources", [])

                    if original_sources:
                        click.echo(
                            f"\n📄 Original content ({len(original_sources)} documents):"
                        )
                        for source in original_sources:
                            click.echo(f"  - {source}")
                            # Show summary if available
                            if config.summarization_enabled:
                                summary = agent.get_summary_for_document(source)
                                if summary:
                                    click.echo(f"    📝 Summary: {summary}")

                    if summary_sources:
                        click.echo(
                            f"\n🎯 Summary content ({len(summary_sources)} summaries):"
                        )
                        for source in summary_sources:
                            click.echo(f"  - {source}")

                    # Show total retrieval info
                    total_chunks = source_details.get(
                        "total_chunks", len(result["sources"])
                    )
                    if config.summarization_enabled and (
                        original_sources or summary_sources
                    ):
                        click.echo(
                            f"\n📊 Retrieved {total_chunks} chunks total from original + summary content"
                        )

                click.echo("\n" + "=" * 50 + "\n")
                main_logger.info(f"Query #{query_count} completed successfully")

            except Exception as e:
                main_logger.error(f"Error processing query #{query_count}: {str(e)}")
                click.echo(f"Error processing question: {e}")

    except Exception as e:
        main_logger.error(f"Critical error in main: {str(e)}")
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
