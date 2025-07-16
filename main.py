#!/usr/bin/env python3
"""
Obsidian AI Search Agent - Main Entry Point

A command-line interface for the Obsidian AI Search Agent that indexes your
Obsidian vault and allows natural language querying using Ollama AI models.
"""

import click
from pathlib import Path

from modules import ObsidianAgent, VaultCopyService, setup_cli_logger


@click.command()
@click.option(
    "--vault-path",
    "-v",
    required=True,
    help="Path to your source Obsidian vault directory",
)
@click.option(
    "--destination",
    "-d",
    default="./vault",
    help="Destination path for copied vault (default: ./vault)",
)
@click.option(
    "--model",
    "-m",
    default="llama3.2",
    help="Ollama model to use (default: llama3.2)",
)
@click.option(
    "--embedding-model",
    "-e",
    default="nomic-embed-text",
    help="Embedding model to use (default: nomic-embed-text)",
)
@click.option(
    "--rebuild",
    "-r",
    is_flag=True,
    help="Force rebuild of vector store",
)
@click.option(
    "--chroma-host",
    default=None,
    help="ChromaDB host for remote connection (default: use local file-based storage)",
)
@click.option(
    "--chroma-port",
    default=8000,
    type=int,
    help="ChromaDB port for remote connection (default: 8000)",
)
@click.option(
    "--collection-name",
    default="obsidian_documents",
    help="ChromaDB collection name (default: obsidian_documents)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging output",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Hide all log messages, show only essential output",
)
def main(vault_path: str, destination: str, model: str, embedding_model: str, rebuild: bool, 
         chroma_host: str, chroma_port: int, collection_name: str, verbose: bool, quiet: bool):
    """Obsidian AI Agent CLI

    An intelligent search agent that copies your Obsidian vault to a working directory,
    indexes it, and allows you to query your knowledge base using natural language.

    Examples:
        python main.py -v "/path/to/vault"
        python main.py -v "/path/to/vault" -d "./my_working_vault"
        python main.py -v "/path/to/vault" -m llama2 -r
        python main.py -v "/path/to/vault" --chroma-host localhost --chroma-port 8000
        python main.py -v "/path/to/vault" --verbose
        python main.py -v "/path/to/vault" --quiet
    """

    # Validate conflicting options
    if verbose and quiet:
        click.echo("Error: --verbose and --quiet cannot be used together", err=True)
        raise click.Abort()

    # Setup CLI logger with verbose and quiet flags
    main_logger = setup_cli_logger(verbose=verbose, quiet=quiet)
    
    # Log startup parameters
    main_logger.info(
        f"CLI started with source vault: {vault_path}, destination: {destination}, "
        f"model: {model}, embedding: {embedding_model}"
    )
    if chroma_host:
        main_logger.info(f"Using remote ChromaDB: {chroma_host}:{chroma_port}")
        main_logger.info(f"Collection name: {collection_name}")
    else:
        main_logger.info("Using local ChromaDB storage")

    try:
        # Initialize vault copy service
        vault_copy_service = VaultCopyService(main_logger)
        
        # Copy vault to destination
        destination_path = Path(destination)
        click.echo(f"Copying vault from '{vault_path}' to '{destination_path}'...")
        working_vault_path = vault_copy_service.copy_vault(vault_path, destination_path)
        click.echo(f"✓ Vault copied successfully to '{working_vault_path}'")

        # Initialize agent with the working vault path and ChromaDB configuration
        main_logger.info("Creating ObsidianAgent instance")
        agent = ObsidianAgent(
            obsidian_vault_path=str(working_vault_path),
            model_name=model,
            embedding_model=embedding_model,
            chroma_host=chroma_host,
            chroma_port=chroma_port if chroma_host else None,
            collection_name=collection_name,
            verbose=verbose,
            quiet=quiet,
        )

        agent.initialize(force_rebuild=rebuild)

        # Display ChromaDB explorer info if using remote
        if chroma_host:
            click.echo(f"\n🌐 ChromaDB Explorer available at: http://{chroma_host}:{chroma_port}")
            click.echo(f"📊 Collection name: {collection_name}")

        # Interactive query loop
        main_logger.info("Starting interactive query session")
        click.echo("\nObsidian AI Agent ready! Type 'quit' to exit.\n")

        query_count = 0
        while True:
            question = click.prompt("Ask a question", type=str)

            if question.lower() in ["quit", "exit", "q"]:
                main_logger.info(f"User exited after {query_count} queries")
                break

            try:
                query_count += 1
                main_logger.info(f"Processing query #{query_count}")
                result = agent.query(question)

                click.echo(f"\nAnswer:")
                click.echo(f"{result['answer']}")

                if result["sources"]:
                    click.echo(f"\nSources:")
                    for source in result["sources"]:
                        click.echo(f"  - {source}")

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
