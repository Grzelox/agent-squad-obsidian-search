#!/usr/bin/env python3
"""
Obsidian AI Search Agent - Main Entry Point

A command-line interface for the Obsidian AI Search Agent that indexes your
Obsidian vault and allows natural language querying using Ollama AI models.
"""

import click
from pathlib import Path

from modules import ObsidianAgent, setup_cli_logger


@click.command()
@click.option(
    "--vault-path",
    "-v",
    required=True,
    help="Path to your Obsidian vault directory",
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
def main(vault_path: str, model: str, embedding_model: str, rebuild: bool):
    """Obsidian AI Agent CLI

    An intelligent search agent that indexes your Obsidian vault and allows
    you to query your knowledge base using natural language.

    Examples:
        python main.py -v "/path/to/vault"
        python main.py -v "/path/to/vault" -m llama2 -r
    """

    # Setup CLI logger
    main_logger = setup_cli_logger()
    main_logger.info(
        f"CLI started with vault: {vault_path}, model: {model}, embedding: {embedding_model}"
    )

    # Validate vault path
    if not Path(vault_path).exists():
        main_logger.error(f"Vault path does not exist: {vault_path}")
        click.echo(f"Error: Vault path '{vault_path}' does not exist")
        return

    try:
        # Initialize agent
        main_logger.info("Creating ObsidianAgent instance")
        agent = ObsidianAgent(
            obsidian_vault_path=vault_path,
            model_name=model,
            embedding_model=embedding_model,
        )

        agent.initialize(force_rebuild=rebuild)

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
        click.echo(f"Error initializing agent: {e}")


if __name__ == "__main__":
    main()
