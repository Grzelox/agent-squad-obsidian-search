import click
from .config import get_config

config = get_config()


def cli_options(func):
    """Decorator that applies all CLI options to the main function."""

    options = [
        click.option(
            "--vault-path",
            "-v",
            required=True,
            help="Path to your source Obsidian vault directory",
        ),
        click.option(
            "--destination",
            "-d",
            default="./vault",
            help="Destination path for copied vault (default: ./vault)",
        ),
        click.option(
            "--model",
            "-m",
            default=config.model_name,
            help=f"Ollama model to use (default: {config.model_name})",
        ),
        click.option(
            "--embedding-model",
            "-e",
            default=config.embedding_model,
            help=f"Embedding model to use (default: {config.embedding_model})",
        ),
        click.option(
            "--rebuild",
            "-r",
            is_flag=True,
            help="Force rebuild of vector store",
        ),
        click.option(
            "--chroma-host",
            default=None,
            help="ChromaDB host for remote connection (default: use local file-based storage)",
        ),
        click.option(
            "--chroma-port",
            default=8000,
            type=int,
            help="ChromaDB port for remote connection (default: 8000)",
        ),
        click.option(
            "--collection-name",
            default=config.collection_name,
            help=f"ChromaDB collection name (default: {config.collection_name})",
        ),
        click.option(
            "--verbose",
            is_flag=True,
            help="Enable verbose logging output",
        ),
        click.option(
            "--quiet",
            is_flag=True,
            help="Hide all log messages, show only essential output",
        ),
        click.option(
            "--enable-summarization",
            is_flag=True,
            help="Enable automatic summarization of long documents during indexing",
        ),
        click.option(
            "--summarization-min-words",
            default=config.summarization_min_words,
            type=int,
            help=f"Minimum word count for document summarization (default: {config.summarization_min_words})",
        ),
        click.option(
            "--markdown-mode",
            default=config.markdown_mode,
            type=click.Choice(["single", "elements"]),
            help="Markdown parsing mode: 'single' for whole documents, 'elements' for structured parsing (default: single)",
        ),
        click.option(
            "--markdown-strategy",
            default=config.markdown_strategy,
            type=click.Choice(["auto", "hi_res", "fast"]),
            help="Markdown parsing strategy: 'auto', 'hi_res' for better quality, 'fast' for speed (default: auto)",
        ),
    ]

    for option in reversed(options):
        func = option(func)

    return func
