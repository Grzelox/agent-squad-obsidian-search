from pathlib import Path
from typing import List
import json
from langchain_core.tools import tool
from modules.vault_copy_service import VaultCopyService
import logging


@tool
def list_obsidian_documents(vault_path: str) -> str:
    """
    List all markdown documents available in the Obsidian vault.

    Args:
        vault_path: Path to the Obsidian vault directory

    Returns:
        JSON string containing list of available documents with their paths and metadata
    """
    try:
        vault = Path(vault_path)
        if not vault.exists():
            return json.dumps({"error": f"Vault path does not exist: {vault_path}"})

        documents = []
        markdown_files = list(vault.rglob("*.md"))

        for md_file in markdown_files:
            try:
                relative_path = md_file.relative_to(vault)
                stat = md_file.stat()

                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        preview = (
                            content[:200] + "..." if len(content) > 200 else content
                        )
                        word_count = len(content.split())
                except:
                    preview = "Could not read content"
                    word_count = 0

                documents.append(
                    {
                        "path": str(relative_path),
                        "full_path": str(md_file),
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                        "word_count": word_count,
                        "preview": preview,
                    }
                )
            except Exception as e:
                continue

        documents.sort(key=lambda x: x["modified"], reverse=True)

        result = {
            "total_documents": len(documents),
            "vault_path": vault_path,
            "documents": documents,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error listing documents: {str(e)}"})


@tool
def search_documents_by_name(vault_path: str, search_term: str) -> str:
    """
    Search for documents by filename or path containing the search term.

    Args:
        vault_path: Path to the Obsidian vault directory
        search_term: Term to search for in document names/paths

    Returns:
        JSON string containing matching documents
    """
    try:
        vault = Path(vault_path)
        if not vault.exists():
            return json.dumps({"error": f"Vault path does not exist: {vault_path}"})

        matching_docs = []
        markdown_files = list(vault.rglob("*.md"))

        search_lower = search_term.lower()

        for md_file in markdown_files:
            try:
                relative_path = md_file.relative_to(vault)
                if (
                    search_lower in str(relative_path).lower()
                    or search_lower in md_file.stem.lower()
                ):
                    stat = md_file.stat()

                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            content = f.read()
                            preview = (
                                content[:200] + "..." if len(content) > 200 else content
                            )
                            word_count = len(content.split())
                    except:
                        preview = "Could not read content"
                        word_count = 0

                    matching_docs.append(
                        {
                            "path": str(relative_path),
                            "full_path": str(md_file),
                            "size_bytes": stat.st_size,
                            "modified": stat.st_mtime,
                            "word_count": word_count,
                            "preview": preview,
                        }
                    )
            except Exception:
                continue

        matching_docs.sort(
            key=lambda x: (
                search_lower not in Path(x["path"]).stem.lower(),
                -x["modified"],
            )
        )

        result = {
            "search_term": search_term,
            "total_matches": len(matching_docs),
            "vault_path": vault_path,
            "documents": matching_docs,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error searching documents: {str(e)}"})


@tool
def get_document_info(vault_path: str, document_path: str) -> str:
    """
    Get detailed information about a specific document.

    Args:
        vault_path: Path to the Obsidian vault directory
        document_path: Relative path to the document within the vault

    Returns:
        JSON string containing detailed document information
    """
    try:
        vault = Path(vault_path)
        doc_path = vault / document_path

        if not doc_path.exists():
            return json.dumps({"error": f"Document does not exist: {document_path}"})

        stat = doc_path.stat()

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            word_count = len(content.split())
            char_count = len(content)

            headers = [line for line in lines if line.strip().startswith("#")]
            links = [line for line in lines if "[[" in line or "](" in line]

            preview_start = "\n".join(lines[:10]) if len(lines) > 10 else content
            preview_end = "\n".join(lines[-5:]) if len(lines) > 15 else ""

        except Exception as e:
            return json.dumps({"error": f"Could not read document content: {str(e)}"})

        result = {
            "path": document_path,
            "full_path": str(doc_path),
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "content_stats": {
                "word_count": word_count,
                "character_count": char_count,
                "line_count": len(lines),
                "header_count": len(headers),
                "link_count": len(links),
            },
            "headers": headers[:10],  # First 10 headers
            "preview_start": preview_start,
            "preview_end": preview_end if preview_end else None,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error getting document info: {str(e)}"})


@tool
def get_vault_info(vault_path: str) -> str:
    """
    Get information about the Obsidian vault directory, such as existence, file count, markdown file count, and errors.

    Args:
        vault_path: Path to the Obsidian vault directory

    Returns:
        JSON string containing vault information
    """
    try:
        logger = logging.getLogger("ObsidianAgent_Tools")
        vault_service = VaultCopyService(logger)
        info = vault_service.get_vault_info(vault_path)
        return json.dumps(info.__dict__, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error getting vault info: {str(e)}"})


AVAILABLE_TOOLS = [
    list_obsidian_documents,
    search_documents_by_name,
    get_document_info,
    get_vault_info,
]


def get_tools_for_agent(vault_path: str) -> List:
    """
    Get all available tools configured for the specific vault path.

    Args:
        vault_path: Path to the Obsidian vault

    Returns:
        List of configured tools
    """
    return AVAILABLE_TOOLS


def format_tool_descriptions() -> str:
    """
    Get a formatted description of all available tools for the agent prompt.

    Returns:
        String description of available tools
    """
    descriptions = []
    for tool in AVAILABLE_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description}")

    return "\n".join(descriptions)
