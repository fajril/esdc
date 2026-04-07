"""CLI commands for document ingestion."""

import json
from pathlib import Path

import typer

from esdc.configs import Config
from esdc.knowledge_graph.graph_db import LadybugDB
from esdc.knowledge_graph.pipeline import IngestionPipeline
from esdc.providers import create_llm_from_config

ingest_app = typer.Typer(help="Ingest documents to knowledge graph")


def get_llm():
    """Get LLM instance from config."""
    provider_config = Config.get_provider_config()
    if not provider_config:
        typer.echo("Error: No provider configured. Run 'esdc chat' first.", err=True)
        raise typer.Exit(1)

    try:
        return create_llm_from_config(provider_config)
    except Exception as e:
        typer.echo(f"Error creating LLM: {e}", err=True)
        raise typer.Exit(1) from None


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="Path to file to ingest"),
    doc_type: str | None = typer.Option(None, "--type", help="Document type override"),
    interactive: bool = typer.Option(False, "--interactive", help="Interactive mode"),
    analyze: bool = typer.Option(False, "--analyze", help="Analyze only, don't store"),
):
    """Ingest a file to knowledge graph.

    Args:
        file_path: Path to the file to ingest.
        doc_type: Optional document type override.
        interactive: Interactive mode flag.
        analyze: Analyze only, don't store.
    """
    path = Path(file_path)
    if not path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processing: {file_path}")
    if analyze:
        typer.echo("Mode: analyze only (no storage)")

    llm = get_llm()
    db = LadybugDB() if not analyze else None

    pipeline = IngestionPipeline(db=db, llm=llm)

    try:
        result = pipeline.ingest(
            file_path=path,
            doc_type_override=doc_type,
            interactive=interactive,
            analyze_only=analyze,
        )

        if result.get("success"):
            if analyze:
                analysis = result.get("analysis", {})
                typer.echo("\n📊 Analysis Results:")
                typer.echo(f"  Document ID: {analysis.get('doc_id')}")
                typer.echo(f"  Type: {analysis.get('doc_type')}")
                typer.echo(f"  Timeless: {analysis.get('is_timeless')}")

                entities = analysis.get("entities", [])
                if entities:
                    typer.echo(f"\n  Entities found ({len(entities)}):")
                    for entity in entities[:10]:
                        typer.echo(f"    - {entity.get('name')} ({entity.get('type')})")
                    if len(entities) > 10:
                        typer.echo(f"    ... and {len(entities) - 10} more")

                sections = analysis.get("sections", [])
                if sections:
                    typer.echo(f"\n  Sections ({len(sections)}):")
                    for section in sections:
                        typer.echo(f"    - {section}")

                if typer.confirm("\nSave full analysis as JSON?", default=False):
                    typer.echo(json.dumps(analysis, indent=2))
            else:
                doc_id = result.get("doc_id")
                typer.echo("\n✅ Document ingested successfully")
                typer.echo(f"  Document ID: {doc_id}")

                if db and not db.is_available():
                    typer.echo(
                        "\n  ⚠️  Note: LadybugDB not available. "
                        "Document processed but not stored.",
                        err=True,
                    )
                    typer.echo(
                        "  Install real-ladybug to enable storage: "
                        "pip install real-ladybug",
                        err=True,
                    )
        else:
            error = result.get("error", "Unknown error")
            typer.echo(f"\n❌ Error: {error}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"\n❌ Error processing document: {e}", err=True)
        raise typer.Exit(1) from None


@ingest_app.command("clear")
def clear_graph(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear knowledge graph.

    Args:
        yes: Skip confirmation prompt.
    """
    db = LadybugDB()

    if not db.is_available():
        typer.echo(
            "Error: LadybugDB not available. Cannot clear knowledge graph.",
            err=True,
        )
        raise typer.Exit(1)

    if not yes:
        typer.echo(
            "⚠️  This will delete ALL documents and entities from the knowledge graph."
        )
        if not typer.confirm("Are you sure?", default=False):
            typer.echo("Cancelled.")
            raise typer.Exit(0)

    typer.echo("Clearing knowledge graph... (TODO)")
    typer.echo("Note: Clear functionality not yet implemented.")


if __name__ == "__main__":
    ingest_app()
