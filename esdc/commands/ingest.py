import typer

ingest_app = typer.Typer(help="Ingest documents to knowledge graph")


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
    typer.echo(f"Ingesting {file_path}... (TODO)")


@ingest_app.command("clear")
def clear_graph(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear knowledge graph.

    Args:
        yes: Skip confirmation prompt.
    """
    typer.echo("Clearing knowledge graph... (TODO)")


if __name__ == "__main__":
    ingest_app()
