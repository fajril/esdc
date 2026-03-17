# esdc/commands/provider.py
import typer
from esdc.configs import Config

provider_app = typer.Typer(invoke_without_command=False)


@provider_app.command("list")
def provider_list():
    """List all configured providers."""
    providers = Config.get_providers()
    if not providers:
        typer.echo(
            "No providers configured. Use 'esdc provider add' to add a provider."
        )
        return

    lines = ["Configured providers:"]
    for name, cfg in providers.items():
        lines.append(f"  - {name}: {cfg.get('provider_type', 'unknown')}")
    typer.echo("\n".join(lines))


@provider_app.command("add")
def provider_add(
    name: str,
    provider_type: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
):
    """Add a new provider."""
    config = {
        "provider_type": provider_type,
    }
    if api_key:
        config["api_key"] = api_key
    if model:
        config["model"] = model
    if base_url:
        config["base_url"] = base_url

    Config.save_provider(name, config)
    typer.echo(f"Provider '{name}' added successfully.")


@provider_app.command("remove")
def provider_remove(name: str):
    """Remove a provider."""
    if Config.remove_provider(name):
        typer.echo(f"Provider '{name}' removed successfully.")
    else:
        typer.echo(f"Provider '{name}' not found.")


@provider_app.command("set-default")
def provider_set_default(name: str):
    """Set the default provider."""
    providers = Config.get_providers()
    if name not in providers:
        typer.echo(f"Provider '{name}' not found.")
        return
    Config.set_default_provider(name)
    typer.echo(f"Default provider set to '{name}'.")


if __name__ == "__main__":
    provider_app()
