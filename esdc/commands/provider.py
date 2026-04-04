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

    default_provider = Config.get_default_provider()

    lines = ["Configured providers:"]
    for name, cfg in providers.items():
        type_str = cfg.get("provider_type", "unknown")
        marker = " (default)" if name == default_provider else ""
        lines.append(f"  - {name}: {type_str}{marker}")
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
    valid_types = ["ollama", "openai", "openai_compatible"]
    if provider_type not in valid_types:
        typer.echo(f"Invalid provider type. Must be one of: {', '.join(valid_types)}")
        return

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


@provider_app.command("test")
def provider_test(name: str = ""):
    """Test provider connection. Uses default provider if name not provided."""
    from esdc.providers import ProviderConfig, get_provider

    if name:
        providers = Config.get_providers()
        if name not in providers:
            typer.echo(f"Provider '{name}' not found.")
            return
        cfg = providers[name]
    else:
        cfg = Config.get_provider_config()
        if not cfg:
            typer.echo("No default provider configured.")
            return
        name = Config.get_default_provider() or "default"

    provider_type = cfg.get("provider_type")
    if not provider_type:
        typer.echo("Provider type not specified.")
        return

    provider_class = get_provider(provider_type)
    if not provider_class:
        typer.echo(f"Unknown provider type: {provider_type}")
        return

    provider_config = ProviderConfig(
        name=name,
        provider_type=provider_type,
        model=cfg.get("model", ""),
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
    )

    typer.echo(f"Testing {provider_type} provider '{name}'...")
    success, message = provider_class.test_connection(provider_config)

    if success:
        typer.secho(f"Success: {message}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Failed: {message}", fg=typer.colors.RED)


if __name__ == "__main__":
    provider_app()
