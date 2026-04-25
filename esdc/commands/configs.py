"""CLI command for `esdc configs`."""

from __future__ import annotations

import typer

from esdc.config_wizard import run_wizard
from esdc.configs import Config

configs_app = typer.Typer(invoke_without_command=False, no_args_is_help=False)


@configs_app.callback(invoke_without_command=True)
def configs(
    reset: bool = typer.Option(False, "--reset", help="Reset ALL config to defaults."),
    reset_key: str | None = typer.Option(
        None, "--reset-key", help="Reset a specific key to default."
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Print current configuration and exit."
    ),
    set_default_provider: str | None = typer.Option(
        None,
        "--set-default-provider",
        help="Set the default provider non-interactively.",
    ),
) -> None:
    """View and edit ESDC configuration interactively.

    Run without flags to open the interactive prompt wizard.
    Use --reset to restore defaults, --reset-key for a single key,
    --show to print config, or --set-default-provider to activate a provider.
    """
    Config.init_config()

    if reset:
        Config.reset_config()
        Config._config_cache = None
        typer.echo("All config reset to defaults.")
        return

    if reset_key:
        try:
            Config.reset_config(reset_key)
            Config._config_cache = None
            typer.echo(f"Reset '{reset_key}' to default.")
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
        return

    if set_default_provider:
        providers = Config.get_providers()
        if set_default_provider not in providers:
            typer.echo(
                f"Error: Provider '{set_default_provider}' not found."
                f" Available: {', '.join(providers.keys())}",
                err=True,
            )
            return
        Config.set_default_provider(set_default_provider)
        typer.echo(f"Default provider set to '{set_default_provider}'.")
        return

    if show:
        Config._config_cache = None
        flat = Config.get_all_config_flat()
        if not flat:
            typer.echo("No configuration found.")
            return
        for key in sorted(flat.keys()):
            print(f"  {key} = {flat[key]}")
        return

    run_wizard()
