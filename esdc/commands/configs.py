"""CLI command for `esdc configs`."""

from __future__ import annotations

from pathlib import Path

import typer

from esdc.config_fields import format_config_value
from esdc.config_wizard import run_wizard
from esdc.configs import Config
from esdc.model_settings import ModelTarget, TaskAssignment

configs_app = typer.Typer(invoke_without_command=False, no_args_is_help=False)


def _pair(raw: str, *, option: str) -> tuple[str, str]:
    """Parse one ``CONNECTION:MODEL`` pair, rejecting an ambiguous half."""
    if ":" not in raw:
        raise ValueError(
            f"{option} needs CONNECTION:MODEL (for example gateway:gpt-4o)."
        )
    connection, model = raw.split(":", 1)
    connection = connection.strip()
    model = model.strip()
    if not connection or not model:
        raise ValueError(
            f"{option} needs both a connection and a model: CONNECTION:MODEL."
        )
    return connection, model


def _save_chat(
    primary_value: str | None = None,
    fallbacks_value: str | None = None,
) -> None:
    """Save Chat primary and/or ordered fallbacks in one canonical write."""
    current = Config.get_model_assignment("chat")
    primary = current.target
    if primary_value is not None:
        connection, model = _pair(primary_value, option="--set-chat-primary")
        options = (
            dict(primary.options)
            if primary is not None
            and (primary.connection, primary.model) == (connection, model)
            else {}
        )
        primary = ModelTarget(connection, model, options)

    if primary is None:
        raise ValueError(
            "Chat needs a primary first; use --set-chat-primary CONNECTION:MODEL."
        )

    fallbacks = current.fallbacks
    if fallbacks_value is not None:
        entries = [
            item.strip() for item in fallbacks_value.split(",") if item.strip()
        ]
        fallbacks = tuple(
            ModelTarget(*_pair(item, option="--set-chat-fallbacks"), {})
            for item in entries
        )

    Config.save_model_assignment(
        "chat",
        TaskAssignment(mode="model", target=primary, fallbacks=fallbacks),
    )
    if primary_value is not None:
        typer.echo(f"Chat primary set to '{primary.connection}' / '{primary.model}'.")
    if fallbacks_value is not None:
        if fallbacks:
            typer.echo(
                "Chat fallbacks set to: "
                + ", ".join(
                    f"{pair.connection}/{pair.model}" for pair in fallbacks
                )
            )
        else:
            typer.echo("Chat fallbacks cleared.")


@configs_app.callback(invoke_without_command=True)
def configs(
    ctx: typer.Context,
    reset: bool = typer.Option(False, "--reset", help="Reset ALL config to defaults."),
    reset_key: str | None = typer.Option(
        None, "--reset-key", help="Reset a specific key to default."
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Print current configuration and exit."
    ),
    set_chat_primary: str | None = typer.Option(
        None,
        "--set-chat-primary",
        help="Set Chat primary as CONNECTION:MODEL (keeps distinct fallbacks).",
    ),
    set_chat_fallbacks: str | None = typer.Option(
        None,
        "--set-chat-fallbacks",
        help="Set ordered Chat fallbacks as CONNECTION:MODEL,... pairs.",
    ),
    set_default_provider: str | None = typer.Option(
        None,
        "--set-default-provider",
        help="Retired; use --set-chat-primary CONNECTION:MODEL.",
    ),
    set_provider_order: str | None = typer.Option(
        None,
        "--set-provider-order",
        help="Retired; use --set-chat-fallbacks CONNECTION:MODEL,...",
    ),
) -> None:
    """View and edit ESDC configuration interactively.

    Run without flags to open the interactive prompt wizard.
    Use --reset to restore defaults, --reset-key for a single key
    (including a whole task as models.<role>), --show to print config,
    --set-chat-primary CONNECTION:MODEL, or --set-chat-fallbacks
    CONNECTION:MODEL,... for Chat failover.
    """
    # Conversion must work even when the existing file is pre-canonical:
    # initializing or validating canonical settings here would block it.
    if ctx.invoked_subcommand == "convert-model-settings":
        return

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
        except (KeyError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None
        return

    if set_default_provider is not None:
        typer.echo(
            "Error: --set-default-provider is retired; use "
            "--set-chat-primary CONNECTION:MODEL.",
            err=True,
        )
        raise typer.Exit(1)
    if set_provider_order is not None:
        typer.echo(
            "Error: --set-provider-order is retired; use "
            "--set-chat-fallbacks CONNECTION:MODEL,...",
            err=True,
        )
        raise typer.Exit(1)

    if set_chat_primary is not None or set_chat_fallbacks is not None:
        try:
            _save_chat(set_chat_primary, set_chat_fallbacks)
        except ValueError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from None
        return

    if show:
        Config._config_cache = None
        flat = Config.get_all_config_flat()
        if not flat:
            typer.echo("No configuration found.")
            return
        for key in sorted(flat.keys()):
            print(f"  {key} = {format_config_value(key, flat[key])}")
        return

    run_wizard()


@configs_app.command("convert-model-settings")
def convert_model_settings_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input",
        help="Pre-canonical or canonical YAML file to preview/convert.",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write the converted settings after a successful preview.",
    ),
    backup: Path | None = typer.Option(  # noqa: B008
        None,
        "--backup",
        help="New private file that receives the complete source backup.",
        dir_okay=False,
    ),
) -> None:
    """Preview or apply the one-time unified model-settings conversion."""
    from esdc.model_config_apply import (
        ModelSettingsApplyError,
        apply_model_settings,
        preview_model_settings,
        render_preview,
    )

    try:
        preview = preview_model_settings(input_path)
    except ModelSettingsApplyError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from None

    for line in render_preview(preview):
        typer.echo(line)

    if preview.conversion.blockers:
        typer.echo("Conversion is blocked; nothing was written.", err=True)
        raise typer.Exit(1)

    if not apply:
        if backup is not None:
            typer.echo("Error: --backup requires --apply.", err=True)
            raise typer.Exit(1)
        return

    if preview.noop:
        typer.echo("Already canonical; nothing was written.")
        return

    if backup is None:
        typer.echo(
            "Error: --apply requires --backup NEW_PATH.", err=True
        )
        raise typer.Exit(1)

    try:
        result = apply_model_settings(input_path, backup)
    except ModelSettingsApplyError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from None

    if result.conversion.blockers:
        for blocker in result.conversion.blockers:
            typer.echo(f"Error: {blocker}", err=True)
        raise typer.Exit(1)
    if result.noop:
        typer.echo("Already canonical; nothing was written.")
        return
    typer.echo(
        f"Converted model settings in '{input_path}'. Backup: '{backup}'."
    )
