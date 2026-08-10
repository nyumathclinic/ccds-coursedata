"""Google Drive CLI commands for ``dataset get drive``."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Annotated, Any, Optional

from loguru import logger
from platformdirs import user_config_path
import typer

try:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google_auth_oauthlib.flow import InstalledAppFlow

    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    google = None
    DefaultCredentialsError = Exception
    Request = None
    Credentials = None
    build = None
    HttpError = Exception
    InstalledAppFlow = None
    GOOGLE_DRIVE_AVAILABLE = False

from coursedata.config import DRIVE_CONFIG, RAW_DATA_DIR
from ._utils import d8

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"


@dataclass(frozen=True)
class ExportTarget:
    mime_type: str
    extension: str


EXPORT_TARGETS = {
    GOOGLE_DOC_MIME_TYPE: {
        "zip": ExportTarget(mime_type="application/zip", extension=".zip"),
    },
    GOOGLE_SHEET_MIME_TYPE: {
        "csv": ExportTarget(mime_type="text/csv", extension=".csv"),
    },
}

FORMAT_ALIASES = {
    "document_zip": "zip",
    "spreadsheet_csv": "csv",
}

app = typer.Typer(help="Fetch configured resources from Google Drive.")


def _configured_resources() -> list[dict[str, Any]]:
    resources = DRIVE_CONFIG.get("resources", [])
    if not isinstance(resources, list):
        raise ValueError("tool.coursedata.drive.resources must be a list of tables.")
    return resources


def _normalize_format(requested_format: str) -> str:
    normalized = requested_format.strip().lower().replace("-", "_")
    return FORMAT_ALIASES.get(normalized, normalized)


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "drive-file"


def _resolve_output_path(raw_path: str, default_base: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = default_base / path
    return path


def _select_export_target(mime_type: str, requested_format: str) -> ExportTarget:
    normalized_format = _normalize_format(requested_format)
    supported_formats = EXPORT_TARGETS.get(mime_type, {})
    try:
        return supported_formats[normalized_format]
    except KeyError as exc:
        supported = ", ".join(sorted(supported_formats)) or "none"
        raise ValueError(
            f"Unsupported download format '{requested_format}' for Drive mime type "
            f"'{mime_type}'. Supported formats: {supported}."
        ) from exc


def _default_token_path() -> Path:
    return Path(user_config_path("coursedata", appauthor="coursedata")) / "google-drive-token.json"


def _default_client_secret_path() -> Path:
    return Path(user_config_path("coursedata", appauthor="coursedata")) / "google-client-secret.json"


def _load_credentials() -> Any:
    if not GOOGLE_DRIVE_AVAILABLE:
        raise RuntimeError(
            "Google Drive dependencies are not installed. Add the project dependencies "
            "and reinstall the environment."
        )

    try:
        credentials, _ = google.auth.default(scopes=SCOPES)
        return credentials
    except DefaultCredentialsError:
        pass

    client_secret_raw = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET_PATH") or DRIVE_CONFIG.get("client_secret_path")
    token_raw = os.getenv("GOOGLE_DRIVE_TOKEN_PATH") or DRIVE_CONFIG.get("token_path")
    token_path = _resolve_output_path(token_raw, Path.cwd()) if token_raw else _default_token_path()
    client_secret_path = (
        _resolve_output_path(client_secret_raw, Path.cwd())
        if client_secret_raw
        else _default_client_secret_path()
    )

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    if not client_secret_path.exists():
        client_secret_hint = (
            "tool.coursedata.drive.client_secret_path / GOOGLE_DRIVE_CLIENT_SECRET_PATH"
            if client_secret_raw
            else f"{_default_client_secret_path()}"
        )
        raise RuntimeError(
            "Google Drive credentials not found. Configure application default credentials "
            f"or provide a client secret at {client_secret_hint}."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
    credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def _build_drive_service() -> Any:
    credentials = _load_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _resource_lookup_key(resource_cfg: dict[str, Any]) -> str:
    return str(resource_cfg.get("name") or resource_cfg.get("id") or "")


def _select_resources(resource_names: Optional[list[str]], *, strict: bool) -> list[dict[str, Any]]:
    resources = _configured_resources()
    if not resources:
        if strict:
            raise ValueError(
                "No Google Drive resources configured. Populate tool.coursedata.drive.resources "
                "in pyproject.toml."
            )
        logger.info("No Google Drive resources configured; skipping Drive download step.")
        return []

    if not resource_names:
        return resources

    requested = set(resource_names)
    selected = [
        resource_cfg
        for resource_cfg in resources
        if _resource_lookup_key(resource_cfg) in requested or str(resource_cfg.get("id")) in requested
    ]
    found = {
        _resource_lookup_key(resource_cfg)
        for resource_cfg in selected
        if _resource_lookup_key(resource_cfg)
    }
    found.update(str(resource_cfg.get("id")) for resource_cfg in selected if resource_cfg.get("id"))
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Unknown Google Drive resources requested: {', '.join(missing)}")
    return selected


def _metadata_for_resource(service: Any, file_id: str) -> dict[str, Any]:
    return service.files().get(
        fileId=file_id,
        fields="id,name,mimeType",
        supportsAllDrives=True,
    ).execute()


def _output_path_for_resource(
    resource_cfg: dict[str, Any],
    metadata: dict[str, Any],
    export_target: ExportTarget,
    output_dir: Optional[Path],
) -> Path:
    base_name = _safe_filename(str(resource_cfg.get("name") or metadata["name"] or metadata["id"]))
    generated_name = f"{base_name}{export_target.extension}"

    raw_output = resource_cfg.get("output")
    if raw_output:
        configured_path = _resolve_output_path(str(raw_output), RAW_DATA_DIR)
        if configured_path.suffix:
            if configured_path.suffix.lower() != export_target.extension:
                raise ValueError(
                    f"Configured output '{configured_path}' must end with {export_target.extension} "
                    f"for format '{resource_cfg.get('format')}'."
                )
            return configured_path
        return configured_path / generated_name

    base_dir = output_dir or RAW_DATA_DIR / "drive" / d8
    return base_dir / generated_name


def _download_resource(
    service: Any,
    resource_cfg: dict[str, Any],
    output_dir: Optional[Path],
    *,
    clean: bool,
) -> Path:
    file_id = str(resource_cfg.get("id", "")).strip()
    requested_format = str(resource_cfg.get("format", "")).strip()
    if not file_id or not requested_format:
        raise ValueError("Each Google Drive resource must define both 'id' and 'format'.")

    metadata = _metadata_for_resource(service, file_id)
    export_target = _select_export_target(str(metadata["mimeType"]), requested_format)
    output_path = _output_path_for_resource(resource_cfg, metadata, export_target, output_dir)

    if clean and output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = service.files().export_media(
        fileId=file_id,
        mimeType=export_target.mime_type,
    ).execute()
    output_path.write_bytes(content)
    logger.success(f"Saved Google Drive resource '{metadata['name']}' to {output_path}")
    return output_path


def _download_configured_resources(
    resource_names: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    clean: bool = False,
    *,
    strict: bool,
) -> list[Path]:
    selected_resources = _select_resources(resource_names, strict=strict)
    if not selected_resources:
        return []

    service = _build_drive_service()
    downloads: list[Path] = []
    try:
        for resource_cfg in selected_resources:
            downloads.append(
                _download_resource(service, resource_cfg, output_dir=output_dir, clean=clean)
            )
    except HttpError as exc:
        raise RuntimeError(f"Google Drive API request failed: {exc}") from exc
    return downloads


def run_all(headless: bool = False) -> None:
    """Download all configured Google Drive resources."""
    del headless
    _download_configured_resources(strict=False)


@app.callback(invoke_without_command=True)
def drive_callback(
    ctx: typer.Context,
    resource: Annotated[
        Optional[list[str]],
        typer.Option(
            "--resource",
            "-r",
            help="Configured resource name or Drive file ID to download. Defaults to all configured resources.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Override the default output directory for resources without an explicit configured output path."
        ),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove each target file before downloading it."),
    ] = False,
) -> None:
    """Download configured Google Drive resources."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        _download_configured_resources(
            resource_names=resource,
            output_dir=output_dir,
            clean=clean,
            strict=True,
        )
    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc