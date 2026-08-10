from datetime import date
from pathlib import Path
from typing import Annotated, Optional
import json
import os
import re
import shutil
import subprocess
import time
from urllib.parse import urlparse

import keyring
from loguru import logger
import typer

try:
    from edubag.gradescope.client import GradescopeClient
    try:
        from edubag.gradescope import add_sections_to_roster_from_brightspace
        GRADESCOPE_SECTIONS_AVAILABLE = True
    except ImportError:
        GRADESCOPE_SECTIONS_AVAILABLE = False
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False
    GRADESCOPE_SECTIONS_AVAILABLE = False

try:
    from edubag.brightspace.client import BrightspaceClient
    BRIGHTSPACE_AVAILABLE = True
except ImportError:
    BRIGHTSPACE_AVAILABLE = False

from coursedata.config import (
    COURSE_NAME,
    TERM_NAME,
    GRADESCOPE_CONFIG,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

TODAY = date.today().isoformat()


def _get_password(service: str, username: str) -> Optional[str]:
    """
    Get password from macOS Keychain.

    First tries to retrieve as an internet password (more common for web services),
    then falls back to generic password if not found.
    """
    try:
        result = subprocess.run(
            ["security", "find-internet-password", "-s", service, "-a", username, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return keyring.get_password(service, username)


def _normalize_course_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value

    parsed = urlparse(value)
    text = parsed.path or value
    match = re.findall(r"\d+", text)
    if match:
        return match[-1]

    return value


def _call_with_headless(func, *args, headless: bool = False, **kwargs):
    try:
        return func(*args, headless=headless, **kwargs)
    except TypeError:
        return func(*args, **kwargs)


def _add_sections_to_roster(roster_path: Path, gradebook_path: Path, output_path: Path) -> Path:
    """Add section information to roster from Brightspace gradebook.
    
    Args:
        roster_path: Path to Gradescope roster CSV
        gradebook_path: Path to Brightspace gradebook CSV
        output_path: Path where output should be saved
        
    Returns:
        Path to the output file with sections added
    """
    result = add_sections_to_roster_from_brightspace(
        roster_csv=roster_path,
        brightspace_csv=gradebook_path,
        output_csv=output_path,
    )
    # The function returns the output path or None
    if result is None:
        return output_path
    # Handle case where result is a list (extract first element)
    if isinstance(result, list):
        return Path(result[0]) if result else output_path
    return Path(result)


def _find_latest_gradebook(save_dir: Path) -> Optional[Path]:
    if not save_dir.exists():
        return None
    candidates = sorted(save_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_latest_gradebook_anywhere() -> Optional[Path]:
    base_dir = RAW_DATA_DIR / "brightspace" / "gradebooks"
    if not base_dir.exists():
        return None
    candidates = sorted(
        base_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


@app.command("sync-gradescope-rosters")
def sync_gradescope_rosters(
    courses: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Overrides the course IDs from pyproject.toml; if omitted, uses tool.coursedata.gradescope.courses",
        ),
    ] = None,
    username: Annotated[
        Optional[str],
        typer.Option(help="Gradescope username; defaults to $GRADESCOPE_USERNAME from environment"),
    ] = None,
    keyring_service: Annotated[
        str,
        typer.Option(help="Keyring service name for Gradescope password storage"),
    ] = "gradescope.com",
):
    """Sync Gradescope rosters for configured courses.

    Reads course IDs from [tool.coursedata.gradescope] in pyproject.toml unless overridden.
    Auth uses $GRADESCOPE_USERNAME and password from the specified keyring service.
    """
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot sync Gradescope rosters.")
        raise typer.Exit(code=1)

    configured_courses = GRADESCOPE_CONFIG.get("courses", [])
    course_ids = courses or configured_courses
    if not course_ids:
        logger.error("No Gradescope courses configured. Set tool.coursedata.gradescope.courses in pyproject.toml or pass --courses.")
        raise typer.Exit(code=1)

    # Resolve credentials (do not pass to sync_roster; let edubag handle)
    if not username:
        username = os.getenv("GRADESCOPE_USERNAME")
    if not username:
        logger.warning(
            "GRADESCOPE_USERNAME not found in environment. Set it in .env or pass --username."
        )
    else:
        # Ensure the environment variable is set for any downstream use
        os.environ["GRADESCOPE_USERNAME"] = username
        # Check for password presence in keyring and guide setup if missing
        pw = keyring.get_password(keyring_service, username)
        if not pw:
            logger.warning(
                f"Password for '{username}' not found in Keychain (service '{keyring_service}'). Add it via:"
            )
            logger.warning(
                f"security add-generic-password -s {keyring_service} -a {username} -w YOUR_PASSWORD"
            )

    # Iterate and sync
    success = 0
    for cid in course_ids:
        try:
            logger.info(f"Syncing Gradescope roster for course {cid}...")
            # edubag handles auth internally via env/keyring; no kwargs
            client = GradescopeClient()
            client.sync_roster(cid)
            logger.success(f"Roster synced for course {cid}")
            success += 1
        except Exception as e:
            logger.error(f"Failed to sync roster for course {cid}: {e}")

    if success == 0:
        raise typer.Exit(code=1)


@app.command("sync-gradescope-sections")
def sync_gradescope_sections(
    gradescope_courses: Annotated[
        Optional[list[str]],
        typer.Option(
            "--gradescope-course",
            help="Gradescope course ID or URL (repeatable)",
        ),
    ] = None,
    brightspace_courses: Annotated[
        Optional[list[str]],
        typer.Option(
            "--brightspace-course",
            help="Brightspace course ID or URL (repeatable)",
        ),
    ] = None,
    load_details: Annotated[
        Optional[Path],
        typer.Option(
            "--load-details",
            help="Path to a class details JSON file",
        ),
    ] = None,
    fetch_details: Annotated[
        Optional[bool],
        typer.Option(
            "--fetch-details/--no-fetch-details",
            help="Fetch class details from Gradescope",
        ),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude",
            help="Gradescope course IDs to exclude from class details",
        ),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging)",
        ),
    ] = True,
):
    """Sync Gradescope rosters with Brightspace sections and upload."""
    if not EDUBAG_AVAILABLE or not GRADESCOPE_SECTIONS_AVAILABLE:
        logger.error("edubag Gradescope modules are not available. Cannot sync sections.")
        raise typer.Exit(code=1)

    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag Brightspace client is not available. Cannot sync sections.")
        raise typer.Exit(code=1)

    gradescope_courses = [
        _normalize_course_id(c) for c in (gradescope_courses or []) if c
    ]
    brightspace_courses = [
        _normalize_course_id(c) for c in (brightspace_courses or []) if c
    ]
    exclude_ids = set(_normalize_course_id(c) for c in (exclude or []) if c)

    if fetch_details is None and not gradescope_courses and not load_details:
        fetch_details = True

    if gradescope_courses or brightspace_courses:
        if not gradescope_courses or not brightspace_courses:
            logger.error(
                "Both --gradescope-course and --brightspace-course must be provided when specifying courses explicitly."
            )
            raise typer.Exit(code=1)
        if len(gradescope_courses) != len(brightspace_courses):
            logger.error("Gradescope and Brightspace course lists must be the same length.")
            raise typer.Exit(code=1)
        course_pairs = list(zip(gradescope_courses, brightspace_courses))
    else:
        details_path = load_details
        if fetch_details:
            details_path = RAW_DATA_DIR / "gradescope" / "class_details" / "class_details.json"
            details_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Fetching Gradescope class details to '{details_path}'"
            )
            username = os.getenv("GRADESCOPE_USERNAME")
            password = None
            if username:
                password = _get_password("gradescope.com", username)

            client = GradescopeClient()
            _call_with_headless(
                client.fetch_class_details,
                COURSE_NAME,
                TERM_NAME,
                output=details_path,
                username=username,
                password=password,
                headless=headless,
            )
        if details_path is None:
            logger.error("No course IDs specified and no class details source provided.")
            raise typer.Exit(code=1)
        if not details_path.exists():
            logger.error(f"Class details file not found: {details_path}")
            raise typer.Exit(code=1)

        with open(details_path, "r", encoding="utf-8") as f:
            details = json.load(f)

        course_pairs = []
        for entry in details:
            gs_course = entry.get("course_id")
            bs_course = entry.get("lms_course_id")
            if not gs_course or not bs_course:
                continue
            gs_course = _normalize_course_id(str(gs_course))
            bs_course = _normalize_course_id(str(bs_course))
            if gs_course in exclude_ids:
                continue
            course_pairs.append((gs_course, bs_course))

    if not course_pairs:
        logger.error("No course pairs found to sync.")
        raise typer.Exit(code=1)

    raw_rosters_dir = RAW_DATA_DIR / "gradescope" / "rosters" / TODAY
    raw_gradebooks_dir = RAW_DATA_DIR / "brightspace" / "gradebooks" / TODAY
    processed_dir = PROCESSED_DATA_DIR / "gradescope" / "rosters-with-sections" / TODAY
    raw_rosters_dir.mkdir(parents=True, exist_ok=True)
    raw_gradebooks_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    gs_username = os.getenv("GRADESCOPE_USERNAME")
    gs_password = None
    if gs_username:
        gs_password = _get_password("gradescope.com", gs_username)

    bs_username = os.getenv("SSO_USERNAME")
    bs_password = None
    if bs_username:
        bs_password = _get_password("nyu-sso", bs_username)

    gradescope_client = GradescopeClient()
    brightspace_client = BrightspaceClient()

    try:
        _call_with_headless(
            gradescope_client.authenticate,
            username=gs_username,
            password=gs_password,
            headless=headless,
        )
    except Exception as e:
        logger.error(f"Gradescope authentication failed: {e}")
        raise typer.Exit(code=1)

    try:
        _call_with_headless(
            brightspace_client.authenticate,
            username=bs_username,
            password=bs_password,
            headless=headless,
        )
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    for gs_course, bs_course in course_pairs:
        logger.info(
            f"Syncing sections for Gradescope {gs_course} with Brightspace {bs_course}..."
        )
        roster_paths = _call_with_headless(
            gradescope_client.save_roster,
            gs_course,
            save_dir=raw_rosters_dir,
            headless=headless,
        )
        if isinstance(roster_paths, list):
            if len(roster_paths) != 1:
                logger.error(
                    f"Expected 1 roster file for course {gs_course}, got {len(roster_paths)}."
                )
                raise typer.Exit(code=1)
            roster_path = Path(roster_paths[0])
        else:
            roster_path = Path(roster_paths)

        gradebook_paths = None
        last_error: Exception | None = None
        for attempt_headless in (headless, False):
            try:
                gradebook_paths = _call_with_headless(
                    brightspace_client.save_gradebook,
                    bs_course,
                    save_dir=raw_gradebooks_dir,
                    headless=attempt_headless,
                )
                break
            except Exception as e:
                last_error = e
                mode = "headless" if attempt_headless else "headed"
                logger.warning(
                    f"Brightspace gradebook download failed in {mode} mode; retrying. Error: {e}"
                )
                time.sleep(2)

        if gradebook_paths is None:
            fallback = _find_latest_gradebook(raw_gradebooks_dir)
            if fallback:
                logger.warning(
                    f"Using most recent gradebook from '{raw_gradebooks_dir}': {fallback.name}"
                )
                gradebook_paths = [fallback]
            else:
                fallback_any = _find_latest_gradebook_anywhere()
                if fallback_any:
                    logger.warning(
                        "Using most recent gradebook from earlier run: "
                        f"{fallback_any}"
                    )
                    gradebook_paths = [fallback_any]
                else:
                    raise typer.Exit(code=1) from last_error
        if isinstance(gradebook_paths, list):
            if len(gradebook_paths) != 1:
                logger.error(
                    f"Expected 1 gradebook file for course {bs_course}, got {len(gradebook_paths)}."
                )
                raise typer.Exit(code=1)
            gradebook_path = Path(gradebook_paths[0])
        else:
            gradebook_path = Path(gradebook_paths)

        output_path = processed_dir / roster_path.name
        roster_with_sections = _add_sections_to_roster(
            roster_path, gradebook_path, output_path
        )
        if roster_with_sections is None:
            roster_with_sections = output_path
        else:
            roster_with_sections = Path(roster_with_sections)
            if roster_with_sections.resolve() != output_path.resolve():
                try:
                    shutil.copy2(roster_with_sections, output_path)
                    roster_with_sections = output_path
                except Exception as e:
                    logger.error(
                        "Failed to copy roster-with-sections to processed directory: "
                        f"{output_path}. Error: {e}"
                    )
                    raise typer.Exit(code=1)

        _call_with_headless(
            gradescope_client.send_roster,
            gs_course,
            roster_with_sections,
            headless=headless,
        )
        logger.success(
            f"Roster with sections uploaded for Gradescope course {gs_course}."
        )


@app.command()
def daily():
    """Run all non-data local tasks for the day.

    Currently runs Gradescope roster sync; add additional tasks here as needed.
    """
    # Uses defaults from configuration and environment
    sync_gradescope_rosters()


if __name__ == "__main__":
    app()
