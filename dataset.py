from datetime import date
from pathlib import Path
import os
import shutil
import subprocess
from typing import Annotated, Optional
import keyring

try:
    from edubag.albert import xls2csv
    from edubag.albert.client import AlbertClient
    from edubag.gmail import filter_from_roster_command
    from edubag.gradescope.client import GradescopeClient
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False

try:
    from edubag.edstem.client import EdstemClient
    EDSTEM_AVAILABLE = True
except ImportError:
    EDSTEM_AVAILABLE = False

# Brightspace client availability is independent of Albert
try:
    from edubag.brightspace.client import BrightspaceClient
    BRIGHTSPACE_AVAILABLE = True
except ImportError:
    BRIGHTSPACE_AVAILABLE = False

from loguru import logger
from tqdm import tqdm
import typer

from coursedata.config import (
    COURSE_NAME,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TERM_NAME,
    GRADESCOPE_CONFIG,
    BRIGHTSPACE_CONFIG,
    EDSTEM_CONFIG,
)
from coursedata.enrollment import (
    find_roster_files,
    generate_enrollment_report,
    generate_enrollment_roster,
)

d8 = date.today().isoformat()


def get_password(service: str, username: str) -> Optional[str]:
    """
    Get password from macOS Keychain.
    
    First tries to retrieve as an internet password (more common for web services),
    then falls back to generic password if not found.
    """
    # Try internet password first
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
    
    # Fall back to generic password
    return keyring.get_password(service, username)


app = typer.Typer()


@app.command()
def daily():
    """Run all daily data processing steps."""
    albert_rosters()
    albert_class_details()
    gradescope_rosters()
    edstem_analytics()
    save_gmail_filters()
    enrollment_rosters()
    enrollment_reports()


@app.command()
def brightspace_gradebooks(
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for Brightspace gradebooks")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing files in output directories before fetching"
        ),
    ] = False,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging)",
        ),
    ] = False,
):
    """
    Fetch Brightspace gradebooks for configured courses and save to output_dir.
    """
    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag brightspace client is not available. Cannot fetch gradebooks.")
        raise typer.Exit(code=1)

    course_ids = BRIGHTSPACE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Brightspace course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "brightspace" / "gradebooks" / d8

    # Clean output directory if requested
    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Brightspace gradebooks for courses {course_ids} to '{output_dir}'"
    )

    # Get credentials from environment and keychain (same as Albert)
    username = os.getenv("SSO_USERNAME")
    if not username:
        logger.warning(
            "SSO_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("nyu-sso", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s nyu-sso -a {username} -w YOUR_PASSWORD"
            )
            password = None

    # Authenticate once for the session
    try:
        client = BrightspaceClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    # Download gradebooks per course
    for course in course_ids:
        try:
            client.save_gradebook(course, save_dir=output_dir, headless=headless)
        except Exception as e:
            logger.error(f"Failed to fetch gradebook for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Brightspace gradebooks fetched successfully.")


@app.command()
def brightspace_attendance(
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for Brightspace attendance files")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing files in output directories before fetching"
        ),
    ] = False,
):
    """
    Fetch Brightspace attendance files for configured courses and save to output_dir.
    """
    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag brightspace client is not available. Cannot fetch attendance.")
        raise typer.Exit(code=1)

    course_ids = BRIGHTSPACE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Brightspace course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "brightspace" / "attendance" / d8

    # Clean output directory if requested
    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Brightspace attendance for courses {course_ids} to '{output_dir}'"
    )

    # Get credentials from environment and keychain (same as Albert)
    username = os.getenv("SSO_USERNAME")
    if not username:
        logger.warning(
            "SSO_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("nyu-sso", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s nyu-sso -a {username} -w YOUR_PASSWORD"
            )
            password = None

    # Authenticate once for the session
    try:
        client = BrightspaceClient()
        client.authenticate(username=username, password=password, headless=True)
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    # Download attendance per course
    for course in course_ids:
        try:
            client.save_attendance(course, save_dir=output_dir, headless=True)
        except Exception as e:
            logger.error(f"Failed to fetch attendance for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Brightspace attendance fetched successfully.")


@app.command()
def albert_rosters(
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for the rosters file")
    ] = None,
    convert_to_csv: Annotated[
        bool, typer.Option(help="Convert the fetched Excel files to CSV format")
    ] = True,
    csv_output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for CSV files")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing files in output directories before fetching"
        ),
    ] = False,
):
    """
    Fetch all rosters for the specified course and term, and save to output_dir.
    """
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch rosters.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "albert" / "rosters" / d8

    # Determine csv_output_dir early if needed
    if csv_output_dir is None:
        csv_output_dir = INTERIM_DATA_DIR / "albert" / "rosters" / d8

    # Clean output directories if requested
    if clean:
        if output_dir.exists():
            logger.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        if convert_to_csv and csv_output_dir is not None and csv_output_dir.exists():
            logger.info(f"Cleaning CSV output directory: {csv_output_dir}")
            shutil.rmtree(csv_output_dir)
    logger.info(
        f"Fetching rosters for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output_dir}'"
    )

    # Get credentials from environment and keychain
    username = os.getenv("SSO_USERNAME")
    if not username:
        logger.warning(
            "SSO_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("nyu-sso", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s nyu-sso -a {username} -w YOUR_PASSWORD"
            )
            password = None

    client = AlbertClient()
    xls_path_list = client.fetch_and_save_rosters(
        COURSE_NAME, TERM_NAME, output_dir, username=username, password=password
    )
    logger.success("Rosters fetched successfully.")
    if convert_to_csv:
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Converting Excel files to CSV in '{csv_output_dir}'")
        for xls_path in tqdm(xls_path_list, desc="Converting to CSV"):
            xls2csv([xls_path], csv_output_dir)
        logger.success("Conversion to CSV complete.")


@app.command()
def albert_class_details(
    output: Annotated[
        Path | None, typer.Option(help="Output path for the class details file")
    ] = None,
):
    """
    Fetch all class details for the specified course and term, and save to output.
    """
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch class details.")
        raise typer.Exit(code=1)

    if output is None:
        output = RAW_DATA_DIR / "albert" / "class_details" / d8 / "class_details.json"

    logger.info(
        f"Fetching class details for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output}'"
    )

    # Get credentials from environment and keychain
    username = os.getenv("SSO_USERNAME")
    if not username:
        logger.warning(
            "SSO_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("nyu-sso", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s nyu-sso -a {username} -w YOUR_PASSWORD"
            )
            password = None

    client = AlbertClient()
    client.fetch_class_details(
        COURSE_NAME, TERM_NAME, output=output, username=username, password=password
    )
    logger.success("Class details fetched successfully.")


@app.command()
def gradescope_class_details(
    output: Annotated[
        Path | None, typer.Option(help="Output path for the class details file")
    ] = None,
):
    """
    Fetch all Gradescope class details for the specified course and term, and save to output.
    """
    if not EDUBAG_AVAILABLE:
        logger.error(
            "edubag module is not available. Cannot fetch Gradescope class details."
        )
        raise typer.Exit(code=1)

    if output is None:
        output = RAW_DATA_DIR / "gradescope" / "class_details" / "class_details.json"

    logger.info(
        f"Fetching Gradescope class details for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output}'"
    )

    # Get credentials from environment and keychain
    username = os.getenv("GRADESCOPE_USERNAME")
    if not username:
        logger.warning(
            "GRADESCOPE_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("gradescope.com", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s gradescope.com -a {username} -w YOUR_PASSWORD"
            )
            password = None

    client = GradescopeClient()
    client.fetch_class_details(
        COURSE_NAME, TERM_NAME, output=output, username=username, password=password
    )
    logger.success("Gradescope class details fetched successfully.")


@app.command()
def gradescope_rosters(
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for Gradescope rosters")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing files in output directories before fetching"
        ),
    ] = False,
):
    """
    Fetch Gradescope rosters for configured courses and save to output_dir.
    """
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch Gradescope rosters.")
        raise typer.Exit(code=1)

    course_ids = GRADESCOPE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Gradescope course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "gradescope" / "rosters" / d8

    # Clean output directory if requested
    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Gradescope rosters for courses {course_ids} to '{output_dir}'"
    )

    # Get credentials from environment and keychain (same pattern as Albert/Brightspace)
    username = os.getenv("GRADESCOPE_USERNAME")
    if not username:
        logger.warning(
            "GRADESCOPE_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("gradescope.com", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s gradescope.com -a {username} -w YOUR_PASSWORD"
            )
            password = None

    # Authenticate and download rosters
    try:
        client = GradescopeClient()
        client.authenticate(username=username, password=password, headless=True)
    except Exception as e:
        logger.error(f"Gradescope authentication failed: {e}")
        raise typer.Exit(code=1)

    # Download rosters per course
    for course in course_ids:
        try:
            client.save_roster(course, save_dir=output_dir, headless=True)
        except Exception as e:
            logger.error(f"Failed to fetch roster for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Gradescope rosters fetched successfully.")


@app.command()
def edstem_analytics(
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for EdStem analytics")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(
            help="Remove existing files in output directories before fetching"
        ),
    ] = False,
):
    """
    Fetch EdStem analytics for configured courses and save to output_dir.
    """
    if not EDSTEM_AVAILABLE:
        logger.error("edubag edstem client is not available. Cannot fetch EdStem analytics.")
        raise typer.Exit(code=1)

    course_ids = EDSTEM_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No EdStem course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "edstem" / "analytics" / d8

    # Clean output directory if requested
    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching EdStem analytics for courses {course_ids} to '{output_dir}'"
    )

    # Get credentials from environment and keychain
    username = os.getenv("EDSTEM_USERNAME")
    if not username:
        logger.warning(
            "EDSTEM_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("edstem.org", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. Store it with: security add-generic-password -s edstem.org -a {username} -w YOUR_PASSWORD"
            )
            password = None

    # Authenticate once for the session
    try:
        client = EdstemClient()
        client.authenticate(username=username, password=password, headless=True)
    except Exception as e:
        logger.error(f"EdStem authentication failed: {e}")
        raise typer.Exit(code=1)

    # Download analytics per course
    for course in course_ids:
        try:
            client.save_analytics(
                course,
                save_dir=output_dir,
                headless=True,
            )
        except Exception as e:
            logger.error(f"Failed to fetch analytics for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("EdStem analytics fetched successfully.")


@app.command()
def enrollment_rosters(
    rosters_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for enrollment rosters")
    ] = None,
):
    """
    Generate enrollment rosters for all sections.
    """
    if rosters_dir is None:
        rosters_dir = INTERIM_DATA_DIR / "albert" / "rosters"

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "enrollment"

    logger.info(f"Finding roster files in {rosters_dir}")
    sections = find_roster_files(rosters_dir)

    if not sections:
        logger.warning(f"No roster files found in {rosters_dir}")
        return

    logger.info(f"Found {len(sections)} sections")

    for section_name, roster_files in sections.items():
        logger.info(f"Processing section: {section_name}")
        generate_enrollment_roster(section_name, roster_files, output_dir)

    logger.success("Enrollment roster generation complete")


@app.command()
def enrollment_reports(
    rosters_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for enrollment reports")
    ] = None,
):
    """
    Generate enrollment reports for all sections.
    """
    if rosters_dir is None:
        rosters_dir = INTERIM_DATA_DIR / "albert" / "rosters"

    if output_dir is None:
        output_dir = REPORTS_DIR / "enrollment"

    logger.info(f"Finding roster files in {rosters_dir}")
    sections = find_roster_files(rosters_dir)

    if not sections:
        logger.warning(f"No roster files found in {rosters_dir}")
        return

    logger.info(f"Found {len(sections)} sections")

    for section_name, roster_files in sections.items():
        logger.info(f"Processing section: {section_name}")
        generate_enrollment_report(section_name, roster_files, output_dir)

    logger.success("Enrollment report generation complete")


@app.command("gmail-filters")
def save_gmail_filters(
    roster_paths: Annotated[
        Optional[list[Path]],
        typer.Option(
            help="One or more Albert roster XLS files. If not set, the most recently downloaded rosters will be used."
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to save the Gmail filter XML file. If not set, save to processed data directory."
        ),
    ] = None,
):
    """Generate Gmail filters XML from a Gradescope roster CSV file."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot generate Gmail filters.")
        raise typer.Exit(code=1)

    if not roster_paths:
        logger.info("No roster files provided. Using most recently downloaded rosters.")
        rosters_base_dir = RAW_DATA_DIR / "albert" / "rosters"

        # Find all date subdirectories and get the most recent one
        date_dirs = sorted([d for d in rosters_base_dir.iterdir() if d.is_dir()])
        if not date_dirs:
            logger.error(f"No roster directories found in {rosters_base_dir}")
            raise typer.Exit(code=1)

        latest_date_dir = date_dirs[-1]
        logger.info(f"Using rosters from {latest_date_dir.name}")

        # Find all .XLS files in the latest date directory
        roster_paths = sorted(latest_date_dir.glob("*.XLS"))

        if not roster_paths:
            logger.error(f"No .XLS files found in {latest_date_dir}")
            raise typer.Exit(code=1)

    if not output:
        output = PROCESSED_DATA_DIR / "gmail" / "gmail_filters.xml"

    filter_from_roster_command(roster_paths, output=output)
    logger.success(f"Gmail filters saved to {output}")


if __name__ == "__main__":
    app()
