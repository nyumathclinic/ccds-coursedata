from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

def _find_project_root() -> Path:
    """Locate the project root containing pyproject.toml with [tool.coursedata]."""

    package_root = Path(__file__).resolve().parents[1]

    # Prefer the current working tree so submodule package code can be reused by parent repos.
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists():
            return candidate

    # Fallback for running inside the generic package repository directly.
    return package_root


# Load configuration from pyproject.toml
PROJ_ROOT = _find_project_root()

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

_config = {}
if tomllib:
    with open(PROJ_ROOT / "pyproject.toml", "rb") as f:
        _toml_data = tomllib.load(f)
        _config = _toml_data.get("tool", {}).get("coursedata", {})

# Optional subsections
LECTURE_COVERS_CONFIG = _config.get("lecture_covers", {})
GRADESCOPE_CONFIG = _config.get("gradescope", {})
BRIGHTSPACE_CONFIG = _config.get("brightspace", {})
ALBERT_CONFIG = _config.get("albert", {})
EDSTEM_CONFIG = _config.get("edstem", {})
DRIVE_CONFIG = _config.get("drive", {})
ENGAGEMENT_CONFIG = _config.get("engagement", {})
PROGRESS_REPORT_CONFIG = _config.get("progress_report", {})
SECTIONS_DASHBOARD_CONFIG = _config.get("sections_dashboard", {})
DAILY_CONFIG = _config.get("daily", {})

# Course Information
try:
    COURSE_NAME = _config["course_name"]
    TERM_NAME = _config["term_name"]
except KeyError as e:
    raise KeyError(
        f"Missing required config key {e} in [tool.coursedata] section of pyproject.toml"
    ) from e


# Paths
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
