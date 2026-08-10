"""
Generate lecture cover PDFs from a CSV schedule file.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import re
from datetime import datetime
import shutil

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import click
from typing import Optional, Annotated, Generator

from coursedata.config import (
    PROJ_ROOT,
    REPORTS_DIR,
    LECTURE_COVERS_CONFIG,
    TERM_NAME,
    PROCESSED_DATA_DIR,
)


app = typer.Typer()


class MeetingPattern(Enum):
    """Enumeration for lecture meeting patterns."""
    MW = "Monday/Wednesday"
    TR = "Tuesday/Thursday" 


DEFAULT_SOURCE_TYPE = "mpl"


@dataclass
class LectureCoversSettings:
    source: Path
    source_type: str
    sections: list[str] | None
    output: Path
    meeting_pattern: MeetingPattern | None = None


class LectureScheduleParser(ABC):
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    @abstractmethod
    def parse(self) -> Generator[tuple[datetime, int, str], None, None]:
        """Yield tuples of (date, number, topic)."""
        raise NotImplementedError


class MPLLectureScheduleParser(LectureScheduleParser):
    def parse(self) -> Generator[tuple[datetime, int, str], None, None]:
        def to_date(date_str: str) -> datetime | None:
            date_str = date_str.strip().replace("\xa0", " ")
            try:
                return datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError as e:
                logger.debug(f"Error parsing date '{date_str}': {e}")
                return None

        df = pd.read_csv(self.csv_path, header=0, dtype=str, converters={"Date": to_date})
        df.dropna(subset=["Date", "Topic", "Type"], inplace=True)
        df = df[df["Type"].str.lower() == "lecture"]
        for lecture_number, row in enumerate(df.iterrows(), start=1):
            lecture_date = datetime.fromisoformat(row[1]["Date"])
            lecture_topic = row[1]["Topic"].strip()
            if pd.isna(lecture_date) or lecture_topic == "" or lecture_number is None:
                logger.debug(f"Skipping row due to missing data: {row}")
                continue
            yield (lecture_date, lecture_number, lecture_topic)


class JuliusLectureScheduleParser(LectureScheduleParser):
    def __init__(self, csv_path: Path, meeting_pattern: MeetingPattern | None = None):
        super().__init__(csv_path)
        self.meeting_pattern = meeting_pattern
        if self.meeting_pattern is None:
            raise ValueError(
                "meeting_pattern is required for JuliusLectureScheduleParser. "
                "Must be MeetingPattern.MW or MeetingPattern.TR."
            )
        if not isinstance(self.meeting_pattern, MeetingPattern):
            raise ValueError(
                f"Invalid meeting_pattern. Must be MeetingPattern enum value."
            )

    def parse(self) -> Generator[tuple[datetime, int, str], None, None]:
        # Column indices depend on meeting pattern
        # Date is always in column 2
        # TR (Tuesday/Thursday): Class # in column 7, Topic in column 8
        # MW (Monday/Wednesday): Class # in column 4, Topic in column 5
        date_col = "Date"
        if self.meeting_pattern == MeetingPattern.TR:
            class_num_col = "Class #.1"
            topic_col = "Topic.1"
        else:  # MW
            class_num_col = "Class #"
            topic_col = "Topic"
        
        def to_date(date_str: str) -> datetime | None:
            # Extract year from TERM_NAME (e.g., "Spring 2026" -> 2026)
            year_str = TERM_NAME.split()[-1]
            date_str = date_str.strip().replace("\xa0", " ")
            try:
                return datetime.strptime(date_str + f", {year_str}", "%B %d, %Y").replace(
                    hour=8, minute=0, second=0, microsecond=0
                )
            except ValueError as e:
                logger.debug(f"Error parsing date '{date_str}': {e}")
                return None

        df = pd.read_csv(self.csv_path, header=2, dtype=str)
        # Apply date conversion
        df[date_col] = df[date_col].apply(to_date)
        df.dropna(subset=[date_col, topic_col, class_num_col], inplace=True)
        
        for _, row in df.iterrows():
            lecture_date = row[date_col]
            lecture_topic = row[topic_col].strip()
            try:
                lecture_number = int(row[class_num_col].strip())
            except (ValueError, TypeError):
                logger.debug(f"Could not parse class number: {row[class_num_col]}")
                continue
            
            if pd.isna(lecture_date) or lecture_topic == "" or lecture_number is None:
                logger.debug(f"Skipping row due to missing data")
                continue
            yield (lecture_date, lecture_number, lecture_topic)


def _resolve_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJ_ROOT / path
    return resolved.resolve()


def get_meeting_pattern_for_section(section: str) -> MeetingPattern:
    """Get meeting pattern for a section from class_details.json.
    
    Args:
        section: Section number as a string (e.g., "011", "016")
    
    Returns:
        MeetingPattern.TR for Tuesday/Thursday sections, MeetingPattern.MW for Monday/Wednesday sections
    """
    # Find the most recent class_details.json file
    class_details_dir = PROCESSED_DATA_DIR / "albert" / "class_details"
    if not class_details_dir.exists():
        raise FileNotFoundError(f"Class details directory not found: {class_details_dir}")
    
    # Get the most recent date directory
    date_dirs = sorted([d for d in class_details_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        raise FileNotFoundError(f"No class details found in {class_details_dir}")
    
    class_details_file = date_dirs[0] / "class_details.json"
    if not class_details_file.exists():
        raise FileNotFoundError(f"Class details file not found: {class_details_file}")
    
    # Load class details and find matching section
    with open(class_details_file, 'r') as f:
        class_details = json.load(f)
    
    for class_detail in class_details:
        if class_detail.get("section") == section:
            days_and_times = class_detail.get("days_and_times", "")
            if days_and_times.startswith("TuTh"):
                return MeetingPattern.TR
            elif days_and_times.startswith("MoWe"):
                return MeetingPattern.MW
            else:
                raise ValueError(
                    f"Unexpected meeting pattern for section {section}: {days_and_times}"
                )
    
    raise ValueError(f"Section {section} not found in class details")


# Map of parser types to parser classes
PARSER_MAP: dict[str, type[LectureScheduleParser]] = {
    "mpl": MPLLectureScheduleParser,
    "julius": JuliusLectureScheduleParser,
}


def load_lecture_covers_settings(
    source: Optional[Path],
    source_type: Optional[str],
    sections: Optional[list[str]],
    output: Optional[Path],
) -> LectureCoversSettings:
    config_data = LECTURE_COVERS_CONFIG

    resolved_source = source or config_data.get("source")
    if resolved_source is None:
        raise ValueError("A source CSV path is required (pyproject.toml or CLI --source).")
    source_path = _resolve_path(Path(resolved_source))
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find schedule CSV at {source_path}")

    # Determine source_type - required parameter
    resolved_source_type = source_type or config_data.get("source_type")
    if not resolved_source_type:
        available_types = ", ".join(f"'{t}'" for t in PARSER_MAP.keys())
        raise ValueError(
            f"source_type is required. Specify it via --source-type option or in pyproject.toml "
            f"[tool.coursedata.lecture_covers] section. Options: {available_types}."
        )
    resolved_source_type = resolved_source_type.lower()
    
    resolved_sections = sections or config_data.get("sections")
    if resolved_sections is not None:
        resolved_sections = [str(section) for section in resolved_sections]

    # For julius parser, sections are required
    if resolved_source_type == "julius" and not resolved_sections:
        raise ValueError(
            "sections are required for julius parser. Specify via --sections option or in "
            "pyproject.toml [tool.coursedata.lecture_covers] section."
        )

    resolved_output = output or config_data.get("output")
    if resolved_output is None:
        resolved_output = REPORTS_DIR / "covers"
    resolved_output = _resolve_path(Path(resolved_output))

    # Load meeting pattern if configured
    resolved_meeting_pattern = None
    if "meeting_pattern" in config_data:
        pattern_str = config_data.get("meeting_pattern").upper()
        try:
            resolved_meeting_pattern = MeetingPattern[pattern_str]
        except KeyError:
            raise ValueError(
                f"Invalid meeting_pattern '{pattern_str}' in pyproject.toml. "
                f"Must be 'MW' or 'TR'."
            )

    return LectureCoversSettings(
        source=source_path,
        source_type=resolved_source_type,
        sections=resolved_sections,
        output=resolved_output,
        meeting_pattern=resolved_meeting_pattern,
    )


def get_parser(source_type: str, csv_path: Path, meeting_pattern: MeetingPattern | None = None) -> LectureScheduleParser:
    try:
        parser_cls = PARSER_MAP[source_type]
    except KeyError as exc:
        available_types = ", ".join(f"'{t}'" for t in PARSER_MAP.keys())
        raise ValueError(
            f"Unsupported source_type '{source_type}'. Choose from: {available_types}."
        ) from exc
    
    # JuliusLectureScheduleParser requires meeting pattern information
    if parser_cls == JuliusLectureScheduleParser:
        if meeting_pattern is None:
            raise ValueError(
                "meeting_pattern parameter is required for julius parser. Must be MeetingPattern.MW or MeetingPattern.TR."
            )
        return parser_cls(csv_path, meeting_pattern=meeting_pattern)
    
    return parser_cls(csv_path)


def get_pdf_filename(
    date: datetime, lecnum: int, topic: str, section: str | None = None
) -> str:
    """Generate a sanitized filename for the lecture PDF."""

    def sanitize_filename(text: str) -> str:
        """Sanitize text to be safe for filenames. Spaces are OK."""
        text = (
            text.strip()
            .replace("\xa0", " ")
            .replace(", ", " ")
            .replace(": ", " ")
            .replace("(", "")
            .replace(")", "")
        )
        return re.sub(r"[^A-Za-z0-9\-§ ]+", "_", text)

    iso_date = date.strftime("%Y-%m-%d")
    lecnum_fmt = f"Lec{lecnum:02d}"
    sanitized_topic = sanitize_filename(topic)
    if section is not None:
        filename = f"{iso_date} {section} {lecnum_fmt} {sanitized_topic}.pdf"
    else:
        filename = f"{iso_date} {lecnum_fmt} {sanitized_topic}.pdf"
    return filename


def get_lectures_from_mpl_csv(
    csv_path: Path,
) -> Generator[tuple[datetime, int, str], None, None]:
    """Wrapper to retain the previous functional interface."""

    yield from MPLLectureScheduleParser(csv_path).parse()


def get_lectures_from_julius_csv(
    csv_path: Path,
) -> Generator[tuple[datetime, int, str], None, None]:
    """Wrapper to retain the previous functional interface."""

    yield from JuliusLectureScheduleParser(csv_path).parse()


def make_pdf(date: datetime, lecnum: int, topic: str, output_path: Path) -> Path:
    """Generate a single lecture cover PDF."""
    # Register a font with broad Unicode support

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleH = styles["Heading1"]

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PDF size: 160mm × 90mm
    # ReportLab expects a filename or file-like object; convert Path to str
    doc = SimpleDocTemplate(str(output_path), pagesize=(160 * mm, 90 * mm))
    elements = []
    elements.append(Paragraph(f"Lecture {lecnum}: {topic}", styleH))
    elements.append(Spacer(1, 12))
    formatted_date = date.strftime("%B %-d, %Y")
    elements.append(Paragraph(f"Date: {formatted_date}", styleN))
    doc.build(elements)
    logger.info(f"Generated PDF: {output_path}")
    return output_path


@app.command()
def make_lecture_covers(
    source: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the schedule CSV file. Defaults to pyproject.toml config."),
    ] = None,
    source_type: Annotated[
        Optional[str],
        typer.Option(
            help="Parser to use for input CSV file.",
            click_type=click.Choice(list(PARSER_MAP.keys()), case_sensitive=False),
            show_default=False,
        ),
    ] = None,
    sections: Annotated[
        list[str] | None, typer.Option(help="Sections to generate covers for")
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output directory. Defaults to pyproject.toml config or reports directory.",
        ),
    ] = None,
):
    """
    Generate lecture cover PDFs from a CSV file into an output directory.
    """
    settings = load_lecture_covers_settings(
        source=source,
        source_type=source_type,
        sections=sections,
        output=output,
    )

    pdf_output_dir = settings.output
    if pdf_output_dir.exists():
        if pdf_output_dir.is_dir():
            shutil.rmtree(pdf_output_dir)
        else:
            pdf_output_dir.unlink()
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = []
    
    # For julius parser, iterate over sections; each section has its own meeting pattern
    if settings.source_type == "julius":
        for section in settings.sections:
            # Use configured meeting pattern if available, otherwise look it up from class details
            if settings.meeting_pattern is not None:
                meeting_pattern = settings.meeting_pattern
            else:
                meeting_pattern = get_meeting_pattern_for_section(section)
            
            parser = get_parser(settings.source_type, settings.source, meeting_pattern=meeting_pattern)
            for lecture_date, lecture_number, lecture_topic in tqdm(
                parser.parse(),
                desc=f"Generating lecture PDFs for section {section}",
            ):
                pdf_filename = get_pdf_filename(
                    lecture_date, lecture_number, lecture_topic, section=section
                )
                output_path = pdf_output_dir / pdf_filename
                pdf_file = make_pdf(
                    lecture_date, lecture_number, lecture_topic, output_path
                )
                pdf_files.append(pdf_file)
    else:
        # For other parsers, parse once and generate for all sections (or no sections)
        parser = get_parser(settings.source_type, settings.source)
        for lecture_date, lecture_number, lecture_topic in tqdm(
            parser.parse(),
            desc="Generating lecture PDFs",
        ):
            if settings.sections is not None:
                for section in settings.sections:
                    pdf_filename = get_pdf_filename(
                        lecture_date, lecture_number, lecture_topic, section=section
                    )
                    output_path = pdf_output_dir / pdf_filename
                    pdf_file = make_pdf(
                        lecture_date, lecture_number, lecture_topic, output_path
                    )
                    pdf_files.append(pdf_file)
            else:
                pdf_filename = get_pdf_filename(
                    lecture_date, lecture_number, lecture_topic
                )
                output_path = pdf_output_dir / pdf_filename
                pdf_file = make_pdf(
                    lecture_date, lecture_number, lecture_topic, output_path
                )
                pdf_files.append(pdf_file)

    logger.info(f"PDFs written to: {settings.output}")
    logger.info(f"Generated {len(pdf_files)} lecture PDFs.")


if __name__ == "__main__":
    app()
