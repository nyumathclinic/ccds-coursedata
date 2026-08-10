"""Enrollment roster and report generation."""

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from loguru import logger
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def find_roster_files(rosters_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """
    Find all roster files grouped by section.

    Args:
        rosters_dir: Base directory containing dated subdirectories with roster CSV files

    Returns:
        Dictionary mapping section names to list of (date, filepath) tuples, sorted by date
    """
    sections = defaultdict(list)

    # Find all CSV files in dated subdirectories
    for date_dir in sorted(rosters_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name

        for csv_file in sorted(date_dir.glob("*.csv")):
            # Extract section identifier from filename (everything before .csv)
            section_name = csv_file.stem
            sections[section_name].append((date_str, csv_file))

    # Sort each section's files by date
    for section_name in sections:
        sections[section_name].sort(key=lambda x: x[0])

    return dict(sections)


def format_date_friendly(date_str: str) -> str:
    """
    Convert ISO date string to human-friendly format.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Human-friendly date string like "January 13, 2026"
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except ValueError:
        return date_str


def generate_enrollment_roster(
    section_name: str,
    roster_files: list[tuple[str, Path]],
    output_dir: Path,
) -> Path | None:
    """
    Generate an enrollment roster with enrollment dates for a section.

    Args:
        section_name: Name of the section (e.g., "MATH-UA_122_001_1264")
        roster_files: List of (date, filepath) tuples for this section, sorted by date
        output_dir: Directory to save the enrollment roster

    Returns:
        Path to the generated enrollment roster CSV file or None if no files
    """
    if not roster_files:
        logger.warning(f"No roster files found for section {section_name}")
        return None

    # Track when each student first appeared
    enrollment_dates = {}
    # Track the first date a student disappeared from the roster
    dropped_dates = {}
    # Keep the most recent row data per student for reuse when they drop
    student_records: dict[str, pd.Series] = {}

    previous_students: set[str] = set()

    # Process each roster file chronologically
    for date_str, csv_file in roster_files:
        try:
            df = pd.read_csv(csv_file)

            # Track new enrollments
            for _, row in df.iterrows():
                campus_id = str(row["Campus ID"])
                student_records[campus_id] = row

                # Only record enrollment date if student has "Enrolled" status
                # and hasn't been tracked yet
                if campus_id not in enrollment_dates and row["Status"] == "Enrolled":
                    enrollment_dates[campus_id] = date_str
            # Detect drops by comparing previous roster to current roster
            current_students = set(str(cid) for cid in df["Campus ID"].tolist())
            dropped_now = previous_students - current_students
            for campus_id in dropped_now:
                if campus_id not in dropped_dates:
                    dropped_dates[campus_id] = date_str
            previous_students = current_students
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            continue

    # Read the most recent roster
    most_recent_date, most_recent_file = roster_files[-1]
    try:
        df = pd.read_csv(most_recent_file)

        # Add enrollment and dropped date columns
        df["Enrollment Date"] = df["Campus ID"].astype(str).map(enrollment_dates)
        df["Dropped Date"] = df["Campus ID"].astype(str).map(dropped_dates)

        current_ids = set(df["Campus ID"].astype(str))
        all_ids = set(student_records.keys())
        missing_ids = all_ids - current_ids

        if missing_ids:
            # Build rows for dropped students who are not in the most recent roster
            extra_rows = []
            for campus_id in missing_ids:
                base = student_records.get(campus_id, {})
                row_dict = {col: base.get(col, None) for col in df.columns}
                row_dict["Campus ID"] = campus_id
                row_dict["Status"] = "Dropped"
                row_dict["Enrollment Date"] = enrollment_dates.get(campus_id)
                row_dict["Dropped Date"] = dropped_dates.get(campus_id)
                extra_rows.append(row_dict)
            if extra_rows:
                df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

        # Sort the final roster by last name, first name
        df.sort_values(by=["Last Name", "First Name"], inplace=True)
        # Save the enrollment roster in date subdirectory
        current_date = date.today().isoformat()
        date_output_dir = output_dir / current_date
        date_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = date_output_dir / f"{section_name}_enrollment.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Generated enrollment roster: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating enrollment roster for {section_name}: {e}")
        return None


def generate_enrollment_report(
    section_name: str,
    roster_files: list[tuple[str, Path]],
    output_dir: Path,
) -> Path | None:
    """
    Generate a chronological enrollment report for a section.

    Args:
        section_name: Name of the section (e.g., "MATH-UA_122_001_1264")
        roster_files: List of (date, filepath) tuples for this section, sorted by date
        output_dir: Directory to save the enrollment report

    Returns:
        Path to the generated enrollment report PDF file or None if no files/events
    """
    if not roster_files:
        logger.warning(f"No roster files found for section {section_name}")
        return None

    # Track student state across dates
    previous_students = {}  # campus_id -> (first_name, last_name, email)
    withdrawn_students_set = set()  # campus_ids that have been withdrawn

    # Collect events for each date
    events_by_date = []

    for date_str, csv_file in roster_files:
        try:
            df = pd.read_csv(csv_file)

            current_students = {}
            new_students = []
            withdrawn_students = []

            for _, row in df.iterrows():
                campus_id = str(row["Campus ID"])
                first_name = row["First Name"]
                last_name = row["Last Name"]
                email = row["Email Address"]
                status = row["Status"]
                status_notes = str(row.get("Status Notes", "")).strip()

                student_info = (first_name, last_name, email)
                current_students[campus_id] = student_info

                # Check for new enrollments
                if campus_id not in previous_students and status == "Enrolled":
                    new_students.append(student_info)

                # Check for withdrawn students (only report first time)
                if status_notes == "Withdrawn" and campus_id not in withdrawn_students_set:
                    withdrawn_students.append(student_info)
                    withdrawn_students_set.add(campus_id)

            # Check for dropped students (not appearing in current roster)
            dropped_students = []
            for campus_id, student_info in previous_students.items():
                if campus_id not in current_students:
                    dropped_students.append(student_info)

            # Sort all lists by last name, then first name
            new_students.sort(key=lambda x: (x[1], x[0]))
            dropped_students.sort(key=lambda x: (x[1], x[0]))
            withdrawn_students.sort(key=lambda x: (x[1], x[0]))

            # Record all dates, even if no events
            events_by_date.append(
                {
                    "date": date_str,
                    "new": new_students,
                    "dropped": dropped_students,
                    "withdrawn": withdrawn_students,
                }
            )

            previous_students = current_students

        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            continue

    # Generate PDF report
    if not events_by_date:
        logger.warning(f"No enrollment events found for section {section_name}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{section_name}_enrollment.pdf"

    try:
        doc = SimpleDocTemplate(str(output_file), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_text = f"Enrollment Report: {section_name.replace('_', ' ')}"
        story.append(Paragraph(title_text, styles["Title"]))
        story.append(Spacer(1, 12))

        # Add events for each date
        for event in events_by_date:
            date_str = event["date"]

            # Date header with human-friendly format
            friendly_date = format_date_friendly(date_str)
            story.append(Paragraph(f"<b>{friendly_date}</b>", styles["Heading2"]))
            story.append(Spacer(1, 6))

            # Check if there are any changes
            has_changes = event["new"] or event["dropped"] or event["withdrawn"]

            if not has_changes:
                story.append(Paragraph("No changes", styles["Normal"]))
                story.append(Spacer(1, 6))
            else:
                # New students
                if event["new"]:
                    story.append(Paragraph("<b>New Students:</b>", styles["Heading3"]))
                    for first, last, email in event["new"]:
                        story.append(
                            Paragraph(f"• {first} {last} &lt;{email}&gt;", styles["Normal"])
                        )
                    story.append(Spacer(1, 6))

                # Dropped students
                if event["dropped"]:
                    story.append(Paragraph("<b>Dropped Students:</b>", styles["Heading3"]))
                    for first, last, email in event["dropped"]:
                        story.append(
                            Paragraph(f"• {first} {last} &lt;{email}&gt;", styles["Normal"])
                        )
                    story.append(Spacer(1, 6))

                # Withdrawn students
                if event["withdrawn"]:
                    story.append(Paragraph("<b>Withdrawn Students:</b>", styles["Heading3"]))
                    for first, last, email in event["withdrawn"]:
                        story.append(
                            Paragraph(f"• {first} {last} &lt;{email}&gt;", styles["Normal"])
                        )
                    story.append(Spacer(1, 6))

            story.append(Spacer(1, 12))

        doc.build(story)
        logger.info(f"Generated enrollment report: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error generating enrollment report for {section_name}: {e}")
        return None
