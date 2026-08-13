Sections Dashboard Workflow
===========================

This workflow turns Albert class-details snapshots into two products:

1. Canonical sections data in `data/processed/sections`.
2. A report bundle in `reports/sections` with:
   - `index.html` (filterable dashboard)
   - `summary.md` (snapshot totals)

The workflow is implemented in `coursedata.dataset` and is designed to be
rerun as data updates arrive.

Configuration
-------------

Configure from the parent repository `pyproject.toml` under
`[tool.coursedata.sections_dashboard]`:

- `title`: Dashboard title for report output.
- `input_path`: Path to Albert class-details snapshots.
- `field_map`: Candidate JSON keys for each canonical column.

Input discovery supports both layouts under `data/raw/albert/class_details`:

- Dated folders containing `class_details.json`.
- Date-stamped JSON files directly in the folder.

The pipeline automatically chooses the most recent snapshot.

Example:

```toml
[tool.coursedata.sections_dashboard]
title = "Calculus II Sections Dashboard"
input_path = "albert/class_details"

[tool.coursedata.sections_dashboard.field_map]
section = ["section", "section_number", "class_section"]
instructors = ["instructor", "instructors", "instructor_name"]
```

Run The Pipeline
----------------

Build canonical sections files:

```bash
python -m coursedata.dataset process sections
```

Generate dashboard report artifacts:

```bash
python -m coursedata.dataset report sections
```

Run with all daily tasks:

```bash
make daily
```

Outputs
-------

Processing writes:

- `data/processed/sections/<YYYY-MM-DD>/sections.csv`
- `data/processed/sections/<YYYY-MM-DD>/sections.json`
- `data/processed/sections/latest/sections.csv`
- `data/processed/sections/latest/sections.json`

Reporting writes:

- `reports/sections/<snapshot>/index.html`
- `reports/sections/<snapshot>/summary.md`

Canonical Schema
----------------

Canonical columns are:

- `section`
- `instructors`
- `location`
- `meeting_times`
- `capacity`
- `enrolled`
- `waitlist`
- `status`
- `snapshot_date`

If upstream JSON keys change, update `field_map` in `pyproject.toml`
instead of changing code.

Publishing Guidance
-------------------

For static publishing, use the canonical JSON as source-of-truth and render a
site page from it during docs build.

Suggested pattern:

1. Run `python -m coursedata.dataset process sections` in CI.
2. Copy `data/processed/sections/latest/sections.json` into docs assets.
3. Build and publish MkDocs via GitHub Pages.

This keeps processing reproducible and presentation decoupled from raw data fetch.
