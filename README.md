# coursedata

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Manage and report on data for a course in a semester

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         coursedata and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── coursedata   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes coursedata a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Installation

Create a course-semester ccds project and import this as a submodule.

### Add as a submodule
1) From the parent repo root:
```bash
git submodule add https://github.com/leingang/ccds-coursedata.git generic
git submodule update --init --recursive
```
2) Commit the submodule pointer in the parent:
```bash
git add coursedata
git commit -m "Add coursedata submodule"
git push
```

### Installation

In the top (course-semester) repository, run:

```bash
uv pip install -e generic
uv pip install ../python-edubag
uv run python -m playwright install
```

## Usage

### Make changes in coursedata and publish them
All changes to the submodule live in its own repository. From inside `generic/`:
```bash
cd coursedata
git status
# edit files
git add .
git commit -m "Describe coursedata change"
git push origin main
cd ..
```
Then update the parent repo to record the new coursedata commit:
```bash
git add coursedata
git commit -m "Update coursedata submodule"
git push
```

This is easy to do in VSCode. The submodule appears in the source control tab browser pane,
and you can add, commit, push, and update the parent repo in that view.

### Pull latest coursedata in another parent repo
From the parent repo root:
```bash
git submodule update --remote coursedata
# or: cd coursedata && git pull && cd ..
git add coursedata
git commit -m "Update coursedata submodule"
git push
```

Again, this is easy to do in VSCode. The source control pane will know when the submodule 
has changes to be pulled in.


### Cloning a repo that already uses this submodule
```bash
git clone --recurse-submodules <PARENT_REPO_URL>
# if already cloned:
git submodule update --init --recursive
```

### Notes
- Always push the submodule changes first, then commit the updated pointer in each parent repo.
- `git status` in a parent repo will show coursedata as `modified (new commits)` when the pointer needs committing.
- To make submodules participate in common commands, you can enable: `git config --global submodule.recurse true`.

### Using the VS Code UI
- In the Source Control panel, submodules appear as nested repositories. Open `coursedata` in the panel to stage/commit/push submodule changes, then switch back to the parent repo entry and stage the `coursedata` folder to record the new pointer.
- To pull latest submodule changes: Command Palette → `Git: Update Submodules` (or right-click the submodule in Source Control and pull), then commit the updated `coursedata` pointer in the parent repo entry.
- When cloning, use `Git: Clone (Recursive)` or run `git clone --recurse-submodules`; if already cloned, run `Git: Update Submodules` to initialize.


--------

