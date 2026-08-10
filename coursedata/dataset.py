"""Backward-compatible entry point for the dataset CLI.

.. deprecated::
   Running ``python dataset.py`` is deprecated.
   Use ``python -m dataset`` instead.

All commands from the old flat CLI are preserved but print a deprecation warning
directing users to the new command-group interface::

    dataset.py get albert rosters
    dataset.py process enrollment
    dataset.py report enrollment
"""

import warnings

warnings.warn(
    "Running 'python dataset.py' directly is deprecated. "
    "Use 'python -m dataset' instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Re-export the app from the new package so this file can still be used as an
# entry point (e.g. python dataset.py <command>).
from dataset import app  # noqa: E402

if __name__ == "__main__":
    app()
