"""Vendored from github.com/zachana-1/Synthetic-Test-Data-Creator (src/stdc), the author's own project.

Kept as close to the original as possible so validation results stay comparable. Local changes are
marked "# local:". Databricks-specific modules (spark_io, pipeline, cli) are not included.
"""


__version__ = "0.1.0"

# Bump whenever a check or repair rule in validation.py changes. It is written
# to every result row so scores can be traced to the script that produced them.
VALIDATION_SCRIPT_VERSION = "3.0.0"  # local: 2.0.0 + history-preserving trims
