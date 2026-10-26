"""Loading, schema contracts and the synthetic fallback."""

from .catalog import (
    DataCatalog, describe_schema, directory_census, discover_files,
    extension_census, from_parquet_cache, load_any, load_run_tree, read_zip_member,
    to_parquet_cache,
)
from .contracts import CHAIN_PANEL, SMILE_PANEL, SchemaSpec, ValidationReport
from .quality import (
    HealthReport, categorical_report, datetime_audit, drift_report, health_report,
    leakage_scan, numeric_report, outlier_report, population_stability_index,
    target_audit,
)
from .panel import (
    PanelAudit, audit_panel, concat_folders, find_dataset_root, leakage_audit,
    load_panel, read_if_exists, synthetic_equity_panel,
)
from .loaders import (
    available, describe_provenance, load_chain_panel, load_smile_panel, provenance,
)
from .synthetic import fbm_variance_paths, rbergomi_panel, synthetic_smile_panel

__all__ = [
    "available", "load_smile_panel", "load_chain_panel", "provenance",
    "describe_provenance", "SchemaSpec", "ValidationReport", "SMILE_PANEL",
    "CHAIN_PANEL", "synthetic_smile_panel", "rbergomi_panel", "fbm_variance_paths",
    "load_panel", "synthetic_equity_panel", "audit_panel", "PanelAudit",
    "leakage_audit", "find_dataset_root", "read_if_exists", "concat_folders",
    "DataCatalog", "discover_files", "extension_census", "directory_census",
    "load_any", "load_run_tree", "read_zip_member", "describe_schema",
    "to_parquet_cache", "from_parquet_cache",
    "health_report", "HealthReport", "numeric_report", "outlier_report",
    "datetime_audit", "categorical_report", "drift_report", "leakage_scan",
    "target_audit", "population_stability_index",
]
