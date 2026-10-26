"""
rvlab.data.catalog
==================
Finding and loading every data file in a repository.

Real projects do not hand you one tidy CSV. This one mixes delimited tables,
pickles, NumPy arrays, JSON, spreadsheets and large archives across research
outputs, registries and nested walk-forward runs — and some of it is gitignored,
so what exists depends on the machine.

The job this module does, in order:

  1. **discover** — walk the tree, with size and directory guards so a 12.9 GB
     archive is catalogued rather than opened;
  2. **classify** — group by extension and directory so the shape of the data is
     visible before anything is read;
  3. **load** — dispatch on extension, with the right defaults for each format;
  4. **describe** — capture each file's schema so the catalogue is searchable
     without re-reading anything.

The recurring lesson is that *metadata often lives in the path*, not in the file.
A `.npy` array here has no date axis at all; the only thing that dates it is the
`windows/2025-08-13/` directory it sits in. `load_run_tree` exists for exactly
that case.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

# Directories that are never data, however deep the walk goes.
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv",
             "node_modules", ".pytest_cache", ".mypy_cache", "build", ".virtual_documents"}

TABULAR = {".csv", ".tsv", ".parquet", ".pkl", ".pickle", ".xlsx", ".xls", ".json"}
ARRAY = {".npy", ".npz"}
ARCHIVE = {".zip", ".gz", ".zst"}


# ── 1. discover ────────────────────────────────────────────────────────────────
def discover_files(root, extensions=None, skip_dirs=SKIP_DIRS,
                   max_mb: float | None = None) -> pd.DataFrame:
    """Walk `root` and return one row per file: path, extension, size, directory.

    `max_mb` filters the *catalogue*, not the walk — a 12.9 GB archive still gets
    a row, it just will not be offered to a loader. Recording it matters: a file
    you decided not to read is a decision, and it should be visible.
    """
    root = Path(root)
    rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        ext = path.suffix.lower()
        if extensions and ext not in extensions:
            continue
        size_mb = path.stat().st_size / 1e6
        rows.append({
            "path": str(path.relative_to(root)),
            "ext": ext,
            "size_mb": round(size_mb, 3),
            "directory": str(path.parent.relative_to(root)),
            "name": path.name,
            "loadable": (ext in TABULAR | ARRAY) and (max_mb is None or size_mb <= max_mb),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["directory", "name"]).reset_index(drop=True)
    out.attrs["root"] = str(root)
    return out


def extension_census(files: pd.DataFrame) -> pd.DataFrame:
    """Counts and total size by extension — the first look at any unfamiliar tree."""
    if not len(files):
        return pd.DataFrame(columns=["ext", "n_files", "total_mb", "largest_mb"])
    return (files.groupby("ext")
            .agg(n_files=("path", "size"), total_mb=("size_mb", "sum"),
                 largest_mb=("size_mb", "max"))
            .sort_values("n_files", ascending=False).reset_index())


def directory_census(files: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Which directories actually hold the data, by file count."""
    if not len(files):
        return pd.DataFrame(columns=["directory", "n_files", "total_mb", "extensions"])
    return (files.groupby("directory")
            .agg(n_files=("path", "size"), total_mb=("size_mb", "sum"),
                 extensions=("ext", lambda s: ", ".join(sorted(set(s)))))
            .sort_values("n_files", ascending=False).head(top).reset_index())


# ── 2. load ────────────────────────────────────────────────────────────────────
def load_any(path, *, allow_unsafe_pickle: bool = False, **kwargs):
    """Load a file by extension, with defaults chosen per format.

    Returns a DataFrame for tabular formats, an array for `.npy`, and a dict for
    JSON. Raises on an unsupported extension rather than guessing — a silent
    wrong-format read is worse than a stop.

    Notable defaults: `.npy` is memory-mapped (so cataloguing a 100 MB array
    costs nothing), and JSON that looks like a list of records is turned into a
    DataFrame while nested JSON is returned as a dict for `json_normalize`.
    Pickles can execute arbitrary code and therefore require the explicit
    ``allow_unsafe_pickle=True`` opt-in, even when their extension is known.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if ext in (".pkl", ".pickle"):
        if not allow_unsafe_pickle:
            raise ValueError(
                "refusing to unpickle without allow_unsafe_pickle=True; "
                "only opt in for a file from a trusted source")
        obj = pd.read_pickle(path, **kwargs)
        # These caches are lists of flat dicts; a DataFrame is what callers want.
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return obj
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    if ext == ".npy":
        return np.load(path, mmap_mode=kwargs.pop("mmap_mode", "r"), **kwargs)
    if ext == ".npz":
        return np.load(path, **kwargs)
    if ext == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return pd.DataFrame(payload)
        return payload

    raise ValueError(f"no loader for {ext!r} ({path.name}); read it explicitly")


def read_zip_member(zip_path, member: str | None = None, **kwargs):
    """Read one member of a zip without extracting the archive.

    The reason to care: this repo's OPRA archive is 12.9 GB holding 127 daily
    files. Extracting it to read one day is not an option, and `ZipFile` streams
    a single member in constant memory.

    With `member=None`, returns the archive's listing instead — which is how you
    find out what is in there in the first place.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if member is None:
            return pd.DataFrame(
                [{"member": i.filename, "size_mb": round(i.file_size / 1e6, 2),
                  "compressed_mb": round(i.compress_size / 1e6, 2)}
                 for i in archive.infolist()]).sort_values("size_mb", ascending=False)
        with archive.open(member) as handle:
            suffix = Path(member).suffix.lower()
            if suffix == ".json":
                return json.load(handle)
            if suffix in (".csv", ".tsv"):
                return pd.read_csv(handle, sep="\t" if suffix == ".tsv" else ",", **kwargs)
            return handle.read()


def load_run_tree(root, pattern: str, path_fields: dict[str, int] | None = None,
                  loader=None, add_source: bool = True) -> pd.DataFrame:
    """Concatenate many small files, lifting metadata out of their paths.

    Experiment trees put the metadata in the directory names —
    `runs/<run_id>/windows/<date>/replay/bs_daily.csv` — and the files themselves
    carry none of it. `path_fields` maps a column name to a position in the path,
    counted **backwards** from the file:

        load_run_tree(root, "**/replay/bs_daily.csv",
                      {"window_date": 2, "run_id": 4})

    Without this the concatenated frame is unusable: 32 files, all one row, all
    claiming to be the same thing.
    """
    root = Path(root)
    loader = loader or load_any
    frames = []
    for path in sorted(root.glob(pattern)):
        try:
            part = loader(path)
        except Exception as exc:                     # one bad file must not stop the sweep
            frames.append(pd.DataFrame([{"_error": f"{type(exc).__name__}: {exc}",
                                         "_source": str(path.relative_to(root))}]))
            continue
        if not isinstance(part, pd.DataFrame):
            part = pd.DataFrame([{"value": part}])
        for field_name, depth in (path_fields or {}).items():
            part[field_name] = path.parts[-(depth + 1)]
        if add_source:
            part["_source"] = str(path.relative_to(root))
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.attrs["n_files"] = len(frames)
    return out


# ── 3. describe ────────────────────────────────────────────────────────────────
def describe_schema(obj, name: str = "") -> pd.DataFrame:
    """Column, dtype, null rate, cardinality and a sample value, for any frame.

    Run this on every file as you catalogue it. It is the difference between "I
    have 77 CSVs" and "I know which two of them share a key".
    """
    if isinstance(obj, np.ndarray):
        return pd.DataFrame([{"name": name or "array", "dtype": str(obj.dtype),
                              "shape": str(obj.shape), "n_null": 0,
                              "null_rate": 0.0, "n_unique": np.nan, "sample": ""}])
    if not isinstance(obj, pd.DataFrame):
        return pd.DataFrame([{"name": name or "object", "dtype": type(obj).__name__,
                              "shape": "", "n_null": 0, "null_rate": 0.0,
                              "n_unique": np.nan, "sample": str(obj)[:60]}])

    rows = []
    for col in obj.columns:
        series = obj[col]
        non_null = series.dropna()
        rows.append({
            "name": col,
            "dtype": str(series.dtype),
            "shape": f"{len(series)}",
            "n_null": int(series.isna().sum()),
            "null_rate": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
            "sample": str(non_null.iloc[0])[:40] if len(non_null) else "",
        })
    return pd.DataFrame(rows)


@dataclass
class DataCatalog:
    """A searchable inventory of a data tree.

        catalog = DataCatalog.build(REPO)
        catalog.census()                      # what is here, by extension
        catalog.find("chain_panel")           # locate by name fragment
        df = catalog.load("demo/data/spy_chain_panel.csv")

    `build` only walks and stats — it reads nothing — so it is fast on a tree with
    a multi-gigabyte archive in it. Schemas are captured lazily by `profile`.
    """

    files: pd.DataFrame
    root: Path
    schemas: dict = field(default_factory=dict)

    @classmethod
    def build(cls, root, max_mb: float = 200.0, extensions=None) -> "DataCatalog":
        root = Path(root)
        return cls(files=discover_files(root, extensions=extensions, max_mb=max_mb),
                   root=root)

    def census(self) -> pd.DataFrame:
        return extension_census(self.files)

    def directories(self, top: int = 15) -> pd.DataFrame:
        return directory_census(self.files, top=top)

    def find(self, fragment: str, loadable_only: bool = False) -> pd.DataFrame:
        hits = self.files[self.files["path"].str.contains(fragment, case=False, regex=False)]
        return hits[hits["loadable"]] if loadable_only else hits

    def load(self, relative_path: str, **kwargs):
        return load_any(self.root / relative_path, **kwargs)

    def profile(self, fragment: str = "", limit: int = 20,
                max_mb: float = 50.0, *,
                allow_unsafe_pickle: bool = False) -> pd.DataFrame:
        """Read the schema of every matching loadable file. One row per column.

        Bounded by `limit` and `max_mb` on purpose — profiling is the step that
        actually opens files, and an unbounded profile of this repo would read
        several gigabytes. Pickles are recorded as unreadable unless the caller
        explicitly sets ``allow_unsafe_pickle=True`` for a trusted tree.
        """
        candidates = self.find(fragment, loadable_only=True)
        candidates = candidates[candidates["size_mb"] <= max_mb].head(limit)

        frames = []
        for rel in candidates["path"]:
            try:
                schema = describe_schema(
                    self.load(rel, allow_unsafe_pickle=allow_unsafe_pickle), name=rel)
            except Exception as exc:
                schema = pd.DataFrame([{"name": "<unreadable>",
                                        "dtype": f"{type(exc).__name__}", "shape": "",
                                        "n_null": 0, "null_rate": np.nan,
                                        "n_unique": np.nan, "sample": str(exc)[:60]}])
            schema.insert(0, "file", rel)
            frames.append(schema)
            self.schemas[rel] = schema

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def shared_columns(self, min_files: int = 2) -> pd.DataFrame:
        """Columns appearing in several profiled files — candidate join keys.

        The mechanical way to answer "what connects to what" across a tree you did
        not build. Profile first, then run this.
        """
        if not self.schemas:
            raise RuntimeError("call profile() first — shared_columns reads its results")
        rows = [{"column": col, "file": path}
                for path, schema in self.schemas.items()
                for col in schema["name"]]
        counts = (pd.DataFrame(rows).groupby("column")
                  .agg(n_files=("file", "nunique"),
                       files=("file", lambda s: ", ".join(sorted(set(s))[:4])))
                  .reset_index())
        return counts[counts["n_files"] >= min_files].sort_values(
            "n_files", ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        n = len(self.files)
        mb = self.files["size_mb"].sum() if n else 0
        return f"DataCatalog({n:,} files, {mb:,.0f} MB, root={self.root.name})"


# ── 4. caching ─────────────────────────────────────────────────────────────────
def to_parquet_cache(df: pd.DataFrame, name: str, cache_dir=None) -> Path:
    """Write a frame to the notebook output cache as parquet, and return the path.

    Parquet over CSV for a cache: it round-trips dtypes (including tz-aware
    timestamps and categoricals, both of which CSV destroys), and it is several
    times smaller and faster to read.
    """
    cache_dir = Path(cache_dir or config.OUTPUT_DIR / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (name if name.endswith(".parquet") else f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def from_parquet_cache(name: str, cache_dir=None) -> pd.DataFrame | None:
    """Read back a `to_parquet_cache` frame, or None when it is not there."""
    cache_dir = Path(cache_dir or config.OUTPUT_DIR / "cache")
    path = cache_dir / (name if name.endswith(".parquet") else f"{name}.parquet")
    return pd.read_parquet(path) if path.exists() else None


__all__ = [
    "discover_files", "extension_census", "directory_census", "load_any",
    "read_zip_member", "load_run_tree", "describe_schema", "DataCatalog",
    "to_parquet_cache", "from_parquet_cache", "SKIP_DIRS", "TABULAR", "ARRAY",
]
