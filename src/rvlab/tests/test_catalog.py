"""Repository-wide discovery and loading."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rvlab.data.catalog import (
    DataCatalog, describe_schema, directory_census, discover_files, extension_census,
    from_parquet_cache, load_any, load_run_tree, read_zip_member, to_parquet_cache,
)


@pytest.fixture
def tree(tmp_path):
    """A miniature version of the repository's shape: nested runs, mixed formats."""
    (tmp_path / "data").mkdir()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(tmp_path / "data/t.csv", index=False)
    pd.DataFrame({"v": [1.0, 2.0]}).to_pickle(tmp_path / "data/t.pkl")
    np.save(tmp_path / "data/arr.npy", np.arange(12, dtype=float).reshape(3, 4))
    (tmp_path / "data/meta.json").write_text(json.dumps({"k": 1, "nested": {"z": 2}}))

    for run in ("run_a", "run_b"):
        for day in ("2025-08-12", "2025-08-13"):
            d = tmp_path / "runs" / run / "windows" / day / "replay"
            d.mkdir(parents=True)
            pd.DataFrame({"date": [day], "pnl": [1.5 if run == "run_a" else 2.5]}
                         ).to_csv(d / "daily.csv", index=False)

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.csv").write_text("a\n1\n")
    return tmp_path


class TestDiscovery:
    def test_finds_files_and_skips_noise_directories(self, tree):
        files = discover_files(tree)
        assert len(files) == 8                      # 4 in data/, 4 replay CSVs
        assert not files["path"].str.contains("__pycache__").any()

    def test_extension_filter(self, tree):
        assert set(discover_files(tree, extensions={".csv"})["ext"]) == {".csv"}

    def test_census_totals_match_the_file_list(self, tree):
        files = discover_files(tree)
        census = extension_census(files)
        assert census["n_files"].sum() == len(files)
        assert census.loc[census["ext"] == ".csv", "n_files"].iloc[0] == 5

    def test_large_files_are_catalogued_but_not_loadable(self, tree):
        files = discover_files(tree, max_mb=0.0)
        assert len(files) == 8 and not files["loadable"].any()

    def test_directory_census_ranks_by_file_count(self, tree):
        top = directory_census(discover_files(tree))
        assert top.iloc[0]["n_files"] >= top.iloc[-1]["n_files"]


class TestLoadAny:
    def test_dispatches_on_extension(self, tree):
        assert isinstance(load_any(tree / "data/t.csv"), pd.DataFrame)
        assert isinstance(load_any(
            tree / "data/t.pkl", allow_unsafe_pickle=True), pd.DataFrame)
        assert isinstance(load_any(tree / "data/meta.json"), dict)
        assert load_any(tree / "data/arr.npy").shape == (3, 4)

    def test_npy_is_memory_mapped_not_copied(self, tree):
        assert isinstance(load_any(tree / "data/arr.npy"), np.memmap)

    def test_list_of_dicts_pickle_becomes_a_frame(self, tmp_path):
        pd.to_pickle([{"a": 1}, {"a": 2}], tmp_path / "recs.pkl")
        out = load_any(tmp_path / "recs.pkl", allow_unsafe_pickle=True)
        assert isinstance(out, pd.DataFrame) and list(out.columns) == ["a"]

    def test_pickle_requires_an_explicit_trust_boundary(self, tree):
        with pytest.raises(ValueError, match="allow_unsafe_pickle=True"):
            load_any(tree / "data/t.pkl")

    def test_unknown_extension_raises_rather_than_guessing(self, tmp_path):
        (tmp_path / "x.weird").write_text("?")
        with pytest.raises(ValueError, match="no loader"):
            load_any(tmp_path / "x.weird")


class TestRunTree:
    def test_concatenates_and_lifts_metadata_out_of_the_path(self, tree):
        out = load_run_tree(tree / "runs", "*/windows/*/replay/daily.csv",
                            {"window_date": 2, "run_id": 4})
        assert len(out) == 4
        assert set(out["run_id"]) == {"run_a", "run_b"}
        assert set(out["window_date"]) == {"2025-08-12", "2025-08-13"}

    def test_without_path_fields_the_rows_are_indistinguishable(self, tree):
        """The reason `path_fields` exists: the files carry no run identity."""
        out = load_run_tree(tree / "runs", "*/windows/*/replay/daily.csv",
                            add_source=False)
        assert len(out) == 4
        assert "run_id" not in out.columns

    def test_a_broken_file_is_recorded_not_fatal(self, tree):
        bad = tree / "runs/run_a/windows/2025-08-12/replay/daily.csv"
        bad.write_text('a,b\n"unterminated\n')
        out = load_run_tree(tree / "runs", "*/windows/*/replay/daily.csv")
        assert len(out) >= 3                        # the other three still loaded

    def test_no_matches_returns_an_empty_frame(self, tree):
        assert load_run_tree(tree / "runs", "*/nothing/*.csv").empty


class TestSchemaAndCatalog:
    def test_describe_schema_reports_nulls_and_cardinality(self):
        df = pd.DataFrame({"a": [1, 1, None], "b": list("xyz")})
        schema = describe_schema(df).set_index("name")
        assert schema.loc["a", "n_null"] == 1
        assert schema.loc["b", "n_unique"] == 3

    def test_catalog_builds_without_reading_files(self, tree):
        catalog = DataCatalog.build(tree)
        assert len(catalog.files) == 8 and catalog.schemas == {}

    def test_find_and_load_round_trip(self, tree):
        catalog = DataCatalog.build(tree)
        hit = catalog.find("t.csv")
        assert len(hit) == 1
        assert len(catalog.load(hit.iloc[0]["path"])) == 2

    def test_profile_then_shared_columns_finds_join_keys(self, tree):
        catalog = DataCatalog.build(tree)
        catalog.profile()
        shared = catalog.shared_columns(min_files=2)
        assert "date" in set(shared["column"])      # the 4 replay CSVs share it

    def test_shared_columns_requires_a_profile_first(self, tree):
        with pytest.raises(RuntimeError, match="profile"):
            DataCatalog.build(tree).shared_columns()


class TestZipAndCache:
    def test_zip_listing_reads_only_the_central_directory(self, tmp_path):
        import zipfile
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("inner.csv", "a,b\n1,2\n")
            z.writestr("meta.json", '{"k": 1}')
        listing = read_zip_member(path)
        assert set(listing["member"]) == {"inner.csv", "meta.json"}

    def test_zip_member_reads_without_extracting(self, tmp_path):
        import zipfile
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("inner.csv", "a,b\n1,2\n")
        assert read_zip_member(path, "inner.csv").shape == (1, 2)

    def test_parquet_cache_preserves_tz_aware_dtypes(self, tmp_path):
        """The reason to cache as parquet rather than CSV: CSV loses this."""
        df = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=3, tz="UTC"),
                           "cat": pd.Categorical(["a", "b", "a"]), "x": [1.0, 2.0, 3.0]})
        to_parquet_cache(df, "roundtrip", cache_dir=tmp_path)
        back = from_parquet_cache("roundtrip", cache_dir=tmp_path)
        assert str(back["ts"].dtype) == str(df["ts"].dtype)
        assert str(back["cat"].dtype) == "category"

    def test_missing_cache_returns_none(self, tmp_path):
        assert from_parquet_cache("absent", cache_dir=tmp_path) is None
