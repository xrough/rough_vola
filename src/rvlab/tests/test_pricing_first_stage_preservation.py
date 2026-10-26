"""Archive indices cannot overwrite evidence on case-insensitive volumes."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("pricing_preservation", ROOT / "scripts/preserve_pricing_first_stage.py")
preserve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preserve)


def test_original_readme_and_distinct_archive_index_can_coexist():
    preserve.validate_destinations([(Path("source"), Path("Readme.md")),
                                    (Path("guide"), Path("ORIGINAL_GUIDE.md"))])
    assert preserve.INDEX_NAME.casefold() != "Readme.md".casefold()


def test_case_only_source_collision_is_rejected_before_copying():
    with pytest.raises(ValueError, match="Case-insensitive"):
        preserve.validate_destinations([(Path("a"), Path("Readme.md")),
                                        (Path("b"), Path("README.md"))])


@pytest.mark.parametrize("reserved", ["ARCHIVE_INDEX.md", "PRESERVATION_MANIFEST.JSON"])
def test_generated_paths_are_reserved_before_copying(reserved):
    with pytest.raises(ValueError, match="destination collision"):
        preserve.validate_destinations([(Path("source"), Path(reserved))])
