"""Dependency and import boundaries for a portable source installation."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_keeps_both_namespaces_with_small_numerical_core():
    metadata = tomllib.loads((ROOT/"pyproject.toml").read_text())
    assert metadata["project"]["name"] == "rough-vola"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert metadata["project"]["urls"]["Repository"] == "https://github.com/xrough/rough_vola"
    assert metadata["project"]["dependencies"] == ["numpy>=1.26", "scipy>=1.11", "pandas>=2.0"]
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["rvlab*", "roughvol*"]
    assert (ROOT/metadata["project"]["readme"]).is_file()


def test_pricing_and_consolidated_reference_import_without_optional_research_dependencies():
    script = textwrap.dedent('''
        import importlib.abc
        import pathlib
        import sys
        blocked = {"matplotlib", "sklearn", "statsmodels", "joblib", "databento",
                   "pyarrow", "requests", "exchange_calendars", "torch", "jax"}
        class NoOptionalPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ImportError("Optional dependency blocked: " + fullname)
                return None
        sys.meta_path.insert(0, NoOptionalPackages())
        import rvlab
        import roughvol
        from rvlab.pricing import BlackScholesParams, price
        from roughvol.models.rough_bergomi_model import RoughBergomiModel
        from roughvol.analytics.roughness import estimate_hurst_exponent
        expected = pathlib.Path(sys.argv[1]).resolve()
        assert pathlib.Path(rvlab.__file__).resolve().is_relative_to(expected)
        assert pathlib.Path(roughvol.__file__).resolve().is_relative_to(expected)
        assert 0 < price(BlackScholesParams(.2), forward=100., discount=1., strike=100., maturity=.1) < 100.
        assert not blocked.intersection(x.split(".")[0] for x in sys.modules)
        print("portable numerical imports passed")
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(ROOT/"src")],
        cwd=ROOT, env={**os.environ, "PYTHONPATH":str(ROOT/"src")},
        text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable numerical imports passed"
