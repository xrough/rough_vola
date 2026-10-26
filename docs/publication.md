# Testing and the public research edition

The public edition preserves the current Python pricing APIs while removing the
C++ build, market datasets, local caches and multi-gigabyte historical archives
from the published Git tree. The original working trees remain intact locally.

## Source tests

```bash
python -m pip install -e '.[dev,notebooks,market-data,research]'
python -m pytest
```

CI runs numerical tests and the portable notebook workflow. Checks whose target
is an immutable private historical report, manifest or archived notebook are
identified separately with explicit skip reasons. They are not evidence for
freshly executed results. Historical source identities and findings are retained
in the consolidation provenance and report exports.

## Notebook execution

```bash
python scripts/run_notebooks.py --all
```

The runner pins fresh kernels to the interpreter that launches it. Public
synthetic examples use current sources and small reproducible inputs. The SPY
notebook reports missing inputs honestly and links to historical aggregate
findings; it does not fabricate a new market calibration.

Full historical numerical campaigns remain independent, expensive reproductions
that require their source/input closures. Their archived runner sources remain
available for methodological inspection, but their private data and large
payloads are not silently substituted in the public demonstrations.

## Migration

The public repository is named `xrough/rough_vola`. Its distribution is
`rough-vola`; Python imports remain `rvlab` and the selected `roughvol` modules.
The old `effective_engine` repository address redirects after the GitHub rename.
The sibling `roughVol` repository is private, and its selected imports are
documented separately. A clean initial research commit starts the new public
main branch. Old local repositories and Git backups preserve earlier history.

Old public clones or cached commits can still exist after a history reset.
Future development should use a fresh clone of the new research main branch,
rather than merging the old engineering ancestry back into it.
