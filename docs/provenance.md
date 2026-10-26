# Consolidation provenance

This public edition consolidates two locally preserved research projects:

| Source | Role in this edition |
| --- | --- |
| `xrough/effective_engine` | Current `rvlab` pricing, calibration, market-data preparation, experiments and tests |
| `xrough/roughVol` | Selected rough-Bergomi simulation, empirical roughness and synthetic hedge-accounting tools |

The GitHub repository was renamed to `rough_vola`; the public research main
branch starts with a clean initial commit. The original C++ tree and its ancestry
are omitted from that branch. Both original working trees, their uncommitted
work and independent local Git backups are retained outside this publication
tree. Nothing in this consolidation claims to erase earlier public copies.

The [original-file fingerprint manifest](source-provenance.json) records the
local source commit identities and hashes of the Python research files used as
inputs. It explicitly includes uncommitted research; the old HEAD alone cannot
identify that state. Portability changes in this edition are new changes rather
than rewrites of the frozen historical sources.

The [selected roughVol import record](roughvol-import.md) lists imported
components, original hashes, integration corrections and limitations. The old
fractional-Riccati pricer and calibration routines are excluded; the corrected
`rvlab.pricing` implementation remains the canonical numerical source.

## Results and reproducibility

Historical campaign findings are curated exports of preserved experiments.
They retain their original numerical settings and failed acceptance checks.
They are not relabelled as results of the new public notebooks. The original
full manifests, raw input receipts and payloads remain in the separately
preserved local campaigns.

Public report exports remove machine-specific paths and contract-level market
observations. Their sources and original manifest fingerprints are identified
in the report metadata. Fresh public notebook calculations use the current
published source and small synthetic inputs. Notebook 03 makes optional data
availability explicit and performs no fabricated real-data fit.

Large historical-archive tests are opt-in. Their public source is retained for
inspection, while ordinary numerical tests exercise the shipped engines and
small fixtures. See [testing guidance](../tests/README.md).

## License

The selected sibling source declared MIT in an incomplete license stub. This
edition includes the complete MIT grant and disclaimer for the consolidated
software. Third-party papers are methodological references; no external market
data is relicensed by this code license.
