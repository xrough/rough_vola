# Three research notebooks

Read in order:

1. [Pricing and exercise](01_pricing_and_exercise.ipynb): current-source European smiles, American BS/Heston prices and deltas, cash events, the guarded American-call reduction, and explicitly archived rough-policy findings.
2. [Synthetic calibration](02_synthetic_calibration.ipynb): a small fresh noisy surface, BS/Heston fitting and controlled rough `v0` recovery, held-out prices and separate numerical checks.
3. [SPY pilot](03_spy_pilot.ipynb): optional-input presence audit and verified aggregate historical findings. No observed market contracts are distributed or refitted.

From the repository root:

```bash
python -m pip install -e '.[notebooks]'
python scripts/run_notebooks.py --all
# Independent executed copies:
python scripts/run_notebooks.py --all --output-dir /tmp/rough-vola-notebooks
python scripts/build_reports.py
```

Each notebook runs in a fresh kernel pinned to the selected interpreter. The
runner records relative source fingerprints and portable execution metadata;
it never requires a sibling repository or a large historical artifact tree.
A missing optional market input yields a not-recomputed status in notebook 03,
not fabricated prices or a synthetic substitute for SPY.

Current-source synthetic examples are new executions. Tables under
[`reports/aggregate_findings`](../reports/aggregate_findings/) are allowlisted
historical derivatives with source-content hashes. They are not fresh fits and
do not include full raw-data/source provenance. Full sealed observations and
research payloads stay local; see [research findings](../reports/research_findings.md)
and the [SPY pilot explanation](../reports/spy_pilot.md).

The examples have modest budgets and preserve failed numerical checks. They do
not certify general American rough-Heston accuracy, unique parameter recovery,
historical rough-state estimation or a hedge replay. No API credentials, raw
vendor archives, contract-level observations or absolute user paths belong in
published outputs.
