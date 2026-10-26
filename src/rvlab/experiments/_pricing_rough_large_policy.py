"""Explicit model bundles for the independent larger-policy stopping study."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np


def load_larger_policy_bundle(folder):
    """Load all frozen policy coefficients without training or new selection."""
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    from ._pricing_rough_frozen_policy import load_frozen_policy_bundle,_decode_models
    folder=Path(folder)
    record=json.loads((folder/"frozen_policy_before_final.json").read_text())
    coefficients=folder/record["coefficient_archive"]
    if hashlib.sha256(coefficients.read_bytes()).hexdigest()!=record["coefficient_archive_sha256"]:
        raise ValueError("Changed larger-policy coefficient archive")
    frozen_path=Path(record["historical_bundle_path"])
    if hashlib.sha256((frozen_path/"manifest.json").read_bytes()).hexdigest()!=record["historical_bundle_sha256"]:
        raise ValueError("Changed historical policy bundle")
    old=load_frozen_policy_bundle(frozen_path)
    candidates={}
    with np.load(coefficients,allow_pickle=False) as arrays:
        for name,meta in record["policies"].items():
            candidates[name]=ControlledExercisePolicy(old["proxy"],arrays["times"],
                _decode_models(meta["models"],arrays),meta["exercise_at_inception"],
                tuple(meta["training_seeds"]),meta["premium_floor"],meta["premium_cap"],meta["diagnostics"])
    return {"proxy":old["proxy"],"weights":old["weights"],"original_policy":old["original_policy"],
        "previous_selected_policy":old["selected_policy"],"candidates":candidates,
        "selected_name":record["selected_name"],"selected_policy":candidates[record["selected_name"]],
        "manifest":record}


__all__=["load_larger_policy_bundle"]
