"""Explicit frozen decision coefficients retain identity without pickle."""
from dataclasses import fields
import hashlib
import json

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_frozen_policy import (
    _decode_models, _encode_models, load_frozen_policy_bundle)
from rvlab.pricing.rough_stopping import _Continuation
from rvlab.pricing.rough_tensor_premium import TensorPremium


def test_whitelisted_continuation_and_tensor_roundtrip_keeps_exact_coefficients(tmp_path):
    models = (
        _Continuation(np.array([.1, -.2]), np.array([.3, .4]), np.arange(11.)/7,
            (.85, 1.), 11, 23.4),
        TensorPremium(np.array([.2, -.3]), np.array([.6, .7]),
            np.arange(30.).reshape(5, 6)/11, (.9, 1.1), 2, 4., 28, 100.))
    arrays = {}
    declaration = _encode_models(models, "test", arrays)
    path = tmp_path/"coefficients.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        restored = _decode_models(json.loads(json.dumps(declaration)), saved)
    for original, copy in zip(models, restored):
        assert type(copy) is type(original)
        for field in fields(original):
            expected, actual = getattr(original, field.name), getattr(copy, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
                assert actual.tobytes() == expected.tobytes()
                assert not actual.flags.writeable
            else:
                assert actual == expected


def test_bundle_encoder_rejects_arbitrary_objects():
    with pytest.raises(ValueError, match="explicit continuation and tensor"):
        _encode_models([object()], "unsafe", {})


@pytest.mark.parametrize("mutation,match", [("class", "Unknown class"),
    ("missing", "Malformed frozen continuation fields"), ("array", "Malformed frozen coefficient array")])
def test_bundle_decoder_rejects_unapproved_or_malformed_content(mutation, match):
    arrays = {}
    declaration = _encode_models([_Continuation(np.zeros(1), np.ones(1), np.zeros(7), (), 1, 1.)],
        "test", arrays)
    if mutation == "class": declaration[0]["class"] = "arbitrary.callable"
    elif mutation == "missing": del declaration[0]["fields"]["rank"]
    else: declaration[0]["fields"]["beta"] = {"array": "missing"}
    with pytest.raises(ValueError, match=match):
        _decode_models(declaration, arrays)


def test_load_rejects_changed_archive_before_reconstructing_any_model(tmp_path):
    path = tmp_path/"coefficients.npz"
    np.savez_compressed(path, harmless=np.arange(3.))
    (tmp_path/"manifest.json").write_text(json.dumps({"version": "frozen_joint_policy_v1",
        "coefficient_archive_sha256": "0"*64}))
    with pytest.raises(ValueError, match="Changed frozen coefficient bundle"):
        load_frozen_policy_bundle(tmp_path)


def test_load_rejects_changed_training_provenance_before_reconstructing_any_model(tmp_path):
    path = tmp_path/"coefficients.npz"
    np.savez_compressed(path, harmless=np.arange(3.))
    source = tmp_path/"checkpoint.json"
    source.write_text("changed checkpoint")
    (tmp_path/"manifest.json").write_text(json.dumps({"version": "frozen_joint_policy_v1",
        "coefficient_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_checkpoint": {"path": str(source), "sha256": "0"*64}}))
    with pytest.raises(ValueError, match="Changed frozen policy source checkpoint"):
        load_frozen_policy_bundle(tmp_path)
