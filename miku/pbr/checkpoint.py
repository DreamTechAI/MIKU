"""Strict LoRA + modules_to_save restoration for this project's PBR models.

Flux2's standard pipeline loader rejects the saved x_embedder.weight because it
is not a LoRA tensor. These helpers retain it and derive adapter targets from the
actual checkpoint, preventing random missing adapters from entering inference.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from peft.utils.other import get_pattern_key
from safetensors import safe_open
from safetensors.torch import load_file


def read_checkpoint(checkpoint):
    """Return (unprefixed tensor state, exact LoraConfig, inspection report)."""
    checkpoint = Path(checkpoint)
    weight_path = checkpoint if checkpoint.suffix == ".safetensors" else checkpoint / "pytorch_lora_weights.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    raw = load_file(str(weight_path), device="cpu")
    if not raw or any(not k.startswith("transformer.") for k in raw):
        raise ValueError("Expected a nonempty transformer-only Diffusers checkpoint")
    state = {k.removeprefix("transformer."): v for k, v in raw.items()}
    targets = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            module = key.removesuffix(".lora_A.weight")
            paired = state.get(module + ".lora_B.weight")
            if value.ndim != 2 or paired is None or paired.ndim != 2 or paired.shape[1] != value.shape[0]:
                raise ValueError(f"Invalid LoRA pair: {module}")
            targets[module] = int(value.shape[0])
    if not targets:
        raise ValueError("No LoRA A/B pairs found")
    lora_keys = {name + suffix for name in targets for suffix in (".lora_A.weight", ".lora_B.weight")}
    if any(".lora_" in key and key not in lora_keys for key in state):
        raise ValueError("Checkpoint contains unsupported or unpaired LoRA tensors")
    ordinary = [k for k in state if ".lora_" not in k]
    if any(not (k.endswith(".weight") or k.endswith(".bias")) for k in ordinary):
        raise ValueError(f"Unrecognized saved modules: {ordinary}")
    saved_modules = sorted({k.rsplit(".", 1)[0] for k in ordinary})
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    config_meta = json.loads(metadata.get("lora_adapter_metadata", "{}"))
    if config_meta.get("transformer.use_dora", False) or config_meta.get("transformer.bias", "none") != "none":
        raise ValueError("This PBR loader expects non-DoRA adapters with bias='none'")
    rank = next(iter(targets.values()))
    alpha = config_meta.get("transformer.lora_alpha", rank)
    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=sorted(targets),
        rank_pattern={name: r for name, r in targets.items() if r != rank},
        alpha_pattern=config_meta.get("transformer.alpha_pattern", {}),
        use_rslora=bool(config_meta.get("transformer.use_rslora", False)),
        lora_dropout=0.0, modules_to_save=saved_modules or None, init_lora_weights=True,
    )
    report = {"verified_tensor_count": len(state), "lora_modules": len(targets),
              "modules_to_save": saved_modules, "rank": rank, "alpha": alpha}
    return state, config, report


def _restore_and_verify(transformer, state, adapter_name):
    # PEFT rewrites modules_to_save keys in the mapping passed to it. Retain the
    # canonical checkpoint mapping for the exact round-trip verification below.
    incompatible = set_peft_model_state_dict(transformer, dict(state), adapter_name=adapter_name)
    if incompatible is not None and incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    transformer.set_adapter(adapter_name)
    restored = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    if set(restored) != set(state):
        raise ValueError(f"Adapter keys differ: missing={set(state)-set(restored)}, extra={set(restored)-set(state)}")
    for key, original in state.items():
        actual = restored[key].detach().cpu()
        if actual.shape != original.shape or not torch.equal(actual, original.to(actual.dtype)):
            raise ValueError(f"Checkpoint tensor was not fully restored: {key}")


def _validate_adapter_scaling(transformer, state, expected_config, adapter_name):
    """Check effective per-layer rank/alpha/scaling before changing any weights."""
    configs = getattr(transformer, "peft_config", {})
    if adapter_name not in configs:
        raise ValueError(f"Adapter does not exist: {adapter_name}")
    restored = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    if set(restored) != set(state):
        raise ValueError(f"Existing adapter keys differ from checkpoint: missing={set(state)-set(restored)}, extra={set(restored)-set(state)}")
    # PEFT may shorten config.target_modules in place (e.g. full paths become
    # "to_q"). Checkpoint keys remain authoritative full module paths.
    module_names = [key.removesuffix(".lora_A.weight") for key in state if key.endswith(".lora_A.weight")]
    for name in module_names:
        layer = transformer.get_submodule(name)
        rank_key = get_pattern_key(expected_config.rank_pattern.keys(), name)
        alpha_key = get_pattern_key(expected_config.alpha_pattern.keys(), name)
        rank = expected_config.rank_pattern.get(rank_key, expected_config.r)
        alpha = expected_config.alpha_pattern.get(alpha_key, expected_config.lora_alpha)
        scale = alpha / (math.sqrt(rank) if expected_config.use_rslora else rank)
        actual = (layer.r.get(adapter_name), layer.lora_alpha.get(adapter_name), layer.scaling.get(adapter_name))
        if (actual[0] != rank or actual[1] != alpha or actual[2] is None
                or not math.isclose(float(actual[2]), float(scale), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"Adapter rank/alpha/scaling mismatch for {name}: expected={(rank, alpha, scale)}, actual={actual}")









def initialize_adapter(transformer, checkpoint, adapter_name="default", *, return_report=False):
    """Restore and freeze exactly the adapter tensors present in the checkpoint."""
    state, config, report = read_checkpoint(checkpoint)
    transformer.add_adapter(config, adapter_name=adapter_name)
    _validate_adapter_scaling(transformer, state, config, adapter_name)
    _restore_and_verify(transformer, state, adapter_name)
    transformer.eval().requires_grad_(False)
    return (config, report) if return_report else config
