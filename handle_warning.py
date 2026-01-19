import torch
import contextlib

def _fail(msg: str) -> None:
    """Fail fast for artifact reproducibility."""
    raise RuntimeError(msg)


def check_state_dict_compat(
    model: torch.nn.Module,
    state: dict,
    *,
    require_no_missing: bool = True,
    suppress_expected_unexpected: bool = True,
) -> None:
    """
    Validate checkpoint compatibility.

    Policy:
    - Any `missing` keys => fail fast (random/default init would break reproducibility).
    - `unexpected` keys are usually safe, but we optionally allow only a known-safe set.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)

    allowed_unexpected = {'bert.embeddings.position_ids'} if suppress_expected_unexpected else set()
    unexpected_set = set(unexpected)
    unexpected_bad = sorted(unexpected_set - allowed_unexpected)

    if require_no_missing and len(missing) > 0:
        _fail(
            "Checkpoint/model mismatch: missing keys while loading state_dict.\n"
            f"Missing keys ({len(missing)}): {missing}\n"
            "Missing keys mean some expected parameters/buffers were not loaded and may remain randomly initialized, "
            "which can change inference and break artifact reproducibility."
        )

    if len(unexpected_bad) > 0:
        _fail(
            "Checkpoint/model mismatch: unexpected keys not in allowlist.\n"
            f"Unexpected keys ({len(unexpected_bad)}): {unexpected_bad}\n"
            "If these are not known-safe non-trainable buffers, they may indicate a model-definition mismatch."
        )


def check_position_ids_policy(state: dict) -> None:
    """
    Position-id buffer policy for reproducibility.
    - Presence in checkpoint is OK; some HF/BERT variants serialize it, some don't.
    - We do NOT fail on its presence. We only use this for cleanliness and to avoid noisy logs.
    """
    # Known-safe helper buffer; no action needed.
    _ = ('bert.embeddings.position_ids' in state)


def check_pooler_final_matches_checkpoint(
    model_bert: torch.nn.Module,
    state: dict,
) -> None:
    """
    Ensure the final in-memory pooler weights used for inference match the release checkpoint.
    This guarantees the earlier `from_pretrained(...)` warning does not affect final inference.
    """
    sd = model_bert.state_dict()

    required = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    for k in required:
        if k not in sd:
            _fail(f"Pooler sanity check failed: '{k}' missing from final model state_dict.")
        if k not in state:
            _fail(f"Pooler sanity check failed: '{k}' missing from release checkpoint state_dict.")

    # Compare exact tensors (CPU to avoid device mismatch).
    w_ok = torch.allclose(sd['bert.pooler.dense.weight'].cpu(), state['bert.pooler.dense.weight'].cpu())
    b_ok = torch.allclose(sd['bert.pooler.dense.bias'].cpu(), state['bert.pooler.dense.bias'].cpu())

    if not (w_ok and b_ok):
        _fail(
            "Pooler sanity check failed: final pooler params do not match the release checkpoint.\n"
            f"weight_match={w_ok}, bias_match={b_ok}\n"
            "This can affect embeddings because OPTango uses output.pooler_output as fn_embeddings."
        )


@contextlib.contextmanager
def suppress_transformers_warnings(level: str = 'error'):
    """
    Suppress HuggingFace Transformers log messages (e.g., 'Some weights were not initialized...').
    This does NOT change model parameters; it only reduces noisy logs for artifact reproduction runs.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        # Transformers not available or different layout; do nothing.
        yield
        return

    prev_verbosity = hf_logging.get_verbosity()
    prev_default_handler = hf_logging._default_handler

    # Reduce verbosity and also disable the default handler to avoid stdout spam.
    hf_logging.set_verbosity_error() if level == 'error' else hf_logging.set_verbosity_warning()
    hf_logging.disable_default_handler()

    try:
        yield
    finally:
        # Restore previous logging state.
        hf_logging.set_verbosity(prev_verbosity)
        hf_logging.enable_default_handler()
        # Some versions track handler differently; keep safe behavior.
        hf_logging._default_handler = prev_default_handler

# EOF
