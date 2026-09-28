"""Our Mamba-3 SISO agrees with upstream's, and loads its weights.

Strategy: drive `recur_sequence` against upstream's two vendored
references over random inputs, from empty and carried states, then
check the model around it: the configuration refusals, the parameter
names and shapes upstream's checkpoint uses, strict loading, and a
whole model run with our recurrence against the same model with
upstream's parallel form swapped in. Passing proves the recurrence is
upstream's, and that the glue feeds any core identically.

What this cannot prove is that the glue matches upstream's glue on
real weights. Only a trained checkpoint can show that, which is what
the hardware probe's perplexity check is for.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference.mamba3 import siso_reference as ref  # noqa: E402
from src.inference import mamba3  # noqa: E402
from src.inference.mamba3 import (  # noqa: E402
    CoreInputs,
    LayerState,
    Mamba3ConfigError,
    Mamba3LM,
    Mamba3LoadError,
    StepTerms,
    config_from_json,
    load,
    load_weights,
    recur_sequence,
)

# A model small enough to run in milliseconds, with every feature the
# real one uses: grouped B and C, a partial rotary, a padded
# vocabulary and a rounded MLP width.
TINY: Dict[str, Any] = {
    "d_model": 32,
    "d_intermediate": 64,
    "n_layer": 2,
    "vocab_size": 50,
    "ssm_cfg": {
        "layer": "Mamba3",
        "d_state": 16,
        "expand": 2,
        "headdim": 8,
        "ngroups": 1,
        "rope_fraction": 0.5,
        "A_floor": 1e-4,
        "is_mimo": False,
        "is_outproj_norm": False,
    },
    "attn_layer_idx": [],
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True,
}

# The pinned checkpoint's config.json, state-spaces/mamba3-siso-1.5b
# at 5cfc721542ec9ccee768088b2fd6b7e8101219d8, copied as published.
REAL: Dict[str, Any] = {
    "d_model": 2048,
    "d_intermediate": 4096,
    "n_layer": 24,
    "vocab_size": 128256,
    "ssm_cfg": {
        "layer": "Mamba3",
        "d_state": 128,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "rope_fraction": 0.5,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 0.0001,
        "A_floor": 0.0001,
        "chunk_size": 64,
        "is_mimo": False,
        "is_outproj_norm": False,
    },
    "attn_layer_idx": [],
    "attn_cfg": {},
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True,
}

BATCH, LENGTH, HEADS, GROUPS = 2, 12, 4, 1
QK_DIM, V_DIM, ANGLES = 16, 8, 4
# Everything here is fp32, and the parallel form sums in a different
# order from the loop, so agreement is to a few ulps, not bit for bit.
TOLERANCE = 1e-4


def _close(first: torch.Tensor, second: torch.Tensor) -> None:
    torch.testing.assert_close(
        first, second, rtol=TOLERANCE, atol=TOLERANCE
    )


def _draw(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=gen)


def _core_inputs(seed: int) -> CoreInputs:
    gen = torch.Generator().manual_seed(seed)
    dt = F.softplus(_draw(gen, BATCH, HEADS, LENGTH))
    rate = F.softplus(_draw(gen, BATCH, HEADS, LENGTH)) + 1e-4
    return CoreInputs(
        q=_draw(gen, BATCH, LENGTH, GROUPS, QK_DIM),
        k=_draw(gen, BATCH, LENGTH, GROUPS, QK_DIM),
        v=_draw(gen, BATCH, LENGTH, HEADS, V_DIM),
        adt=-rate * dt,
        dt=dt,
        trap=_draw(gen, BATCH, HEADS, LENGTH),
        q_bias=_draw(gen, HEADS, QK_DIM),
        k_bias=_draw(gen, HEADS, QK_DIM),
        angles=_draw(gen, BATCH, LENGTH, HEADS, ANGLES),
        d=_draw(gen, HEADS),
        z=_draw(gen, BATCH, LENGTH, HEADS, V_DIM),
    )


def _empty() -> LayerState:
    return LayerState(
        torch.zeros(BATCH, HEADS, ANGLES),
        torch.zeros(BATCH, HEADS, V_DIM, QK_DIM),
        torch.zeros(BATCH, HEADS, QK_DIM),
        torch.zeros(BATCH, HEADS, V_DIM),
    )


def _carried(seed: int) -> LayerState:
    gen = torch.Generator().manual_seed(seed)
    turns = torch.rand(BATCH, HEADS, ANGLES, generator=gen)
    return LayerState(
        turns * 2 * math.pi,
        0.1 * _draw(gen, BATCH, HEADS, V_DIM, QK_DIM),
        _draw(gen, BATCH, HEADS, QK_DIM),
        _draw(gen, BATCH, HEADS, V_DIM),
    )


def _reference_args(inputs: CoreInputs) -> Tuple[Any, ...]:
    return (
        inputs.q, inputs.k, inputs.v, inputs.adt, inputs.dt,
        inputs.trap, inputs.q_bias, inputs.k_bias, inputs.angles,
        inputs.d, inputs.z,
    )


def _step_core(
    inputs: CoreInputs, state: LayerState
) -> Tuple[torch.Tensor, LayerState]:
    """Upstream's loop, called as a core."""
    out, final = ref.mamba3_siso_step_ref(
        *_reference_args(inputs), Input_States=tuple(state)
    )
    return out, LayerState(*final)


def _parallel_core(
    inputs: CoreInputs, state: LayerState
) -> Tuple[torch.Tensor, LayerState]:
    """Upstream's parallel form, called as a core."""
    out, final = ref.mamba3_siso_fwd_ref(
        *_reference_args(inputs), Initial_States=tuple(state)
    )
    return out, LayerState(*final)


def _same_state(first: LayerState, second: LayerState) -> None:
    """Angles compared on the circle, since the forms wrap them at
    different moments; everything else directly."""
    _close(torch.cos(first.angle), torch.cos(second.angle))
    _close(torch.sin(first.angle), torch.sin(second.angle))
    for left, right in zip(first[1:], second[1:], strict=True):
        _close(left, right)


def _random_value(name: str, parameter: torch.Tensor) -> torch.Tensor:
    """Norm weights and D near one, everything else small and signed,
    so no parameter sits at an initial value that could hide a
    mistake."""
    noise = torch.randn(parameter.shape)
    if name.endswith(("norm.weight", "norm2.weight", ".D")):
        return 1.0 + 0.2 * noise
    return 0.3 * noise


def _tiny_model(
    seed: int, dtype: torch.dtype = torch.float32
) -> Mamba3LM:
    torch.manual_seed(seed)
    model = Mamba3LM(config_from_json(TINY), dtype=dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(_random_value(name, parameter))
    return model.eval()


def _tokens(seed: int, length: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    shape = (2, length)
    return torch.randint(0, TINY["vocab_size"], shape, generator=gen)


# -- the recurrence is upstream's --


def test_the_recurrence_matches_the_step_reference() -> None:
    inputs, state = _core_inputs(1), _carried(2)

    ours, ours_state = recur_sequence(inputs, state)
    theirs, their_state = _step_core(inputs, state)

    _close(ours, theirs)
    _same_state(ours_state, their_state)


def test_the_recurrence_matches_the_parallel_form() -> None:
    """The independent check: a quadratic form derived separately, not
    a second copy of the loop. From an empty state."""
    inputs = _core_inputs(3)

    ours, ours_state = recur_sequence(inputs, _empty())
    theirs, their_state = _parallel_core(inputs, _empty())

    _close(ours, theirs)
    _same_state(ours_state, their_state)


def test_the_parallel_form_agrees_from_a_carried_state() -> None:
    """A carried state exercises the first step's trapezoid term,
    which writes the previous token's key and value a second time."""
    inputs, state = _core_inputs(4), _carried(5)

    ours, ours_state = recur_sequence(inputs, state)
    theirs, their_state = _parallel_core(inputs, state)

    _close(ours, theirs)
    _same_state(ours_state, their_state)


def test_the_gate_and_the_skip_are_optional() -> None:
    """The gate is optional, as upstream's references allow; the
    retention tests rely on reading the ungated output."""
    inputs = _core_inputs(6)._replace(z=None)
    zero_skip = inputs._replace(d=torch.zeros(HEADS))

    ours, _ = recur_sequence(zero_skip, _empty())
    theirs, _ = _parallel_core(zero_skip, _empty())

    _close(ours, theirs)


# -- the configuration --


def test_the_real_config_reads_as_upstream_derives_it() -> None:
    config = config_from_json(REAL)

    # 2 * 4096 + 2 * 128 + 3 * 64 + 32, by hand.
    assert sum(config.in_proj_sizes) == 8672
    assert config.d_inner == 4096
    assert config.nheads == 64
    assert config.num_rope_angles == 32
    assert config.vocab_size == 128256
    assert config.mlp_hidden == 4096


def test_absent_switches_take_upstreams_defaults() -> None:
    """A switch missing from the file means what upstream runs."""
    trimmed = json.loads(json.dumps(TINY))
    for key in ("is_mimo", "is_outproj_norm", "ngroups",
                "rope_fraction", "A_floor"):
        del trimmed["ssm_cfg"][key]
    for key in ("rms_norm", "residual_in_fp32", "tie_embeddings",
                "attn_layer_idx", "pad_vocab_size_multiple"):
        del trimmed[key]

    config = config_from_json(trimmed)

    assert config.rope_fraction == 0.5
    assert config.a_floor == 1e-4
    assert config.vocab_size == 56, "padded to upstream's default 8"


@pytest.mark.parametrize(
    "path,value",
    [
        ("ssm_cfg.layer", "Mamba2"),
        ("ssm_cfg.is_mimo", True),
        ("ssm_cfg.is_outproj_norm", True),
        ("ssm_cfg.ngroups", 2),
        ("ssm_cfg.rope_fraction", 0.25),
        ("ssm_cfg.headdim", 7),
        ("ssm_cfg.d_state", 0),
        ("attn_layer_idx", [1]),
        ("rms_norm", False),
        ("tie_embeddings", False),
        ("residual_in_fp32", False),
        ("d_model", 0),
        ("d_intermediate", 0),
        ("n_layer", True),
    ],
)
def test_a_configuration_it_does_not_describe_is_refused(
    path: str, value: Any
) -> None:
    """Refused by name, never run as a model this file does not
    implement. `True` for n_layer because it is an int in Python."""
    changed = json.loads(json.dumps(TINY))
    section = changed
    *parents, key = path.split(".")
    for parent in parents:
        section = section[parent]
    section[key] = value

    with pytest.raises(Mamba3ConfigError, match=key):
        config_from_json(changed)


def test_a_config_without_an_ssm_section_is_refused() -> None:
    changed = {key: TINY[key] for key in TINY if key != "ssm_cfg"}

    with pytest.raises(Mamba3ConfigError, match="ssm_cfg"):
        config_from_json(changed)


# -- the names, shapes and dtypes upstream's checkpoint has --


def _upstream_shapes() -> Dict[str, Tuple[int, ...]]:
    """TINY's layout as upstream's modules would save it, written out
    by hand rather than read off ours, so a renamed or reshaped
    parameter cannot agree with itself. in_proj is 2 * 64 + 2 * 16 +
    3 * 8 + 4 = 188; the MLP rounds 64 up to 128; the vocabulary pads
    50 up to 64."""
    shapes: Dict[str, Tuple[int, ...]] = {
        "backbone.embedding.weight": (64, 32),
        "backbone.norm_f.weight": (32,),
        "lm_head.weight": (64, 32),
    }
    per_layer = {
        "norm.weight": (32,),
        "norm2.weight": (32,),
        "mixer.in_proj.weight": (188, 32),
        "mixer.dt_bias": (8,),
        "mixer.B_bias": (8, 1, 16),
        "mixer.C_bias": (8, 1, 16),
        "mixer.B_norm.weight": (16,),
        "mixer.C_norm.weight": (16,),
        "mixer.D": (8,),
        "mixer.out_proj.weight": (32, 64),
        "mlp.fc1.weight": (256, 32),
        "mlp.fc2.weight": (32, 128),
    }
    for layer in range(TINY["n_layer"]):
        for name, shape in per_layer.items():
            shapes[f"backbone.layers.{layer}.{name}"] = shape
    return shapes


def test_the_parameter_names_and_shapes_are_upstreams() -> None:
    weights = Mamba3LM(config_from_json(TINY)).state_dict()

    found = {name: tuple(tensor.shape) for name, tensor in
             weights.items()}

    assert found == _upstream_shapes()


def test_the_small_parameters_stay_fp32_in_a_bf16_model() -> None:
    """Upstream creates dt_bias, the B and C biases and D in fp32
    whatever the model's dtype; rounding them to bf16 would move every
    decay and every rotation."""
    model = Mamba3LM(config_from_json(TINY), dtype=torch.bfloat16)
    mixer = model.backbone.layers[0].mixer

    for parameter in (mixer.dt_bias, mixer.B_bias, mixer.C_bias,
                      mixer.D):
        assert parameter.dtype == torch.float32
    assert mixer.in_proj.weight.dtype == torch.bfloat16
    assert model.lm_head.weight is model.backbone.embedding.weight


# -- the glue --


def test_every_head_decays_at_least_by_the_floor() -> None:
    """Upstream clamps A at -A_floor, so alpha = exp(A * dt) is never
    one: no head can hold its state forever."""
    model = _tiny_model(7)
    mixer = model.backbone.layers[0].mixer
    hidden = torch.randn(2, 9, TINY["d_model"]) * 10

    inputs = mixer.project(hidden)

    floor = TINY["ssm_cfg"]["A_floor"]
    assert bool((inputs.adt <= -floor * inputs.dt + 1e-9).all())
    assert bool((inputs.dt > 0).all())


def test_the_heavy_tail_activation_is_upstreams() -> None:
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])

    expected = torch.tensor([0.25, 1.0 / 1.5, 1.0, 1.5, 4.0])
    torch.testing.assert_close(mamba3._heavy_tail(x), expected)


# -- the whole model --


def test_whole_model_logits_agree_across_cores() -> None:
    """The glue feeds either core identically: the same model with
    upstream's parallel form in place of our loop gives the same
    logits."""
    model = _tiny_model(8)
    tokens = _tokens(9, 10)

    with torch.no_grad():
        ours, _ = model(tokens, model.empty_states(2))
        theirs, _ = model(
            tokens, model.empty_states(2), core=_parallel_core
        )

    _close(ours, theirs)


def test_one_token_at_a_time_matches_the_whole_sequence() -> None:
    """Decoding carries the state between calls, so feeding tokens one
    by one must end where feeding them together does."""
    model = _tiny_model(10)
    tokens = _tokens(11, 8)

    with torch.no_grad():
        whole, _ = model(tokens, model.empty_states(2))
        states = model.empty_states(2)
        steps: List[torch.Tensor] = []
        for position in range(tokens.shape[1]):
            logits, states = model(tokens[:, position:position + 1],
                                   states)
            steps.append(logits)

    _close(torch.cat(steps, dim=1), whole)


def test_capture_records_every_step_of_every_layer() -> None:
    model = _tiny_model(12)
    tokens = _tokens(13, 6)
    capture: List[List[StepTerms]] = [[] for _ in range(2)]

    with torch.no_grad():
        model(tokens, model.empty_states(2), capture=capture)

    for layer in capture:
        assert len(layer) == 6
        assert layer[0].adt.shape == (2, 8)
        assert bool((layer[0].adt < 0).all())
        assert bool(((layer[0].gate > 0) & (layer[0].gate < 1)).all())


# -- loading a checkpoint --


def _save_checkpoint(
    directory: Path, weights: Dict[str, torch.Tensor]
) -> None:
    (directory / "config.json").write_text(json.dumps(TINY))
    torch.save(weights, directory / "pytorch_model.bin")


def test_a_checkpoint_round_trips_through_load(
    tmp_path: Path,
) -> None:
    model = _tiny_model(14)
    _save_checkpoint(tmp_path, model.state_dict())
    tokens = _tokens(15, 5)

    loaded = load(tmp_path, device="cpu", dtype=torch.float32)

    with torch.no_grad():
        expected, _ = model(tokens, model.empty_states(2))
        actual, _ = loaded(tokens, loaded.empty_states(2))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert loaded.lm_head.weight is loaded.backbone.embedding.weight


def test_a_checkpoint_without_its_tied_head_loads(
    tmp_path: Path,
) -> None:
    """Some savers drop a tied weight; nothing is lost by that."""
    weights = _tiny_model(16).state_dict()
    del weights["lm_head.weight"]
    _save_checkpoint(tmp_path, weights)

    loaded = load(tmp_path, device="cpu", dtype=torch.float32)

    assert loaded.lm_head.weight is loaded.backbone.embedding.weight


def test_an_unexpected_weight_is_refused_by_name() -> None:
    model = _tiny_model(17)
    weights = dict(model.state_dict())
    weights["backbone.layers.0.mixer.mimo_x"] = torch.zeros(1)

    with pytest.raises(Mamba3LoadError, match="mimo_x"):
        load_weights(model, weights)


def test_a_missing_weight_is_refused_by_name() -> None:
    model = _tiny_model(18)
    weights = dict(model.state_dict())
    del weights["backbone.layers.1.mixer.D"]

    with pytest.raises(Mamba3LoadError, match="layers.1.mixer.D"):
        load_weights(model, weights)


def test_a_misshapen_weight_is_refused() -> None:
    model = _tiny_model(19)
    weights = dict(model.state_dict())
    weights["backbone.layers.0.mixer.D"] = torch.zeros(3)

    with pytest.raises(Mamba3LoadError, match="mixer.D"):
        load_weights(model, weights)
