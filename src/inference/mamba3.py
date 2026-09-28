"""Mamba-3 SISO in plain PyTorch.

Mamba-3 is a state-space language model: in place of attention, every
layer carries a fixed-size recurrent state that each token decays and
writes into. That state is why this project wants the model, since
what a model keeps and forgets can be read straight off it.

**Why this is our own implementation.** Upstream's `mamba-ssm`
decodes every token through a CuTe kernel its own docstring says is
tested only on H100, and this project's card is an RTX 4090, compute
capability 8.9. Installing it would also take a newer torch, Triton
3.5, TileLang and QuACK built from git source, and `transformers` has
no Mamba-3 at all. The recurrence itself is small, and upstream ships
two plain PyTorch statements of it in its tests. `reference/mamba3/`
vendors them, and `tests/inference/test_mamba3.py` requires this
module to agree with both.

**Names follow upstream's checkpoint**, `backbone.layers.{i}.mixer`
and so on, so a strict load of its weights fails loudly on any layout
this file gets wrong. The glue around the recurrence follows
upstream's `mamba3.py`, `block.py` and `mlp.py` at commit
e9594ce1c732d97440f0332fdc43170a2294dbfa.

**Only what the pinned checkpoint uses is implemented**: SISO rather
than MIMO, one group for B and C, no output-projection norm, RMSNorm
throughout and tied embeddings. `config_from_json` refuses anything
else by name rather than running a model this file does not describe.

**The recurrence is a pure function between shared glue.**
`Mamba3Mixer.project` turns hidden states into the inputs upstream's
references take, a core runs the recurrence, and `finish` projects
back. Swapping the core for upstream's parallel form is how the probe
checks this module on real activations. Every per-token quantity is an
ordinary tensor here, which is what makes the state observable;
`StepTerms` records them when asked.

Build a model in the dtype it will run in. `dt_bias`, the B and C
biases and `D` are fp32 whatever that dtype is, as upstream creates
them, so converting a built model with `.to(dtype)` would be wrong.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

# Upstream's epsilon for every RMSNorm in the model: the block norms,
# the final norm, and the norms on B and C.
NORM_EPS = 1e-5
TWO_PI = 2.0 * math.pi
# Upstream's GatedMLP rounds its hidden width up to this multiple.
MLP_MULTIPLE = 128

assert 0.0 < NORM_EPS < 1e-3, "an epsilon this large changes the norm"
assert MLP_MULTIPLE > 0, "the MLP width rounds to a positive multiple"

DeviceLike = Union[str, torch.device]


class Mamba3ConfigError(ValueError):
    """A configuration naming what this module does not describe."""


class Mamba3LoadError(RuntimeError):
    """Weights that do not fit the model the config describes."""


# Switches upstream offers that this module implements one side of:
# (key, the only value accepted, upstream's default when absent).
_SSM_SWITCHES: Tuple[Tuple[str, Any, Any], ...] = (
    ("is_mimo", False, False),
    ("is_outproj_norm", False, False),
    ("ngroups", 1, 1),
)
_MODEL_SWITCHES: Tuple[Tuple[str, Any, Any], ...] = (
    ("rms_norm", True, True),
    ("tie_embeddings", True, True),
    ("residual_in_fp32", True, True),
)
# Upstream's defaults for the two recurrence constants and the
# vocabulary padding, used only when a configuration omits them.
_DEFAULT_ROPE_FRACTION = 0.5
_DEFAULT_A_FLOOR = 1e-4
_DEFAULT_VOCAB_MULTIPLE = 8


@dataclass(frozen=True)
class Mamba3Config:
    """The architecture, as read from a checkpoint's `config.json`."""

    d_model: int
    d_intermediate: int
    n_layer: int
    # After upstream's padding to a multiple, which is the embedding's
    # real size.
    vocab_size: int
    d_state: int
    expand: int
    headdim: int
    rope_fraction: float
    a_floor: float

    def __post_init__(self) -> None:
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert self.d_inner % self.headdim == 0, (
            "the inner width must be a whole number of heads"
        )
        assert self.rope_fraction in (0.5, 1.0), self.rope_fraction
        assert self.num_rope_angles > 0, "no rotary angles left"
        assert self.a_floor > 0.0, "A_floor must be positive"

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    @property
    def num_rope_angles(self) -> int:
        """Rotary angles per head, as upstream derives them."""
        rotated = int(self.d_state * self.rope_fraction)
        if rotated % 2 != 0:
            rotated -= 1
        return rotated // 2

    @property
    def mlp_hidden(self) -> int:
        """GatedMLP's hidden width, rounded up as upstream does."""
        steps = -(-self.d_intermediate // MLP_MULTIPLE)
        return steps * MLP_MULTIPLE

    @property
    def in_proj_sizes(self) -> Tuple[int, ...]:
        """`in_proj`'s outputs in upstream's order: z, x, B, C, dd_dt,
        dd_A, trap, angles. One group and no MIMO, so B and C are each
        a single `d_state` vector."""
        return (
            self.d_inner,
            self.d_inner,
            self.d_state,
            self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        )


def config_from_json(data: Dict[str, Any]) -> Mamba3Config:
    """Read and check a checkpoint's configuration.

    A switch absent from the file takes upstream's default, because
    that is what upstream would run. A switch selecting something this
    module does not implement is refused by name, and so is a missing
    or non-positive dimension.
    """
    ssm = data.get("ssm_cfg")
    if not isinstance(ssm, dict):
        raise Mamba3ConfigError("config.json has no ssm_cfg object")
    layer = ssm.get("layer")
    _refuse_unless(layer == "Mamba3", "ssm_cfg.layer", layer)
    _check_switches(ssm, _SSM_SWITCHES, "ssm_cfg.")
    _check_switches(data, _MODEL_SWITCHES, "")
    attention = data.get("attn_layer_idx")
    _refuse_unless(not attention, "attn_layer_idx", attention)
    rope = ssm.get("rope_fraction", _DEFAULT_ROPE_FRACTION)
    _refuse_unless(rope in (0.5, 1.0), "ssm_cfg.rope_fraction", rope)

    d_model = _positive_int(data, "d_model", "")
    expand = _positive_int(ssm, "expand", "ssm_cfg.")
    headdim = _positive_int(ssm, "headdim", "ssm_cfg.")
    fits = (expand * d_model) % headdim == 0
    _refuse_unless(fits, "ssm_cfg.headdim", headdim)
    return Mamba3Config(
        d_model=d_model,
        d_intermediate=_positive_int(data, "d_intermediate", ""),
        n_layer=_positive_int(data, "n_layer", ""),
        vocab_size=_padded_vocab(data),
        d_state=_positive_int(ssm, "d_state", "ssm_cfg."),
        expand=expand,
        headdim=headdim,
        rope_fraction=float(rope),
        a_floor=float(ssm.get("A_floor", _DEFAULT_A_FLOOR)),
    )


def read_config(checkpoint_dir: Path) -> Mamba3Config:
    """The configuration stored beside a checkpoint's weights."""
    text = (checkpoint_dir / CONFIG_NAME).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise Mamba3ConfigError(f"{CONFIG_NAME} is not an object")
    return config_from_json(data)


def _refuse_unless(accepted: bool, field: str, found: Any) -> None:
    if not accepted:
        raise Mamba3ConfigError(
            f"{field} is {found!r}, which this implementation does"
            " not describe"
        )


def _check_switches(
    section: Dict[str, Any],
    switches: Tuple[Tuple[str, Any, Any], ...],
    prefix: str,
) -> None:
    for key, accepted, default in switches:
        found = section.get(key, default)
        _refuse_unless(found == accepted, prefix + key, found)


def _positive_int(
    section: Dict[str, Any], key: str, prefix: str
) -> int:
    value = section.get(key)
    usable = isinstance(value, int) and not isinstance(value, bool)
    if not usable or value <= 0:
        raise Mamba3ConfigError(
            f"{prefix}{key} must be a positive integer, not {value!r}"
        )
    return value


def _padded_vocab(data: Dict[str, Any]) -> int:
    """The vocabulary padded up to a multiple, as upstream pads it
    before building the embedding; the weights have this size."""
    vocab = _positive_int(data, "vocab_size", "")
    if "pad_vocab_size_multiple" not in data:
        multiple = _DEFAULT_VOCAB_MULTIPLE
    else:
        multiple = _positive_int(data, "pad_vocab_size_multiple", "")
    return -(-vocab // multiple) * multiple


# -- the recurrence --


class CoreInputs(NamedTuple):
    """One layer's recurrence inputs, in upstream's reference layout.

    q and k are (batch, length, groups, d_state); v and z (batch,
    length, heads, headdim); adt, dt and trap (batch, heads, length);
    angles (batch, length, heads, rotary angles); q_bias and k_bias
    (heads, d_state); d (heads,). `trap` and `angles` are the raw
    projections, before the sigmoid and tanh the recurrence applies.
    """

    q: Tensor
    k: Tensor
    v: Tensor
    adt: Tensor
    dt: Tensor
    trap: Tensor
    q_bias: Tensor
    k_bias: Tensor
    angles: Tensor
    d: Tensor
    z: Optional[Tensor]


class LayerState(NamedTuple):
    """What one layer carries between tokens, in upstream's order.

    angle (batch, heads, rotary angles) is the running rotation; ssm
    (batch, heads, headdim, d_state) is the state proper; k and v are
    the previous token's rotated key and its value, which the
    trapezoid rule writes a second time on the next step. All fp32.
    """

    angle: Tensor
    ssm: Tensor
    k: Tensor
    v: Tensor


class StepTerms(NamedTuple):
    """What one step of one layer did, per batch row and head.

    `adt` is log(alpha), the decay applied to the whole state, kept as
    a logarithm so a long product of decays cannot underflow. This
    token's write is weighted `dt * gate` now and the previous token's
    `alpha * dt * (1 - gate)`. q and k are the rotated vectors, and
    `state_norm` is the state's Frobenius norm after the step.
    """

    adt: Tensor  # (batch, heads)
    dt: Tensor  # (batch, heads)
    gate: Tensor  # (batch, heads), sigmoid of the raw trap
    q: Tensor  # (batch, heads, d_state)
    k: Tensor  # (batch, heads, d_state)
    v: Tensor  # (batch, heads, headdim)
    state_norm: Tensor  # (batch, heads)


Core = Callable[[CoreInputs, LayerState], Tuple[Tensor, LayerState]]


def recur_sequence(
    inputs: CoreInputs,
    state: LayerState,
    sink: Optional[List[StepTerms]] = None,
) -> Tuple[Tensor, LayerState]:
    """The Mamba-3 SISO recurrence over a sequence, a step at a time.

    Written apart from upstream's `mamba3_siso_step_ref`, and held to
    it and to the parallel `mamba3_siso_fwd_ref` by the tests.
    Computes in fp32 and returns (batch, length, heads, headdim) with
    the state after the last token. `sink`, when given, receives every
    step's `StepTerms`.
    """
    q = _per_head_vectors(inputs.q, inputs.q_bias)
    k = _per_head_vectors(inputs.k, inputs.k_bias)
    v = inputs.v.float()
    adt = inputs.adt.float()
    dt = inputs.dt.float()
    gate = torch.sigmoid(inputs.trap.float())
    turn = torch.tanh(inputs.angles.float()) * math.pi
    outputs: List[Tensor] = []
    for t in range(v.shape[1]):
        step = (adt[..., t], dt[..., t], gate[..., t])
        y, state, q_rot = _step(
            q[:, t], k[:, t], v[:, t], step, turn[:, t], state
        )
        outputs.append(y)
        if sink is not None:
            sink.append(_terms(step, q_rot, state))
    y = torch.stack(outputs, dim=1)
    y = y + inputs.d.float()[:, None] * v
    if inputs.z is not None:
        y = y * F.silu(inputs.z.float())
    return y, state


# One step's (adt, dt, gate), each (batch, heads).
StepScalars = Tuple[Tensor, Tensor, Tensor]


def _step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scalars: StepScalars,
    turn: Tensor,
    state: LayerState,
) -> Tuple[Tensor, LayerState, Tensor]:
    """One token through one layer, before the skip and the gate.

    The trapezoid rule writes each token twice: `gamma` weights this
    token now, and `beta` weights the previous one again, decayed.
    """
    adt, dt, gate = scalars
    angle = _wrap(state.angle + turn * dt[..., None])
    q_rot = _rotate(q, angle)
    k_rot = _rotate(k, angle)
    alpha = torch.exp(adt)
    beta = alpha * dt * (1.0 - gate)
    gamma = dt * gate
    ssm = (
        _per_head(alpha) * state.ssm
        + _per_head(beta) * _outer(state.v, state.k)
        + _per_head(gamma) * _outer(v, k_rot)
    )
    y = torch.einsum("bhpn,bhn->bhp", ssm, q_rot)
    return y, LayerState(angle, ssm, k_rot, v), q_rot


def _per_head(weight: Tensor) -> Tensor:
    """(batch, heads), shaped to scale a state per head."""
    return weight[..., None, None]


def _terms(
    scalars: StepScalars, q_rot: Tensor, state: LayerState
) -> StepTerms:
    adt, dt, gate = scalars
    norm = torch.linalg.vector_norm(state.ssm, dim=(-2, -1))
    return StepTerms(adt, dt, gate, q_rot, state.k, state.v, norm)


def _per_head_vectors(vectors: Tensor, bias: Tensor) -> Tensor:
    """(batch, length, groups, n) to one biased row per head, in fp32.

    Each group repeats for its heads in a row, as upstream's layout
    has it; `bias` is (heads, n) and sets the head count.
    """
    heads, groups = bias.shape[0], vectors.shape[2]
    assert heads % groups == 0, "heads must divide evenly into groups"
    grown = vectors.float().repeat_interleave(heads // groups, dim=2)
    return grown + bias.float()


def _wrap(angle: Tensor) -> Tensor:
    """Angles reduced into [0, 2*pi), as upstream keeps them."""
    return angle - TWO_PI * torch.floor(angle / TWO_PI)


def _rotate(vectors: Tensor, angle: Tensor) -> Tensor:
    """Rotate adjacent pairs (0, 1), (2, 3), ... of the last axis.

    Pairs beyond the angles given are left alone, which is how a
    rotary covering part of each vector works.
    """
    pairs = vectors.reshape(*vectors.shape[:-1], -1, 2)
    untouched = pairs.shape[-2] - angle.shape[-1]
    assert untouched >= 0, "more angles than pairs to rotate"
    cos = F.pad(torch.cos(angle), (0, untouched), value=1.0)
    sin = F.pad(torch.sin(angle), (0, untouched), value=0.0)
    first, second = pairs[..., 0], pairs[..., 1]
    turned = torch.stack(
        (first * cos - second * sin, first * sin + second * cos),
        dim=-1,
    )
    return turned.reshape(vectors.shape)


def _outer(v: Tensor, k: Tensor) -> Tensor:
    """Per head, v (b, h, p) times k (b, h, n) to (b, h, p, n)."""
    return v[..., :, None] * k[..., None, :]


def _heavy_tail(x: Tensor) -> Tensor:
    """Upstream's activation for the data-dependent A: 1 + x above
    zero and 1 / (1 - x) below, positive and smooth at zero."""
    return x.clamp_min(0.0) + torch.reciprocal(1.0 - x.clamp_max(0.0))


# -- the modules, named as upstream names them --


def _rms_norm(x: Tensor, weight: Tensor) -> Tensor:
    """Upstream's RMSNorm arithmetic, always in fp32."""
    x32 = x.float()
    mean_square = x32.square().mean(dim=-1, keepdim=True)
    return x32 * torch.rsqrt(mean_square + NORM_EPS) * weight.float()


class RMSNorm(nn.Module):
    """Upstream's RMSNorm: fp32 arithmetic, a weight and no bias."""

    def __init__(
        self, size: int, *, dtype: torch.dtype, device: DeviceLike
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(size, dtype=dtype, device=device)
        )

    def forward(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return _rms_norm(x, self.weight).to(dtype)


class GatedMLP(nn.Module):
    """Upstream's GatedMLP: fc1, split values from gate, fc2."""

    def __init__(
        self,
        config: Mamba3Config,
        *,
        dtype: torch.dtype,
        device: DeviceLike,
    ) -> None:
        super().__init__()
        hidden = config.mlp_hidden
        self.fc1 = nn.Linear(
            config.d_model, 2 * hidden, bias=False,
            dtype=dtype, device=device,
        )
        self.fc2 = nn.Linear(
            hidden, config.d_model, bias=False,
            dtype=dtype, device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        values, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(values * F.silu(gate))


class Mamba3Mixer(nn.Module):
    """One layer's Mamba-3 SISO mixer."""

    def __init__(
        self,
        config: Mamba3Config,
        *,
        dtype: torch.dtype,
        device: DeviceLike,
    ) -> None:
        super().__init__()
        self.config = config
        heads, width = config.nheads, config.d_state
        self.in_proj = nn.Linear(
            config.d_model, sum(config.in_proj_sizes), bias=False,
            dtype=dtype, device=device,
        )
        # Upstream creates these four in fp32 whatever the model's
        # dtype, and so does this.
        self.dt_bias = nn.Parameter(_fp32(device, heads))
        self.B_bias = nn.Parameter(_fp32(device, heads, 1, width))
        self.C_bias = nn.Parameter(_fp32(device, heads, 1, width))
        self.D = nn.Parameter(_fp32(device, heads))
        self.B_norm = RMSNorm(width, dtype=dtype, device=device)
        self.C_norm = RMSNorm(width, dtype=dtype, device=device)
        self.out_proj = nn.Linear(
            config.d_inner, config.d_model, bias=False,
            dtype=dtype, device=device,
        )

    def project(self, u: Tensor) -> CoreInputs:
        """Hidden states (batch, length, d_model) to recurrence
        inputs, as upstream's `Mamba3.forward` builds them for its
        kernel."""
        c = self.config
        batch, length, _ = u.shape
        split = torch.split(self.in_proj(u), c.in_proj_sizes, dim=-1)
        z, x, b, cc, dd_dt, dd_a, trap, angles = split
        a = torch.clamp(-_heavy_tail(dd_a.float()), max=-c.a_floor)
        dt = F.softplus(dd_dt.float() + self.dt_bias)
        per_head = (batch, length, c.nheads, c.headdim)
        rotary = (batch, length, c.nheads, c.num_rope_angles)
        return CoreInputs(
            q=self.C_norm(cc, cc.dtype).reshape(batch, length, 1, -1),
            k=self.B_norm(b, b.dtype).reshape(batch, length, 1, -1),
            v=x.reshape(per_head),
            adt=(a * dt).transpose(1, 2),
            dt=dt.transpose(1, 2),
            trap=trap.transpose(1, 2),
            q_bias=self.C_bias.squeeze(1),
            k_bias=self.B_bias.squeeze(1),
            angles=angles.float().unsqueeze(2).expand(rotary),
            d=self.D,
            z=z.reshape(per_head),
        )

    def finish(self, y: Tensor, dtype: torch.dtype) -> Tensor:
        """The recurrence's output, (batch, length, heads, headdim),
        back to the model width, cast first as upstream casts it."""
        batch, length = y.shape[:2]
        return self.out_proj(y.reshape(batch, length, -1).to(dtype))

    def forward(
        self, u: Tensor, state: LayerState, core: Core
    ) -> Tuple[Tensor, LayerState]:
        y, state = core(self.project(u), state)
        return self.finish(y, u.dtype), state


class Mamba3Block(nn.Module):
    """Norm, mixer, norm, MLP, around an fp32 residual stream.

    Upstream adds before it normalises, returning the residual beside
    the output, so its MLP output joins the stream in the next block.
    Adding it at the end of this block instead is the same sum.
    """

    def __init__(
        self,
        config: Mamba3Config,
        *,
        dtype: torch.dtype,
        device: DeviceLike,
    ) -> None:
        super().__init__()
        width = config.d_model
        self.norm = RMSNorm(width, dtype=dtype, device=device)
        self.mixer = Mamba3Mixer(config, dtype=dtype, device=device)
        self.norm2 = RMSNorm(width, dtype=dtype, device=device)
        self.mlp = GatedMLP(config, dtype=dtype, device=device)

    def forward(
        self,
        residual: Tensor,
        state: LayerState,
        core: Core,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, LayerState]:
        normed = self.norm(residual, dtype)
        mixed, state = self.mixer(normed, state, core)
        residual = residual + mixed.float()
        grown = self.mlp(self.norm2(residual, dtype))
        return residual + grown.float(), state


class Mamba3Backbone(nn.Module):
    """The embedding, the blocks and the final norm."""

    def __init__(
        self,
        config: Mamba3Config,
        *,
        dtype: torch.dtype,
        device: DeviceLike,
    ) -> None:
        super().__init__()
        width = config.d_model
        self.embedding = nn.Embedding(
            config.vocab_size, width, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            Mamba3Block(config, dtype=dtype, device=device)
            for _ in range(config.n_layer)
        )
        self.norm_f = RMSNorm(width, dtype=dtype, device=device)


class Mamba3LM(nn.Module):
    """The model: a backbone, and a head tied to the embedding."""

    def __init__(
        self,
        config: Mamba3Config,
        *,
        dtype: torch.dtype = torch.float32,
        device: DeviceLike = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = dtype
        self.backbone = Mamba3Backbone(
            config, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False,
            dtype=dtype, device=device,
        )
        self.tie()

    def tie(self) -> None:
        """Share the head's weight with the embedding."""
        self.lm_head.weight = self.backbone.embedding.weight

    def empty_states(self, batch: int) -> List[LayerState]:
        """Every layer's state before the first token."""
        assert batch > 0, "a state needs at least one batch row"
        c = self.config
        device = self.lm_head.weight.device
        shapes = (
            (batch, c.nheads, c.num_rope_angles),
            (batch, c.nheads, c.headdim, c.d_state),
            (batch, c.nheads, c.d_state),
            (batch, c.nheads, c.headdim),
        )
        return [
            LayerState(*(_zeros(device, shape) for shape in shapes))
            for _ in range(c.n_layer)
        ]

    def forward(
        self,
        input_ids: Tensor,
        states: Sequence[LayerState],
        core: Core = recur_sequence,
        capture: Optional[List[List[StepTerms]]] = None,
    ) -> Tuple[Tensor, List[LayerState]]:
        """Logits (batch, length, vocab) in fp32, and the new states.

        `capture`, one list per layer, collects every step's terms. It
        needs this module's own recurrence, the only core that exposes
        them.
        """
        assert len(states) == self.config.n_layer, "a state per layer"
        if capture is not None:
            assert core is recur_sequence, "capture needs our core"
            assert len(capture) == len(states), "a list per layer"
        dtype = self.compute_dtype
        residual = self.backbone.embedding(input_ids).float()
        after: List[LayerState] = []
        for index, layer in enumerate(self.backbone.layers):
            layer_core = core
            if capture is not None:
                layer_core = functools.partial(
                    recur_sequence, sink=capture[index]
                )
            residual, state = layer(
                residual, states[index], layer_core, dtype
            )
            after.append(state)
        hidden = self.backbone.norm_f(residual, dtype)
        return self.lm_head(hidden).float(), after


def _fp32(device: DeviceLike, *shape: int) -> Tensor:
    return torch.ones(*shape, dtype=torch.float32, device=device)


def _zeros(device: torch.device, shape: Tuple[int, ...]) -> Tensor:
    return torch.zeros(*shape, dtype=torch.float32, device=device)


# -- loading a checkpoint --


def load(
    checkpoint_dir: Path, *, device: DeviceLike, dtype: torch.dtype
) -> Mamba3LM:
    """Build the model a checkpoint describes, and load it strictly.

    Built on the meta device and then allocated, so no memory or time
    goes on initialising weights the checkpoint is about to replace.
    The weights are memory-mapped rather than read whole.
    """
    config = read_config(checkpoint_dir)
    weights = torch.load(
        checkpoint_dir / WEIGHTS_NAME,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model = Mamba3LM(config, dtype=dtype, device="meta")
    model.to_empty(device=device)
    model.tie()
    load_weights(model, weights)
    return model.eval()


def load_weights(model: Mamba3LM, weights: Dict[str, Tensor]) -> None:
    """Load weights under upstream's names, refusing any mismatch.

    A checkpoint saved without its tied head is completed from the
    embedding: the one omission that loses nothing.
    """
    head, embedding = "lm_head.weight", "backbone.embedding.weight"
    if head not in weights and embedding in weights:
        weights = {**weights, head: weights[embedding]}
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        message = f"weights do not fit: {error}"
        raise Mamba3LoadError(message) from error
    if result.missing_keys or result.unexpected_keys:
        raise Mamba3LoadError(
            f"missing {sorted(result.missing_keys)}, unexpected"
            f" {sorted(result.unexpected_keys)}"
        )
