"""Lightweight SpinQuant-style rotation + quantization utilities.

This module intentionally avoids the full SpinQuant dependency (and its Cayley
optimization loop) by providing a simpler, maintainable alternative:

- Learn *blockwise orthogonal rotations* on weight matrices.
- Use an easy-to-audit objective: minimize the weight reconstruction error after
  fake quantization of the rotated weights.

This is not intended to be a paper-faithful reproduction of SpinQuant; rather it
is a practical engineering backend for this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RotationQuantConfig:
    """Configuration for rotation learning + (fake) quantization."""

    bits: int = 8
    block_size: int = 64
    num_steps: int = 50
    lr: float = 5e-2
    num_sweeps: int = 2
    max_layers: int = 16
    # Optional: use captured activation vectors to weight the rotation objective
    # toward preserving the actual function on calibration data.
    use_activation_objective: bool = True
    # Rotation backend selection.
    # - blockwise_givens: learned rotation via Givens sweeps (slowest, best effort)
    # - hadamard: fixed fast Hadamard rotation (no learning; very fast)
    # Default to the fast fixed rotation backend. The learned rotation backend can
    # take a very long time on large models.
    backend: str = "hadamard"
    module_name_substrings: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _iter_target_linears(model: nn.Module, name_substrings: Sequence[str]) -> Iterator[Tuple[str, nn.Linear]]:
    """Yield (name, module) pairs for matching Linear modules."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(s in name for s in name_substrings):
            yield name, module


def _symmetric_fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric fake quant (dequantized float tensor).

    This is a simple stand-in to provide a stable optimization signal.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.abs().max().clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.round(x / scale).clamp(min=-qmax - 1, max=qmax)
    return q * scale


_HADAMARD_CACHE: Dict[int, torch.Tensor] = {}


def _hadamard_matrix(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an n×n Hadamard matrix (normalized, orthogonal).

    Requires n to be a power of 2.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be power of 2, got {n}")

    if n not in _HADAMARD_CACHE:
        # Build on CPU float32 for stability.
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        H = H / (n ** 0.5)
        _HADAMARD_CACHE[n] = H
    return _HADAMARD_CACHE[n].to(device=device, dtype=dtype)


@torch.no_grad()
def apply_hadamard_rotation_and_fake_quant(
    linear: nn.Linear,
    *,
    bits: int,
    block_size: int,
) -> None:
    """In-place: apply a blockwise Hadamard rotation over input dimension and fake-quant."""
    w = linear.weight.data
    out_features, in_features = w.shape
    device = w.device
    # Compute in float32 then cast back.
    w_fp = w.detach().to(device=device, dtype=torch.float32)

    if block_size <= 0:
        block_size = in_features

    # If block_size is not power of 2, fall back to identity (no-op) to avoid errors.
    if (block_size & (block_size - 1)) != 0:
        block_size = 0

    if block_size and block_size <= in_features:
        chunks = []
        for start in range(0, in_features, block_size):
            end = min(start + block_size, in_features)
            B = end - start
            if B != block_size:
                # Tail block: skip rotation to avoid non-power-of-2 size.
                chunks.append(w_fp[:, start:end])
                continue
            H = _hadamard_matrix(B, device=device, dtype=torch.float32)
            chunks.append(w_fp[:, start:end] @ H)
        w_rot = torch.cat(chunks, dim=1)
    else:
        w_rot = w_fp

    w_q = _symmetric_fake_quant(w_rot, bits=bits)
    linear.weight.data = w_q.to(dtype=w.dtype)


@torch.no_grad()
def apply_hadamard_rotations(
    model: nn.Module,
    cfg: RotationQuantConfig,
    logger: Optional[object] = None,
) -> Dict[str, Dict[str, Any]]:
    """Apply fixed Hadamard rotation + fake quant to a subset of Linear layers."""
    summary: Dict[str, Dict[str, Any]] = {}
    num_done = 0
    for name, linear in _iter_target_linears(model, cfg.module_name_substrings):
        if num_done >= cfg.max_layers:
            break
        if logger:
            logger.info(f"[SpinQuant-hadamard] applying hadamard rotation for {name} shape={tuple(linear.weight.shape)}")

        apply_hadamard_rotation_and_fake_quant(
            linear,
            bits=int(cfg.bits),
            block_size=int(cfg.block_size),
        )
        summary[name] = {
            "out_features": int(linear.weight.shape[0]),
            "in_features": int(linear.weight.shape[1]),
            "block_size": int(cfg.block_size),
            "bits": int(cfg.bits),
            "backend": "hadamard",
        }
        num_done += 1

    if logger:
        logger.info(f"[SpinQuant-hadamard] applied hadamard rotation+fake-quant to {num_done} Linear layers")
    return summary


class BlockGivensRotation(nn.Module):
    """Blockwise Givens rotations applied to the input dimension of a weight matrix.

    Parameterization:
    - For each block of size B, store `num_sweeps` sweeps of (B-1) angles.
    - Each sweep applies adjacent-plane rotations (i, i+1).

    This is fast and avoids building a dense BxB rotation matrix.
    """

    def __init__(self, in_features: int, block_size: int, num_sweeps: int = 2):
        super().__init__()
        if block_size <= 1:
            raise ValueError("block_size must be > 1")
        self.in_features = in_features
        self.block_size = block_size
        self.num_sweeps = num_sweeps

        num_blocks = (in_features + block_size - 1) // block_size
        # angles: [num_blocks, num_sweeps, block_size-1]
        self.angles = nn.Parameter(torch.zeros(num_blocks, num_sweeps, block_size - 1))

    def forward_rotate(self, w: torch.Tensor) -> torch.Tensor:
        """Return w @ R (rotation applied on input/features dimension)."""
        if w.ndim != 2:
            raise ValueError("Expected 2D weight tensor")

        out, in_features = w.shape
        if in_features != self.in_features:
            raise ValueError(f"in_features mismatch: {in_features} vs {self.in_features}")

        device = w.device
        dtype = w.dtype
        # NOTE: We avoid in-place ops on views into `w` because they can break
        # autograd during rotation optimization (version counter mismatch).
        w_rot = w

        for b in range(self.angles.shape[0]):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, in_features)
            B = end - start
            if B <= 1:
                continue

            # Work on a cloned block to avoid view/in-place autograd issues.
            block = w_rot[:, start:end].clone()
            # Apply sweeps of adjacent rotations.
            for s in range(self.num_sweeps):
                thetas = self.angles[b, s, : B - 1].to(device=device, dtype=dtype)
                cos_t = torch.cos(thetas)
                sin_t = torch.sin(thetas)
                # Each step rotates columns i and i+1
                for i in range(B - 1):
                    c = cos_t[i]
                    si = sin_t[i]
                    # Clone the source columns before assignment to avoid
                    # reading from tensors that will be modified in-place.
                    col_i = block[:, i].clone()
                    col_j = block[:, i + 1].clone()
                    block[:, i] = c * col_i - si * col_j
                    block[:, i + 1] = si * col_i + c * col_j

            w_rot = torch.cat([w_rot[:, :start], block, w_rot[:, end:]], dim=1)

        return w_rot

    def inverse_rotate(self, w: torch.Tensor) -> torch.Tensor:
        """Return w @ R^T for the same rotation sequence.

        For a product of Givens rotations, the inverse is applying the rotations
        in reverse order with -theta.
        """
        if w.ndim != 2:
            raise ValueError("Expected 2D weight tensor")

        out, in_features = w.shape
        if in_features != self.in_features:
            raise ValueError(f"in_features mismatch: {in_features} vs {self.in_features}")

        device = w.device
        dtype = w.dtype
        # Same autograd safety note as `forward_rotate`.
        w_inv = w

        for b in range(self.angles.shape[0]):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, in_features)
            B = end - start
            if B <= 1:
                continue

            block = w_inv[:, start:end].clone()
            for s in reversed(range(self.num_sweeps)):
                thetas = (-self.angles[b, s, : B - 1]).to(device=device, dtype=dtype)
                cos_t = torch.cos(thetas)
                sin_t = torch.sin(thetas)
                for i in reversed(range(B - 1)):
                    c = cos_t[i]
                    si = sin_t[i]
                    col_i = block[:, i].clone()
                    col_j = block[:, i + 1].clone()
                    block[:, i] = c * col_i - si * col_j
                    block[:, i + 1] = si * col_i + c * col_j

            w_inv = torch.cat([w_inv[:, :start], block, w_inv[:, end:]], dim=1)

        return w_inv


@torch.no_grad()
def apply_rotation_and_fake_quant(
    linear: nn.Linear,
    rot: BlockGivensRotation,
    bits: int,
) -> None:
    """In-place: rotate weights, fake-quantize, and store dequantized weights."""
    w = linear.weight.data
    w_rot = rot.forward_rotate(w)
    w_q = _symmetric_fake_quant(w_rot, bits=bits)
    linear.weight.data = w_q


def optimize_blockwise_rotation_for_weight(
    w: torch.Tensor,
    cfg: RotationQuantConfig,
    device: Optional[torch.device] = None,
    x_samples: Optional[torch.Tensor] = None,
) -> BlockGivensRotation:
    """Learn a blockwise rotation that reduces fake quant error for a single weight matrix.

    If `x_samples` is provided, we optimize an activation-weighted proxy objective:
    E_x ||(W_rot - Q(W_rot)) x||^2, which better correlates with functional error
    than plain weight MSE.
    """
    if w.ndim != 2:
        raise ValueError("Expected 2D weight tensor")

    device = device or (w.device if w.is_cuda else torch.device("cpu"))
    w = w.detach().to(device=device, dtype=torch.float32)
    out_features, in_features = w.shape

    rot = BlockGivensRotation(in_features=in_features, block_size=cfg.block_size, num_sweeps=cfg.num_sweeps).to(device)
    opt = torch.optim.Adam([rot.angles], lr=cfg.lr)

    if x_samples is not None:
        x_samples = x_samples.detach().to(device=device, dtype=torch.float32)

    for _ in range(cfg.num_steps):
        opt.zero_grad(set_to_none=True)
        w_rot = rot.forward_rotate(w)
        w_q = _symmetric_fake_quant(w_rot, bits=cfg.bits)

        if cfg.use_activation_objective and x_samples is not None and x_samples.numel() > 0:
            # Output error proxy on sampled inputs.
            # For Linear: y = x @ W^T, so delta_y = x @ (delta_W)^T
            delta = (w_rot - w_q)
            delta_y = x_samples @ delta.t()
            loss = torch.mean(delta_y ** 2)
        else:
            # Plain weight-space MSE in rotated space (stable baseline).
            loss = torch.mean((w_rot - w_q) ** 2)
        loss.backward()
        opt.step()

    return rot


def learn_and_apply_rotations(
    model: nn.Module,
    cfg: RotationQuantConfig,
    logger: Optional[object] = None,
    activation_samples: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Learn rotations for a subset of Linear layers, then apply rotation+fake quant.

    Returns a small summary dict for logging/reporting.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    num_done = 0

    for name, linear in _iter_target_linears(model, cfg.module_name_substrings):
        if num_done >= cfg.max_layers:
            break

        # Learn rotation using the module's weight.
        w = linear.weight
        if logger:
            logger.info(f"[SpinQuant-lite] optimizing rotation for {name} shape={tuple(w.shape)}")

        x = None
        if activation_samples is not None:
            x = activation_samples.get(name)

        # Snapshot original weight for reporting.
        w_fp = w.detach().to(dtype=torch.float32)

        rot = optimize_blockwise_rotation_for_weight(w, cfg, device=w.device, x_samples=x)

        # Compute quantization reconstruction error in rotated space.
        with torch.no_grad():
            w_rot = rot.forward_rotate(w_fp)
            w_q = _symmetric_fake_quant(w_rot, bits=cfg.bits)
            mse = torch.mean((w_rot - w_q) ** 2).item()
            max_abs = float(w_rot.abs().max().item())

        apply_rotation_and_fake_quant(linear, rot, bits=cfg.bits)

        summary[name] = {
            "out_features": int(w.shape[0]),
            "in_features": int(w.shape[1]),
            "block_size": int(cfg.block_size),
            "num_sweeps": int(cfg.num_sweeps),
            "num_steps": int(cfg.num_steps),
            "bits": int(cfg.bits),
            "rotated_quant_mse": float(mse),
            "rotated_max_abs": float(max_abs),
        }
        num_done += 1

    if logger:
        logger.info(f"[SpinQuant-lite] applied rotations+fake-quant to {num_done} Linear layers")

    return summary
