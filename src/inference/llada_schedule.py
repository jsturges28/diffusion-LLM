"""Dependency-light validation for LLaDA's block schedule.

The supervisor validates durable generation snapshots before it
publishes a branch, while the worker validates the same relationship
before inference. Keeping the arithmetic here lets both boundaries use
one rule without making the supervisor import torch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSchedule:
    """How a run is divided into blocks, and steps within a block."""

    num_blocks: int
    steps_per_block: int

    @property
    def total_steps(self) -> int:
        return self.num_blocks * self.steps_per_block


def block_schedule(
    *, gen_length: int, block_length: int, steps: int
) -> BlockSchedule:
    """Divide one canvas and its steps without discarding either."""
    if gen_length <= 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be positive"
        )
    if block_length <= 0:
        raise ValueError(
            f"block_length ({block_length}) must be positive"
        )
    if steps <= 0:
        raise ValueError(f"steps ({steps}) must be positive")
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be"
            f" divisible by block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    assert num_blocks > 0
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by"
            f" num_blocks ({num_blocks})"
        )
    schedule = BlockSchedule(
        num_blocks=num_blocks,
        steps_per_block=steps // num_blocks,
    )
    assert schedule.steps_per_block > 0
    assert schedule.total_steps == steps
    return schedule
