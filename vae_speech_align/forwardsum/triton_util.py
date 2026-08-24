from dataclasses import dataclass

import triton

# The autotune key is K_next_power_of_2, so the candidates spread
# BLOCK_SIZE_K to let each K bucket pick a block close to its own
# size; oversized blocks waste most lanes on the typical K of a few
# hundred states. num_stages stays at 2 because the DP synchronizes
# every time step, leaving nothing to pipeline.
configs_K = [
    triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
]


@dataclass
class TritonConfigK:
    BLOCK_SIZE_K: int = 1024
    num_warps: int = 4
    num_stages: int = 2
