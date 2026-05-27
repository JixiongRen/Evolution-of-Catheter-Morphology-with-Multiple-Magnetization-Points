from __future__ import annotations

from typing import List


def apply_advancer_protrude_length(
    *,
    flex_lengths_in: List[float],
    rigid_lengths: List[float],
    L_protrude: float,
    L1_min: float = 1e-6,
) -> List[float]:
    """Paper-aligned advancer model.

    Insertion/retraction changes ONLY the first flexible segment length.

    Semantics:
      - The solver models the portion of the catheter protruding from the sheath exit.
      - L_protrude is the total protruding length (flex + rigid) starting at the sheath exit.
      - All segments except the first flexible segment keep their configured lengths.
      - The first flexible segment length is inferred so that the total equals L_protrude.

    Inputs:
      - flex_lengths_in can be either:
          * length N  : [L1_placeholder, L2, ..., LN] (L1 will be overwritten), or
          * length N-1: [L2, ..., LN] (L1 will be created)
        where N == len(rigid_lengths).

    Raises:
      - ValueError if inferred L1_new < L1_min (i.e., would require crossing a segment boundary).
    """
    if L_protrude is None:
        raise ValueError('L_protrude must be provided to apply the advancer model.')
    if float(L_protrude) <= 0.0:
        raise ValueError(f'L_protrude must be positive, got {L_protrude}.')

    N = len(rigid_lengths)
    if N <= 0:
        raise ValueError('rigid_lengths is empty; cannot infer segment count.')

    if len(flex_lengths_in) == N:
        tail = list(flex_lengths_in[1:])
    elif len(flex_lengths_in) == N - 1:
        tail = list(flex_lengths_in)
    else:
        raise ValueError(
            'When --L_protrude is set, flex_lengths must provide either N values '
            '(including a placeholder for L1) or N-1 values (L2..LN). '
            f'Got len(flex_lengths)={len(flex_lengths_in)} while N=len(rigid_lengths)={N}.'
        )

    L_fixed = float(sum(tail) + sum(rigid_lengths))
    L1_new = float(L_protrude) - L_fixed

    if L1_new < float(L1_min):
        raise ValueError(
            f'Infeasible protruding length: inferred L1_new={L1_new:.6g} m < L1_min={float(L1_min):.6g} m. '
            'This implies the requested L_protrude would require crossing a segment boundary '
            '(i.e., retracting beyond the first flexible segment).'
        )

    flex_lengths_out = [L1_new] + tail
    total = float(sum(flex_lengths_out) + sum(rigid_lengths))
    if not (abs(total - float(L_protrude)) <= 1e-9):
        raise ValueError(f'Internal length inconsistency: total={total} vs L_protrude={float(L_protrude)}.')

    return flex_lengths_out
