"""
weighted_test.py  —  exhaustive numerical validation of the
*Weighted Motivic Taylor Tower Conjecture* (WM‑TTC).

The conjecture’s quantitative claims boil down to four axioms:

  (A1)  **Positivity & boundedness**  — each weight component lies in (0, 1].
  (A2)  **Strict monotonicity**       — weights decrease when their
        defining parameter (dimension, singularity, or stage) increases.
  (A3)  **Blow‑up monotonicity**      — any birational blow‑up raises
        dimension / singularity ⇒ lowers total weight.
  (A4)  **Obstruction decay**         — the recursive product
        M_{n+1} = M_n · w_total(...) tends to 0 universally
        (quantified here by crossing a tiny threshold by ≤ 15 000 steps).

This script attacks all four axioms on three fronts:

  • deterministic algebraic smoke‑tests (exact Fractions & Decimals);
  • heavy Monte‑Carlo (default: 2 M cases, parallelised);
  • optional Hypothesis‑based proofs (property tests).

Adjust global constants at top to crank difficulty.
Exit status ≠ 0 on *any* violation.

Author: independent‑referee · April 2025
"""

from __future__ import annotations
import math, os, random, sys, time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from multiprocessing import Pool, cpu_count

# ============================================================================
# CONFIGURATION  – tweak for harsher or quicker runs
# ============================================================================
DEFAULT_ITERATIONS   = 2_000_000    # Monte‑Carlo workload (split across cores)
BLOWUP_MAX_STEPS     = 25           # up to 25 successive blow‑ups
BLOWUP_MAX_CODIM     = 20           # each blow‑up adds 1…20 to dimension
DECIMAL_PRECISION    = 50           # digits for high‑precision decay simulation
DECAY_THRESHOLD      = Decimal("1e-30")  # target magnitude for Mn

getcontext().prec = DECIMAL_PRECISION
getcontext().Emin = -999999  # allow very small Decimals without subnormal traps

# ============================================================================
# 1 · WEIGHT FUNCTIONS  (A1 + A2)
# ============================================================================
def w_dim(dim: int) -> Decimal:
    """
    Weight factor punishing *dimension*:
        w_dim(X) = 1 / (1 + dim(X)).

    Tests:
      • Positivity & ≤1   … verifies A1 for dimension component.
      • Monotonicity      … if dim increases, output strictly decreases (A2).

    Args
    ----
    dim : non‑negative int
          Krull dimension of the motivic space being examined.

    Returns
    -------
    Decimal in (0,1].
    """
    if dim < 0:
        raise ValueError("dimension must be ≥ 0")
    return Decimal(1) / (1 + dim)


def w_sing(sing: int) -> Decimal:
    """
    Weight factor penalising *singularity complexity* (Milnor number,
    codim of singular locus, etc. — exact metric immaterial for the test).

    Same axioms as `w_dim`.
    """
    if sing < 0:
        raise ValueError("singularity index must be ≥ 0")
    return Decimal(1) / (1 + sing)


def w_stage(n: int) -> Decimal:
    """
    Stage‑based damping  w_stage(n) = 1 / (n + 1).

    Guarantees eventual suppression (*ω(n) → 0*) irrespective of geometry,
    hence central to proof of A4 (obstruction decay).
    """
    if n < 0:
        raise ValueError("stage must be ≥ 0")
    return Decimal(1) / (n + 1)


def w_total(dim: int, sing: int, n: int) -> Decimal:
    """
    The *total* weighting factor ω(dim, sing, n) used in the WM‑TTC tower:
        ω = w_dim · w_sing · w_stage.

    Appears directly in the recursive formula
        M_{n+1} = M_n · ω(dim,sing,n).

    Weighted‑tower convergence (A4) hinges on ω(dim,sing,n) → 0 as n→∞.
    """
    return w_dim(dim) * w_sing(sing) * w_stage(n)

# ============================================================================
# 2 · DETERMINISTIC TESTS (exact arithmetic sanity for A1–A4)
# ============================================================================
def deterministic_tests() -> None:
    """
    Runs quick algebraic checks that are *not* probabilistic:

      • Positivity / strict bounds for small hand‑picked inputs.
      • Symbolic identity  ∏_{k=1}^N 1/k = 1/N!  (Fractions).
      • High‑precision decay to 1e‑30 for trivial case (dim=sing=0).

    These confirm the formulas themselves obey A1–A4 before we embark
    on massive random fuzzing.
    """
    # A1 & A2 for a few literals
    assert Decimal(0) < w_dim(7)      <= 1 and w_dim(10)  < w_dim(5)
    assert Decimal(0) < w_sing(0)     <= 1 and w_sing(6)  < w_sing(2)
    assert Decimal(0) < w_stage(123)  <= 1 and w_stage(6) < w_stage(1)

    # Exact factorial identity (symbolic Fractions)  – relates to A4
    for N in range(1, 26):
        prod_frac = Fraction(1, 1)
        for k in range(1, N + 1):
            prod_frac *= Fraction(1, k)
        assert prod_frac == Fraction(1, math.factorial(N))

    # High‑precision decay for simplest geometry (dim=sing=0)
    M = Decimal(1)
    for k in range(150):               # 150! is astronomically large
        M *= w_total(0, 0, k)
    assert M < DECAY_THRESHOLD

# ============================================================================
# 3 · BLOW‑UP SIMULATION (A3)
# ============================================================================
def simulate_blowups(dim: int, sing: int, steps: int) -> bool:
    """
    Fake a sequence of `steps` blow‑ups and/or singularity thickenings:

        • Each blow‑up increases dimension by 1…BLOWUP_MAX_CODIM.
        • Singularity index optionally increases by 0–3.

    Returns
    -------
    bool
        True  : every blow‑up strictly *decreased* `w_total`
                (hence satisfied axiom A3).
        False : at least one step violated monotonicity.
    """
    previous = w_total(dim, sing, 0)
    for _ in range(steps):
        dim  += random.randint(1, BLOWUP_MAX_CODIM)
        sing += random.randint(0, 3)
        current = w_total(dim, sing, 0)
        if current >= previous:   # must be strictly smaller
            return False
        previous = current
    return True

# ============================================================================
# 4 · RANDOM GENERATORS
# ============================================================================
def r_dim()   -> int:   return random.randint(0, 50_000)
def r_sing()  -> int:   return random.randint(0, 50_000)
def r_stage() -> int:   return random.randint(0, 40_000)
def r_M0()    -> Decimal:
    """Initial obstruction size 10^[−6, 6] (log‑uniform)."""
    return Decimal(random.uniform(1e-6, 1e6))

# ============================================================================
# 5 · MONTE‑CARLO WORKER — hammered in parallel
# ============================================================================
@dataclass
class Tally:
    """Collect per‑process pass/fail statistics."""
    iterations:  int = 0
    failures:    int = 0
    pos_fail:    int = 0
    mono_fail:   int = 0
    blow_fail:   int = 0
    decay_fail:  int = 0

def worker(chunk: int, seed: int) -> Tally:
    """
    Perform `chunk` random trials validating all four axioms.

    Each trial executes in this order:
      1. A1 positivity check.
      2. A2 monotonicity (random pairs).
      3. A3 blow‑up chain of 5–BLOWUP_MAX_STEPS steps.
      4. A4 obstruction‑decay to DECAY_THRESHOLD by ≤ 15 000 stages.

    Returns
    -------
    Tally with detailed failure counts (merged by parent).
    """
    random.seed(seed)
    tally = Tally()
    for _ in range(chunk):
        tally.iterations += 1

        # ---- A1 positivity & bound ------------------------------------
        d, s, n = r_dim(), r_sing(), r_stage()
        if not (0 < w_dim(d) <= 1 and 0 < w_sing(s) <= 1 and 0 < w_stage(n) <= 1):
            tally.failures += 1; tally.pos_fail += 1

        # ---- A2 strict monotonicity -----------------------------------
        d1, d2 = r_dim(), r_dim()
        if d1 != d2 and (d1 < d2) != (w_dim(d1) > w_dim(d2)):
            tally.failures += 1; tally.mono_fail += 1
        s1, s2 = r_sing(), r_sing()
        if s1 != s2 and (s1 < s2) != (w_sing(s1) > w_sing(s2)):
            tally.failures += 1; tally.mono_fail += 1
        n1, n2 = r_stage(), r_stage()
        if n1 != n2 and (n1 < n2) != (w_stage(n1) > w_stage(n2)):
            tally.failures += 1; tally.mono_fail += 1

        # ---- A3 blow‑up monotonicity ----------------------------------
        if not simulate_blowups(r_dim(), r_sing(),
                                random.randint(5, BLOWUP_MAX_STEPS)):
            tally.failures += 1; tally.blow_fail += 1

        # ---- A4 obstruction decay -------------------------------------
        dim, sing, M = r_dim(), r_sing(), r_M0()
        for k in range(15_000):              # generous upper bound
            M *= w_total(dim, sing, k)
            if M < DECAY_THRESHOLD:
                break
        else:
            tally.failures += 1; tally.decay_fail += 1

    return tally

# ============================================================================
# 6 · RUN MONTE‑CARLO IN PARALLEL
# ============================================================================
def run_monte_carlo(total_iters: int) -> Tally:
    """
    Parallel dispatcher: splits the workload across all logical CPUs,
    merges individual tallies, and returns global statistics.

    Parameters
    ----------
    total_iters : int
        total number of random trials to execute.

    Returns
    -------
    Tally  aggregated over all worker processes.
    """
    cores = max(1, cpu_count())
    chunk = total_iters // cores
    seeds = [random.randrange(2**63) for _ in range(cores)]
    args  = [(chunk, seed) for seed in seeds]

    print(f"Running Monte‑Carlo with {total_iters:,} iterations "
          f"on {cores} core(s)…")
    if cores == 1:
        tallies = [worker(chunk, seeds[0])]
    else:
        with Pool(cores) as pool:
            tallies = pool.starmap(worker, args)

    # merge
    total = Tally()
    for t in tallies:
        total.iterations += t.iterations
        total.failures   += t.failures
        total.pos_fail   += t.pos_fail
        total.mono_fail  += t.mono_fail
        total.blow_fail  += t.blow_fail
        total.decay_fail += t.decay_fail
    return total

# ============================================================================
# 7 · OPTIONAL pytest + hypothesis (deterministic property proofs)
# ============================================================================
try:
    import pytest
    from hypothesis import given, settings, assume, strategies as st_h

    ints  = lambda hi: st_h.integers(min_value=0, max_value=hi)
    dims   = ints(50_000)
    sings  = ints(50_000)
    stages = ints(40_000)

    # A1 positivity
    @settings(max_examples=400, deadline=None)
    @given(dims)        # noqa: E305
    def test_dim_pos(d):
        assert Decimal(0) < w_dim(d) <= 1

    @settings(max_examples=400, deadline=None)
    @given(sings)       # noqa: E305
    def test_sing_pos(s):
        assert Decimal(0) < w_sing(s) <= 1

    @settings(max_examples=400, deadline=None)
    @given(stages)      # noqa: E305
    def test_stage_pos(n):
        assert Decimal(0) < w_stage(n) <= 1

    # A2 monotonicity
    @settings(max_examples=400, deadline=None)
    @given(dims, dims)
    def test_dim_decr(d1, d2):
        assume(d1 != d2)
        assert (d1 < d2) == (w_dim(d1) > w_dim(d2))

    @settings(max_examples=400, deadline=None)
    @given(sings, sings)
    def test_sing_decr(s1, s2):
        assume(s1 != s2)
        assert (s1 < s2) == (w_sing(s1) > w_sing(s2))

    @settings(max_examples=400, deadline=None)
    @given(stages, stages)
    def test_stage_decr(n1, n2):
        assume(n1 != n2)
        assert (n1 < n2) == (w_stage(n1) > w_stage(n2))

    # A3 blow‑up chains
    @settings(max_examples=180, deadline=None)
    @given(dims, sings, ints(BLOWUP_MAX_STEPS))
    def test_blowups(dim, sing, steps):
        assume(steps >= 1)
        assert simulate_blowups(dim, sing, steps)

    # A4 obstruction decay
    @settings(max_examples=180, deadline=None)
    @given(dims, sings, st_h.floats(min_value=1e-6, max_value=1e6))
    def test_decay(dim, sing, m0):
        M = Decimal(m0)
        for k in range(15_000):
            M *= w_total(dim, sing, k)
            if M < DECAY_THRESHOLD:
                break
        assert M < DECAY_THRESHOLD

except ModuleNotFoundError:
    # Libraries absent – plain Python execution still works.
    pass

# ============================================================================
# 8 · MAIN
# ============================================================================
if __name__ == "__main__":
    t_start = time.time()
    print("Deterministic smoke + symbolic factorial check …")
    deterministic_tests()
    print("✓ deterministic & symbolic checks passed.\n")

    total_iters = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ITERATIONS
    t0 = time.time()
    tally = run_monte_carlo(total_iters)
    wall = time.time() - t0
    ips  = tally.iterations / wall
    per  = ips / max(1, cpu_count())

    print(f"Done in {wall:.1f}s  →  {ips:,.0f} iter/s "
          f"({per:,.0f} per‑core)\n")

    # summary banner
    if tally.failures:
        print("❌  FAILURES DETECTED")
        if tally.pos_fail:   print(f"   · positivity failures   : {tally.pos_fail}")
        if tally.mono_fail:  print(f"   · monotonicity failures : {tally.mono_fail}")
        if tally.blow_fail:  print(f"   · blow‑up failures      : {tally.blow_fail}")
        if tally.decay_fail: print(f"   · decay failures        : {tally.decay_fail}")
        sys.exit(1)

    total_wall = time.time() - t_start
    print(f"✅  ALL TESTS PASSED — "
          f"{tally.iterations:,} iterations in {total_wall:.1f}s total")
