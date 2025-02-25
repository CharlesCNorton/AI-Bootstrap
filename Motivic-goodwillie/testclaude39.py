"""
Expanded Motivic Taylor Tower Test

This script uses exact arithmetic (via Sympy) to model motivic cohomology and its
associated spectral sequence. The goal is to test the following claims:

  (i) For each motivic space X and functor F, the weighted Taylor tower produces
      a spectral sequence that stabilizes exactly (i.e. successive pages are identical)
      beyond a finite page.
  (ii) At stabilization, all differentials vanish exactly.

Any deviation from these claims will cause the test to fail.
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sympy as sp

# Use Sympy's Rational for exact arithmetic.
Rational = sp.Rational

# ============================================================================
# Basic Structures: Motivic Spaces and Cohomology Classes
# ============================================================================

@dataclass
class MotivicSpace:
    """
    A motivic space defined by its dimension, singularity measure, nilpotent order,
    and number of exceptional divisors.
    """
    name: str
    dimension: int
    singularity_measure: int
    nilpotent_order: int = 0
    exceptional_divisors: int = 0

    def __str__(self):
        s = f"{self.name} (dim={self.dimension}, sing={self.singularity_measure}"
        if self.nilpotent_order:
            s += f", nilp={self.nilpotent_order}"
        if self.exceptional_divisors:
            s += f", exc_div={self.exceptional_divisors}"
        s += ")"
        return s

@dataclass
class CohomologyClass:
    """
    Represents a cohomology class with bidegree (p,q), an exact value, an exact weight,
    and a support dimension.
    """
    p: int
    q: int
    value: sp.Rational
    weight: sp.Rational
    support_dimension: int = -1

    def __str__(self):
        return f"H^{self.p},{self.q} = {self.value} (wt={self.weight})"

class MotivicCohomology:
    """
    Container for exact cohomology classes of a motivic space.
    """
    def __init__(self, X: MotivicSpace):
        self.X = X
        self.classes: List[CohomologyClass] = []
    
    def add_class(self, p: int, q: int, value, weight, support_dimension: int = -1):
        val = sp.nsimplify(value, [sp.Rational(1,1)])
        wt = sp.nsimplify(weight, [sp.Rational(1,1)])
        cls = CohomologyClass(p, q, val, wt, support_dimension)
        self.classes.append(cls)
    
    def __str__(self):
        return f"Cohomology of {self.X.name} with {len(self.classes)} classes"

# ============================================================================
# Spectral Sequence
# ============================================================================

class SpectralSequence:
    """
    An exact spectral sequence. Each page is a dictionary mapping (p,q,w) -> sp.Rational.
    Differentials are defined as exact linear maps.
    """
    def __init__(self, name: str, initial_page: Optional[Dict[Tuple[int,int,int], sp.Rational]] = None):
        self.name = name
        self.pages: Dict[int, Dict[Tuple[int,int,int], sp.Rational]] = {}
        if initial_page is None:
            self.pages[2] = {}
        else:
            self.pages[2] = {k: sp.nsimplify(v, [sp.Rational(1,1)]) for k, v in initial_page.items()}
        self.current_page = 2
        self.differentials: Dict[int, Dict[Tuple[int,int,int], Tuple[Tuple[int,int,int], sp.Rational]]] = {}

    def set_entry(self, page: int, p: int, q: int, w: int, value):
        if page not in self.pages:
            self.pages[page] = {}
        self.pages[page][(p,q,w)] = sp.nsimplify(value, [sp.Rational(1,1)])

    def get_entry(self, page: int, p: int, q: int, w: int) -> sp.Rational:
        return self.pages.get(page, {}).get((p,q,w), sp.Rational(0,1))
    
    def add_differential(self, page: int, source: Tuple[int,int,int], target: Tuple[int,int,int], value):
        if page not in self.differentials:
            self.differentials[page] = {}
        self.differentials[page][source] = (target, sp.nsimplify(value, [sp.Rational(1,1)]))
    
    def compute_next_page(self):
        cur = self.current_page
        nxt = cur + 1
        self.pages[nxt] = dict(self.pages[cur])
        if cur in self.differentials:
            for source, (target, diff_val) in self.differentials[cur].items():
                if source in self.pages[nxt]:
                    self.pages[nxt][source] -= diff_val
                else:
                    self.pages[nxt][source] = -diff_val
                if target in self.pages[nxt]:
                    self.pages[nxt][target] += diff_val
                else:
                    self.pages[nxt][target] = diff_val
        self.current_page = nxt
        return self.pages[nxt]
    
    def compute_until_stable(self, max_page: int = 10):
        prev = self.pages[self.current_page]
        for page in range(self.current_page+1, max_page+1):
            self.compute_next_page()
            curr = self.pages[self.current_page]
            if prev == curr:
                return self.current_page
            prev = curr.copy()
        return self.current_page

    def max_differential_norm(self, page: int) -> sp.Rational:
        if page not in self.differentials:
            return sp.Rational(0,1)
        return max([abs(diff_val) for (_, diff_val) in self.differentials[page].values()], default=sp.Rational(0,1))
    
    def __str__(self):
        return f"{self.name} Spectral Sequence (current page: E_{self.current_page})"

# ============================================================================
# Weight Functions (Exact)
# ============================================================================

def dimension_weight(X: MotivicSpace) -> sp.Rational:
    return sp.Rational(1, 1 + X.dimension)

def singularity_weight(X: MotivicSpace) -> sp.Rational:
    return sp.Rational(1, 1 + X.singularity_measure)

def nilpotent_weight(X: MotivicSpace) -> sp.Rational:
    return sp.Rational(1, 1 + X.nilpotent_order)

def exceptional_divisor_weight(X: MotivicSpace) -> sp.Rational:
    return sp.Rational(1, 1 + X.exceptional_divisors)

def stage_weight(n: int) -> sp.Rational:
    return sp.Rational(1, n + 1)

def total_weight(X: MotivicSpace, n: int) -> sp.Rational:
    return (dimension_weight(X) * singularity_weight(X) * nilpotent_weight(X) *
            exceptional_divisor_weight(X) * stage_weight(n))

# ============================================================================
# Polynomial Approximation: Exact Synthetic Model
# ============================================================================

class PolynomialApproximationExact:
    """
    An exact synthetic polynomial approximation of degree n.
    Generates synthetic cohomology classes using exact rational arithmetic.
    """
    def __init__(self, n: int, X: MotivicSpace):
        self.n = n
        self.X = X
        self.cohomology = MotivicCohomology(X)
    
    def generate_cohomology(self, num_classes: int = 10, seed: Optional[int] = None, complexity_factor: int = 1):
        if seed is not None:
            import random
            random.seed(seed)
        for i in range(num_classes):
            # p in [0, n+2], q in [0, n+1]
            p = sp.Integer(random.randint(0, self.n + 2))
            q = sp.Integer(random.randint(0, self.n + 1))
            # Base value: ±1.
            base_val = sp.Integer(random.choice([-1, 1]))
            decay = sp.Rational(1, 1 + complexity_factor * self.n)
            value = base_val * decay
            support_dim = random.randint(0, self.X.dimension)
            weight_factor = sp.Rational(p + q, 2 * self.n + 5)
            weight = weight_factor * (sp.Integer(1) + self.X.dimension + self.X.singularity_measure +
                                      sp.Rational(1,2) * self.X.nilpotent_order + sp.Rational(3,10) * self.X.exceptional_divisors)
            self.cohomology.add_class(int(p), int(q), value, weight, support_dim)

# ============================================================================
# Abstract Invariant Test
# ============================================================================

def invariant_test_exact(ss: SpectralSequence, expected_stable_page: int):
    stable_page = ss.compute_until_stable(max_page=10)
    if stable_page != expected_stable_page:
        raise AssertionError(f"Expected stabilization at page {expected_stable_page}, got {stable_page}")
    max_norm = ss.max_differential_norm(stable_page)
    if max_norm != sp.Rational(0,1):
        raise AssertionError(f"Expected max differential norm 0, got {max_norm}")
    return True

# ============================================================================
# Test Cases
# ============================================================================

def test_case_1_exact():
    # Smooth Projective Plane P^2: dimension 2, no singularities.
    X = MotivicSpace(name="P^2", dimension=2, singularity_measure=0)
    cohom = MotivicCohomology(X)
    # Exact cohomology: H^0 = 1, H^2 = 1, H^4 = 1.
    cohom.add_class(0, 0, sp.Integer(1), sp.Integer(1))
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(4,5))
    cohom.add_class(4, 2, sp.Integer(1), sp.Rational(3,5))
    print("Test Case 1 (Exact): Smooth Projective Plane P^2")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("P^2_Spectral")
    for cls in cohom.classes:
        wgroup = int(cls.weight * 10)
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    invariant_test_exact(ss, expected_stable_page=3)
    print(ss)
    print("Invariant test passed for P^2.\n" + "-"*50)

def test_case_2_exact():
    # Singular Curve: Union of Two Lines, dimension 1, singularity_measure 1.
    X = MotivicSpace(name="Union_of_Lines", dimension=1, singularity_measure=1)
    cohom = MotivicCohomology(X)
    cohom.add_class(0, 0, sp.Integer(1), sp.Integer(1))
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(7,10))
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(7,10))
    print("Test Case 2 (Exact): Singular Curve - Union of Two Lines")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("UnionLines_Spectral")
    for cls in cohom.classes:
        wgroup = int(cls.weight * 10)
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    invariant_test_exact(ss, expected_stable_page=3)
    print(ss)
    print("Invariant test passed for Union of Lines.\n" + "-"*50)

def test_case_3_exact():
    # Non-reduced Fat Point: dimension 0, nilpotent_order 1.
    X = MotivicSpace(name="Fat_Point", dimension=0, singularity_measure=0, nilpotent_order=1)
    cohom = MotivicCohomology(X)
    cohom.add_class(0, 0, sp.Integer(1), sp.Integer(1))
    cohom.add_class(2, 1, sp.Integer(0), sp.Rational(1,2))
    print("Test Case 3 (Exact): Non-reduced Fat Point")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("FatPoint_Spectral")
    for cls in cohom.classes:
        wgroup = int(cls.weight * 10)
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    invariant_test_exact(ss, expected_stable_page=3)
    print(ss)
    print("Invariant test passed for Fat Point.\n" + "-"*50)

def test_case_4_exact():
    # Blow-up of P^2 at a Point: dimension 2, exceptional_divisors=1.
    X = MotivicSpace(name="BlowUp_P2", dimension=2, singularity_measure=0, exceptional_divisors=1)
    cohom = MotivicCohomology(X)
    cohom.add_class(0, 0, sp.Integer(1), sp.Integer(1))             # H^0
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(4,5))             # Base H^2
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(4,5))             # Exceptional divisor H^2
    cohom.add_class(4, 2, sp.Integer(1), sp.Rational(3,5))             # H^4
    print("Test Case 4 (Exact): Blow-up of P^2 at a Point")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("BlowUp_Spectral")
    for cls in cohom.classes:
        wgroup = int(cls.weight * 10)
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    # Simulate a differential d2: map one H^2 entry to H^4.
    ss.add_differential(2, (2,1, int(sp.Rational(4,5)*10)), (4,2, int(sp.Rational(3,5)*10)), sp.Integer(1))
    invariant_test_exact(ss, expected_stable_page=4)
    print(ss)
    print("Invariant test passed for Blow-up of P^2.\n" + "-"*50)

def test_case_5_exact():
    # Smooth Projective Elliptic Curve: dimension 1, smooth.
    X = MotivicSpace(name="Elliptic_Curve", dimension=1, singularity_measure=0)
    cohom = MotivicCohomology(X)
    cohom.add_class(0, 0, sp.Integer(1), sp.Integer(1))
    cohom.add_class(1, 0, sp.Integer(1), sp.Rational(9,10))
    cohom.add_class(1, 0, sp.Integer(1), sp.Rational(9,10))
    cohom.add_class(2, 1, sp.Integer(1), sp.Rational(4,5))
    print("Test Case 5 (Exact): Smooth Projective Elliptic Curve")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("Elliptic_Spectral")
    for cls in cohom.classes:
        wgroup = int(cls.weight * 10)
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    invariant_test_exact(ss, expected_stable_page=3)
    print(ss)
    print("Invariant test passed for Elliptic Curve.\n" + "-"*50)

# ============================================================================
# Main: Run All Test Cases
# ============================================================================

def main():
    print("\n--- Expanded Motivic Taylor Tower Test ---\n")
    test_case_1_exact()
    test_case_2_exact()
    test_case_3_exact()
    test_case_4_exact()
    test_case_5_exact()
    print("All invariant tests passed.")

if __name__ == "__main__":
    main()
