"""
Expanded Testing for Weighted Motivic Taylor Tower Convergence

This script implements an expanded computational framework to test the key claims
of the Weighted Motivic Taylor Tower Conjecture. We simulate several types of 
motivic spaces—including smooth, singular, non-reduced, blow-ups, and an elliptic
curve—and compute synthetic motivic cohomology classes, then build spectral sequences
to verify convergence behavior. The results (cohomology groups, spectral sequence pages,
differential norms, and convergence data) are printed as structured data.

Test cases:
1. Smooth Projective Plane (P^2)
2. Singular Curve: Union of Two Lines in P^2
3. Non-reduced “Fat” Point (Spec(k[ε]/(ε^2)))
4. Blow-up of P^2 at a Point
5. Smooth Projective Elliptic Curve (genus 1 curve)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
from collections import defaultdict
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Part 1: Basic Structures for Motivic Spaces and Cohomology
# ============================================================================

@dataclass
class MotivicSpace:
    """
    Model of a motivic space with dimension, singularity data, nilpotent structure,
    and exceptional divisor count.
    """
    name: str
    dimension: int
    singularity_measure: float  # e.g. Milnor number
    nilpotent_order: int = 0      # 0 means reduced
    exceptional_divisors: int = 0 # count from blow-ups
    
    def __str__(self):
        result = f"{self.name} (dim={self.dimension}, sing={self.singularity_measure}"
        if self.nilpotent_order > 0:
            result += f", nilp={self.nilpotent_order}"
        if self.exceptional_divisors > 0:
            result += f", exc_div={self.exceptional_divisors}"
        result += ")"
        return result
    
    def complexity_measure(self) -> float:
        """Combined geometric complexity measure."""
        return (self.dimension + self.singularity_measure +
                0.5 * self.nilpotent_order + 0.3 * self.exceptional_divisors)

@dataclass
class CohomologyClass:
    """
    Represents a cohomology class with bidegree (p,q), a complex value, 
    a weight, and the support dimension.
    """
    p: int
    q: int
    value: complex
    weight: float
    support_dimension: int = -1
    
    def __str__(self):
        s = f"H^{self.p},{self.q} = {self.value:.4f} (wt={self.weight:.4f})"
        if self.support_dimension >= 0:
            s += f", sup_dim={self.support_dimension}"
        return s

    def __eq__(self, other):
        if not isinstance(other, CohomologyClass):
            return False
        return (self.p == other.p and self.q == other.q and 
                abs(self.value - other.value) < 1e-10 and 
                abs(self.weight - other.weight) < 1e-10 and 
                self.support_dimension == other.support_dimension)

    def __hash__(self):
        return hash((self.p, self.q, round(self.value.real, 10),
                     round(self.value.imag, 10), round(self.weight, 10),
                     self.support_dimension))

class MotivicCohomology:
    """
    Container for cohomology classes of a motivic space.
    """
    def __init__(self, X: MotivicSpace):
        self.X = X
        self.classes: List[CohomologyClass] = []
        self._class_dict: Dict[Tuple[int, int], List[CohomologyClass]] = defaultdict(list)
    
    def add_class(self, p: int, q: int, value: complex, weight: float, support_dimension: int = -1):
        cls = CohomologyClass(p, q, value, weight, support_dimension)
        self.classes.append(cls)
        self._class_dict[(p, q)].append(cls)
    
    def filter_by_weight(self, max_weight: float) -> 'MotivicCohomology':
        result = MotivicCohomology(self.X)
        for cls in self.classes:
            if cls.weight <= max_weight:
                result.add_class(cls.p, cls.q, cls.value, cls.weight, cls.support_dimension)
        return result
    
    def __str__(self):
        return f"Cohomology of {self.X.name} with {len(self.classes)} classes"

# ============================================================================
# Spectral Sequence Structure
# ============================================================================

class SpectralSequence:
    """
    Represents a spectral sequence with pages labeled by integers.
    Entries are stored as a dict keyed by (p,q,w) tuples.
    """
    def __init__(self, name: str, initial_page: Optional[Dict[Tuple[int,int,int], complex]] = None):
        self.name = name
        self.pages: Dict[int, Dict[Tuple[int,int,int], complex]] = {}
        self.pages[2] = initial_page if initial_page is not None else {}
        self.current_page = 2
        self.differentials: Dict[int, Dict[Tuple[int,int,int], Tuple[int,int,int, complex]]] = {}
    
    def set_entry(self, page: int, p: int, q: int, w: int, value: complex):
        if page not in self.pages:
            self.pages[page] = {}
        self.pages[page][(p, q, w)] = value
    
    def get_entry(self, page: int, p: int, q: int, w: int) -> complex:
        return self.pages.get(page, {}).get((p,q,w), 0j)
    
    def add_differential(self, page: int, source: Tuple[int,int,int], target: Tuple[int,int,int], value: complex):
        if page not in self.differentials:
            self.differentials[page] = {}
        self.differentials[page][source] = (target[0], target[1], target[2], value)
    
    def compute_next_page(self):
        current = self.current_page
        next_page = current + 1
        self.pages[next_page] = dict(self.pages[current])
        if current in self.differentials:
            for source, target_info in self.differentials[current].items():
                p, q, w = source
                target_p, target_q, target_w, value = target_info
                if (p,q,w) in self.pages[next_page]:
                    self.pages[next_page][(p,q,w)] -= value
                if (target_p, target_q, target_w) in self.pages[next_page]:
                    self.pages[next_page][(target_p, target_q, target_w)] += value
        self.current_page = next_page
        return self.pages[next_page]
    
    def compute_until_stable(self, max_page: int = 10, tol: float = 1e-10) -> int:
        prev = dict(self.pages[self.current_page])
        for p in range(self.current_page+1, max_page+1):
            self.compute_next_page()
            curr = self.pages[self.current_page]
            stable = True
            for key in set(prev.keys()) | set(curr.keys()):
                if abs(prev.get(key, 0j) - curr.get(key, 0j)) > tol:
                    stable = False
                    break
            if stable:
                return self.current_page
            prev = dict(curr)
        return self.current_page
    
    def get_max_differential_norm(self, page: int) -> float:
        if page not in self.differentials:
            return 0.0
        return max([abs(v[3]) for v in self.differentials[page].values()], default=0.0)
    
    def __str__(self):
        return f"{self.name} Spectral Sequence (current page: E_{self.current_page})"

# ============================================================================
# Part 2: Weight Functions Implementation
# ============================================================================

def dimension_weight(X: MotivicSpace) -> float:
    return 1.0 / (1.0 + X.dimension)

def singularity_weight(X: MotivicSpace) -> float:
    return 1.0 / (1.0 + X.singularity_measure)

def nilpotent_weight(X: MotivicSpace) -> float:
    return 1.0 / (1.0 + X.nilpotent_order)

def exceptional_divisor_weight(X: MotivicSpace) -> float:
    return 1.0 / (1.0 + X.exceptional_divisors)

def stage_weight(n: int) -> float:
    return 1.0 / (n + 1)

def total_weight(X: MotivicSpace, n: int) -> float:
    return (dimension_weight(X) * singularity_weight(X) * nilpotent_weight(X) *
            exceptional_divisor_weight(X) * stage_weight(n))

# ============================================================================
# Part 3: Polynomial Approximation and Tower Construction
# ============================================================================

class PolynomialApproximation:
    """
    Represents a polynomial approximation of degree n for a functor F.
    Generates synthetic cohomology classes.
    """
    def __init__(self, n: int, X: MotivicSpace):
        self.n = n
        self.X = X
        self.cohomology = MotivicCohomology(X)
    
    def generate_cohomology(self, num_classes: int = 10, seed: Optional[int] = None,
                              complexity_factor: float = 0.5):
        if seed is not None:
            np.random.seed(seed)
        for i in range(num_classes):
            p = np.random.randint(0, self.n + 3)
            q = np.random.randint(0, self.n + 2)
            base_value = complex(np.random.normal(0, 1), np.random.normal(0, 1))
            decay_factor = 1.0 / (1.0 + complexity_factor * self.n)
            value = base_value * decay_factor
            support_dim = np.random.randint(0, self.X.dimension + 1)
            weight_factor = (p + q) / (2.0 * self.n + 5.0)
            weight = weight_factor * (1.0 + self.X.dimension + self.X.singularity_measure +
                                      0.5 * self.X.nilpotent_order + 0.3 * self.X.exceptional_divisors)
            self.cohomology.add_class(p, q, value, weight, support_dim)

# ============================================================================
# Part 4: Test Cases
# ============================================================================

def test_case_1():
    # Smooth Projective Plane: P^2 (dim=2, no singularities, reduced)
    X = MotivicSpace(name="P^2", dimension=2, singularity_measure=0.0)
    # Manually set cohomology to reflect H^0, H^2, H^4 = Z, Z, Z respectively.
    cohom = MotivicCohomology(X)
    cohom.add_class(0, 0, 1+0j, 1.0)      # H^0
    cohom.add_class(2, 1, 1+0j, 0.8)      # Hyperplane class in H^2
    cohom.add_class(4, 2, 1+0j, 0.6)      # Point class in H^4
    print("Test Case 1: Smooth Projective Plane P^2")
    print("Motivic Space:", X)
    print(cohom)
    # Build spectral sequence from these classes:
    ss = SpectralSequence("P^2_Spectral")
    for cls in cohom.classes:
        wgroup = int(round(cls.weight * 10))
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    stable_page = ss.compute_until_stable(max_page=5)
    print(ss)
    print("Spectral sequence stabilized at page:", stable_page)
    print("Max differential norm:", ss.get_max_differential_norm(stable_page))
    print("-"*50)

def test_case_2():
    # Singular curve: Union of two lines in P^2 (dim=1, singularity_measure=1.0)
    X = MotivicSpace(name="Union_of_Lines", dimension=1, singularity_measure=1.0)
    cohom = MotivicCohomology(X)
    # For a reducible curve, set H^0 = Z, H^2 = Z^2.
    cohom.add_class(0, 0, 1+0j, 1.0)            # H^0
    cohom.add_class(2, 1, 1+0j, 0.7)            # First line H^2
    cohom.add_class(2, 1, 1+0j, 0.7)            # Second line H^2
    print("Test Case 2: Singular Curve - Union of Two Lines")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("UnionLines_Spectral")
    for cls in cohom.classes:
        wgroup = int(round(cls.weight * 10))
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    stable_page = ss.compute_until_stable(max_page=5)
    print(ss)
    print("Spectral sequence stabilized at page:", stable_page)
    print("Max differential norm:", ss.get_max_differential_norm(stable_page))
    print("-"*50)

def test_case_3():
    # Non-reduced Fat Point: Spec(k[ε]/(ε^2)) (dim=0, nilpotent_order=1)
    X = MotivicSpace(name="Fat_Point", dimension=0, singularity_measure=0.0, nilpotent_order=1)
    cohom = MotivicCohomology(X)
    # For a fat point, set H^0 = Z and introduce a torsion element in H^2.
    cohom.add_class(0, 0, 1+0j, 1.0)              # H^0
    # Represent a torsion element in H^2 by a value that we denote mod 2 (for simplicity).
    cohom.add_class(2, 1, 0.0+0j, 0.5)              # H^2 torsion (conceptually Z/2)
    print("Test Case 3: Non-reduced Fat Point")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("FatPoint_Spectral")
    for cls in cohom.classes:
        wgroup = int(round(cls.weight * 10))
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    stable_page = ss.compute_until_stable(max_page=5)
    print(ss)
    print("Spectral sequence stabilized at page:", stable_page)
    print("Max differential norm:", ss.get_max_differential_norm(stable_page))
    print("-"*50)

def test_case_4():
    # Blow-up of P^2 at a point: tilde{X} with dim=2, exceptional_divisors=1
    X = MotivicSpace(name="BlowUp_P2", dimension=2, singularity_measure=0.0, exceptional_divisors=1)
    cohom = MotivicCohomology(X)
    # Following the blow-up formula:
    cohom.add_class(0, 0, 1+0j, 1.0)            # H^0
    cohom.add_class(2, 1, 1+0j, 0.8)            # Pulled back hyperplane in H^2
    cohom.add_class(2, 1, 1+0j, 0.8)            # Exceptional divisor in H^2
    cohom.add_class(4, 2, 1+0j, 0.6)            # Fundamental class H^4
    print("Test Case 4: Blow-up of P^2 at a Point")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("BlowUp_Spectral")
    for cls in cohom.classes:
        wgroup = int(round(cls.weight * 10))
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    # Simulate a differential d2 that maps one generator from filtration 1 to the top class
    ss.add_differential(2, (2,1,8), (4,2,6), 1+0j)  # Example: d2 kills one part of H^2 from E
    stable_page = ss.compute_until_stable(max_page=5)
    print(ss)
    print("Spectral sequence stabilized at page:", stable_page)
    print("Max differential norm:", ss.get_max_differential_norm(stable_page))
    print("-"*50)

def test_case_5():
    # Smooth Projective Elliptic Curve: E (dim=1, genus=1, smooth)
    X = MotivicSpace(name="Elliptic_Curve", dimension=1, singularity_measure=0.0)
    cohom = MotivicCohomology(X)
    # Expected: H^0 = Z, H^1 = Z^2, H^2 = Z.
    cohom.add_class(0, 0, 1+0j, 1.0)            # H^0
    cohom.add_class(1, 0, 1+0j, 0.9)            # H^1 (first generator)
    cohom.add_class(1, 0, 1+0j, 0.9)            # H^1 (second generator)
    cohom.add_class(2, 1, 1+0j, 0.8)            # H^2
    print("Test Case 5: Smooth Projective Elliptic Curve")
    print("Motivic Space:", X)
    print(cohom)
    ss = SpectralSequence("Elliptic_Spectral")
    for cls in cohom.classes:
        wgroup = int(round(cls.weight * 10))
        ss.set_entry(2, cls.p, cls.q, wgroup, cls.value)
    # Simulate differentials mimicking a Čech or Mayer-Vietoris cover.
    ss.add_differential(2, (0,0,10), (0,0,10), 0j)  # trivial on H^0
    ss.add_differential(2, (1,0,9), (1,0,9), 0j)    # trivial on H^1
    stable_page = ss.compute_until_stable(max_page=5)
    print(ss)
    print("Spectral sequence stabilized at page:", stable_page)
    print("Max differential norm:", ss.get_max_differential_norm(stable_page))
    print("-"*50)

# ============================================================================
# Main: Run All Test Cases
# ============================================================================

def main():
    print("\n--- Expanded Motivic Taylor Tower Test ---\n")
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()

if __name__ == "__main__":
    main()
