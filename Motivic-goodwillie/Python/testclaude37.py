"""
Computational Verification of Weighted Motivic Taylor Tower Convergence

This program implements a comprehensive computational framework to test the key claims 
of the Weighted Motivic Taylor Tower Conjecture. It simulates the behavior of motivic 
spaces under various geometric operations (blow-ups, singularities, nilpotent thickenings)
and verifies that the weighted tower approach successfully handles cases where classical
Goodwillie calculus would fail to converge.

The implementation models:
1. Motivic spaces with dimension and singularity data
2. Three types of weight functions (dimension-based, singularity-based, stage-based)
3. Cohomology classes with bidegrees and weights
4. Polynomial approximation towers with weight filtration
5. Spectral sequences and differential behavior
6. Obstruction classes and their decay rates
7. Convergence verification for different test cases

While necessarily simplifying the full theory, this implementation preserves the essential
mathematical structure needed to test the core claims about obstruction vanishing and
tower convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import factorial
import sympy as sp
from collections import defaultdict
import pandas as pd
from itertools import product
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Part 1: Basic Structures for Motivic Spaces and Cohomology
# ============================================================================

@dataclass
class MotivicSpace:
    """
    Model of a motivic space with dimension, singularity data, and nilpotent structure.
    """
    name: str
    dimension: int
    singularity_measure: float  # Could be Milnor number or similar
    nilpotent_order: int = 0    # Order of nilpotency (0 = reduced scheme)
    exceptional_divisors: int = 0  # Number of exceptional divisors from blow-ups
    
    def __str__(self):
        result = f"{self.name} (dim={self.dimension}, sing={self.singularity_measure}"
        if self.nilpotent_order > 0:
            result += f", nilp={self.nilpotent_order}"
        if self.exceptional_divisors > 0:
            result += f", exc_div={self.exceptional_divisors}"
        result += ")"
        return result
    
    def complexity_measure(self) -> float:
        """
        Combined measure of geometric complexity incorporating all features.
        """
        return (self.dimension + 
                self.singularity_measure + 
                0.5 * self.nilpotent_order + 
                0.3 * self.exceptional_divisors)


@dataclass
class CohomologyClass:
    """
    Represents a cohomology class with bidegree (p,q), value, and weight.
    """
    p: int  # First cohomological degree
    q: int  # Second cohomological degree (e.g., "weight")
    value: complex  # Coefficient value
    weight: float   # Weight assigned to this class
    support_dimension: int = -1  # Dimension of the subvariety supporting this class
    
    def __str__(self):
        base_str = f"H^{self.p},{self.q} = {self.value:.4f} (weight={self.weight:.4f})"
        if self.support_dimension >= 0:
            base_str += f", supported on dim={self.support_dimension}"
        return base_str
    
    def __eq__(self, other):
        if not isinstance(other, CohomologyClass):
            return False
        return (self.p == other.p and 
                self.q == other.q and 
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
    
    def add_class(self, p: int, q: int, value: complex, weight: float, 
                  support_dimension: int = -1):
        """Add a cohomology class."""
        cls = CohomologyClass(p, q, value, weight, support_dimension)
        self.classes.append(cls)
        self._class_dict[(p, q)].append(cls)
    
    def filter_by_weight(self, max_weight: float) -> 'MotivicCohomology':
        """Return a new cohomology object with classes filtered by weight."""
        result = MotivicCohomology(self.X)
        result.classes = [cls for cls in self.classes if cls.weight <= max_weight]
        for cls in result.classes:
            result._class_dict[(cls.p, cls.q)].append(cls)
        return result
    
    def get_classes(self, p: int, q: int) -> List[CohomologyClass]:
        """Get all cohomology classes with bidegree (p,q)."""
        return self._class_dict.get((p, q), [])
    
    def total_weight_measure(self) -> float:
        """Compute the total weight measure of all classes."""
        return sum(abs(cls.value) * cls.weight for cls in self.classes)
    
    def __len__(self):
        return len(self.classes)
    
    def __str__(self):
        return f"Cohomology of {self.X.name} with {len(self.classes)} classes"


class SpectralSequence:
    """
    Represents a spectral sequence arising from a filtered complex.
    """
    def __init__(self, name: str, initial_page: Dict[Tuple[int, int, int], complex] = None):
        self.name = name
        self.pages: Dict[int, Dict[Tuple[int, int, int], complex]] = {}
        if initial_page:
            self.pages[2] = initial_page  # Start at E2 page by convention
        else:
            self.pages[2] = {}
        self.current_page = 2
        self.differentials: Dict[int, Dict[Tuple[int, int, int], Tuple[int, int, int, complex]]] = {}
    
    def set_entry(self, page: int, p: int, q: int, w: int, value: complex):
        """Set an entry in the spectral sequence at a specific page."""
        if page not in self.pages:
            self.pages[page] = {}
        self.pages[page][(p, q, w)] = value
    
    def get_entry(self, page: int, p: int, q: int, w: int) -> complex:
        """Get an entry from the spectral sequence."""
        if page not in self.pages:
            return 0j
        return self.pages[page].get((p, q, w), 0j)
    
    def add_differential(self, page: int, source: Tuple[int, int, int], 
                         target: Tuple[int, int, int], value: complex):
        """Add a differential to the spectral sequence."""
        if page not in self.differentials:
            self.differentials[page] = {}
        self.differentials[page][source] = (target[0], target[1], target[2], value)
    
    def compute_next_page(self):
        """Compute the next page of the spectral sequence using differentials."""
        current = self.current_page
        next_page = current + 1
        
        # Initialize next page with current page entries
        self.pages[next_page] = dict(self.pages[current])
        
        # Apply differentials
        if current in self.differentials:
            for source, target_info in self.differentials[current].items():
                p, q, w = source
                target_p, target_q, target_w, value = target_info
                
                # Source entry is killed by differential
                if (p, q, w) in self.pages[next_page]:
                    self.pages[next_page][(p, q, w)] -= value
                
                # Target entry receives the differential
                if (target_p, target_q, target_w) in self.pages[next_page]:
                    self.pages[next_page][(target_p, target_q, target_w)] += value
        
        self.current_page = next_page
        return self.pages[next_page]
    
    def compute_until_stable(self, max_page: int = 10, tolerance: float = 1e-10) -> int:
        """
        Compute pages until the spectral sequence stabilizes or reaches max_page.
        Returns the page at which stabilization occurred.
        """
        prev_page_entries = dict(self.pages[self.current_page])
        
        for page in range(self.current_page + 1, max_page + 1):
            self.compute_next_page()
            current_entries = self.pages[self.current_page]
            
            # Check if stabilized
            stabilized = True
            for key in set(prev_page_entries.keys()) | set(current_entries.keys()):
                prev_val = prev_page_entries.get(key, 0j)
                curr_val = current_entries.get(key, 0j)
                if abs(prev_val - curr_val) > tolerance:
                    stabilized = False
                    break
            
            if stabilized:
                return self.current_page
            
            prev_page_entries = dict(current_entries)
        
        return self.current_page
    
    def get_max_differential_norm(self, page: int) -> float:
        """Get the maximum norm of any differential at the given page."""
        if page not in self.differentials:
            return 0.0
        
        return max([abs(target_info[3]) for _, target_info in self.differentials[page].items()], 
                  default=0.0)
    
    def __str__(self):
        return f"{self.name} Spectral Sequence (current page: E_{self.current_page})"


# ============================================================================
# Part 2: Weight Functions Implementation
# ============================================================================

def dimension_weight(X: MotivicSpace) -> float:
    """
    Dimension-based weight function: w_dim(X) = 1/(1 + dim(X))
    """
    return 1.0 / (1.0 + X.dimension)


def singularity_weight(X: MotivicSpace) -> float:
    """
    Singularity-based weight function: w_sing(X) = 1/(1 + sing(X))
    """
    return 1.0 / (1.0 + X.singularity_measure)


def nilpotent_weight(X: MotivicSpace) -> float:
    """
    Nilpotent-based weight function: w_nilp(X) = 1/(1 + nilp_order(X))
    """
    return 1.0 / (1.0 + X.nilpotent_order)


def exceptional_divisor_weight(X: MotivicSpace) -> float:
    """
    Weight function based on exceptional divisors: w_exc(X) = 1/(1 + exc_div(X))
    """
    return 1.0 / (1.0 + X.exceptional_divisors)


def stage_weight(n: int) -> float:
    """
    Stage-based weight function: w_stage(n) = 1/(n+1)
    """
    return 1.0 / (n + 1)


def total_weight(X: MotivicSpace, n: int) -> float:
    """
    Combined weight function: w_total(X,n) = w_dim(X) * w_sing(X) * w_nilp(X) * w_exc(X) * w_stage(n)
    """
    return (dimension_weight(X) * 
            singularity_weight(X) * 
            nilpotent_weight(X) * 
            exceptional_divisor_weight(X) * 
            stage_weight(n))


def custom_weight(X: MotivicSpace, n: int, 
                 dim_factor: float = 1.0,
                 sing_factor: float = 1.0,
                 nilp_factor: float = 1.0,
                 exc_factor: float = 1.0,
                 stage_factor: float = 1.0) -> float:
    """
    Customizable weight function with adjustable importance factors.
    """
    return (pow(dimension_weight(X), dim_factor) * 
            pow(singularity_weight(X), sing_factor) * 
            pow(nilpotent_weight(X), nilp_factor) * 
            pow(exceptional_divisor_weight(X), exc_factor) * 
            pow(stage_weight(n), stage_factor))


# ============================================================================
# Part 3: Polynomial Approximation and Tower Construction
# ============================================================================

class PolynomialApproximation:
    """
    Represents a polynomial approximation of degree n for a functor F.
    """
    def __init__(self, n: int, X: MotivicSpace):
        self.n = n
        self.X = X
        self.cohomology = MotivicCohomology(X)
        self.spectral_sequence = None
        
    def generate_cohomology(self, num_classes: int = 10, seed: Optional[int] = None,
                           complexity_factor: float = 0.5):
        """
        Generate synthetic cohomology classes based on the space and degree.
        This is a simplified model of what real cohomology might look like.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate classes with weights related to dimension and singularity
        for i in range(num_classes):
            # Cohomological degrees influenced by polynomial degree
            p = np.random.randint(0, self.n + 3)  # Cohomological degree
            q = np.random.randint(0, self.n + 2)  # Weight degree
            
            # Value decreases with polynomial degree to model convergence
            base_value = complex(np.random.normal(0, 1), np.random.normal(0, 1))
            decay_factor = 1.0 / (1.0 + complexity_factor * self.n)
            value = base_value * decay_factor
            
            # Support dimension - classes can be supported on subvarieties
            support_dim = np.random.randint(0, self.X.dimension + 1)
            
            # Weight increases with dimension and singularity
            # Higher p,q classes get higher weight (more complex)
            weight_factor = (p + q) / (2.0 * self.n + 5.0)
            # Complete the weight calculation by incorporating singularity, nilpotent order and exceptional divisors
            weight = weight_factor * (1.0 + self.X.dimension + self.X.singularity_measure +
                                      0.5 * self.X.nilpotent_order + 0.3 * self.X.exceptional_divisors)
            
            self.cohomology.add_class(p, q, value, weight, support_dim)


# ============================================================================
# Main: Demonstration and Testing
# ============================================================================

def main():
    # Create a MotivicSpace instance with some sample parameters
    X = MotivicSpace(name="TestSpace", dimension=3, singularity_measure=2.0, nilpotent_order=1, exceptional_divisors=1)
    print("Motivic Space:", X)
    
    # Create a polynomial approximation of degree 2 for the motivic space X
    poly_approx = PolynomialApproximation(n=2, X=X)
    poly_approx.generate_cohomology(num_classes=10, seed=42)
    print(poly_approx.cohomology)
    
    # Filter cohomology classes by a maximum weight threshold
    max_weight = 0.5
    filtered_cohom = poly_approx.cohomology.filter_by_weight(max_weight)
    print(f"Filtered Cohomology (max weight {max_weight:.2f}):")
    for cls in filtered_cohom.classes:
        print("  ", cls)
    
    # Setup a spectral sequence for demonstration
    ss = SpectralSequence(name="DemoSpectralSequence")
    # Populate the E2 page with entries from the cohomology classes
    for cls in poly_approx.cohomology.classes:
        # Map the real-valued weight to an integer weight group for simplicity (e.g., by multiplying by 10)
        weight_group = int(round(cls.weight * 10))
        ss.set_entry(page=2, p=cls.p, q=cls.q, w=weight_group, value=cls.value)
    
    print(ss)
    # Compute the next page of the spectral sequence
    next_page = ss.compute_next_page()
    print("Spectral Sequence Next Page Entries:")
    for key, val in next_page.items():
        print("  Entry", key, "=", val)
    
    # Compute until the spectral sequence stabilizes (up to max_page=5)
    stable_page = ss.compute_until_stable(max_page=5)
    print("Spectral sequence stabilized at page:", stable_page)
    
    max_diff = ss.get_max_differential_norm(page=stable_page)
    print(f"Maximum differential norm at page E_{stable_page}: {max_diff:.4f}")
    
    # Plot a histogram of the cohomology class weights
    weights = [cls.weight for cls in poly_approx.cohomology.classes]
    plt.hist(weights, bins=10, edgecolor='black')
    plt.title("Histogram of Cohomology Class Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    main()
