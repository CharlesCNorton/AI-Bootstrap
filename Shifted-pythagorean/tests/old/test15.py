from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple, FrozenSet, Generator
from math import gcd, sqrt, isqrt, log, ceil, pi, e
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, product, chain
from enum import Enum, auto
from scipy.stats import entropy, gaussian_kde
from scipy.ndimage import maximum_filter
from scipy.stats import linregress
from scipy.signal import find_peaks_cwt
import networkx as nx
from fractions import Fraction
from sympy import Matrix, solve, symbols, Rational, S

@dataclass(frozen=True)
class Solution:
    x: int
    y: int
    z: int

    def __post_init__(self):
        assert self.x * self.x + self.y * self.y == self.z * self.z + 1

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def normalized_ratios(self) -> Tuple[float, float, float]:
        return tuple(round(r, 4) for r in self.ratios())

    def is_primitive(self) -> bool:
        return gcd(gcd(self.x, self.y), self.z) == 1

class StructureType(Enum):
    FUNDAMENTAL = auto()
    DERIVED = auto()
    COMPOSITE = auto()
    SYMMETRIC = auto()
    ASYMMETRIC = auto()

class RationalStructure:
    def __init__(self, num: Tuple[int, ...], den: Tuple[int, ...]):
        self.numerators = num
        self.denominators = den
        self.ratios = tuple(Rational(n, d) for n, d in zip(num, den))
        self._compute_properties()

    def _compute_properties(self):
        self.complexity = sum(n.bit_length() + d.bit_length()
                            for n, d in zip(self.numerators, self.denominators))
        self.symmetry = len(set(self.ratios)) < len(self.ratios)
        self.structure_type = self._determine_type()
        self.normalized_ratios = self._normalize_ratios()

    def _determine_type(self) -> StructureType:
        if all(d == 1 for d in self.denominators):
            return StructureType.FUNDAMENTAL
        elif self.symmetry:
            return StructureType.SYMMETRIC
        elif max(self.denominators) <= 4:
            return StructureType.DERIVED
        else:
            return StructureType.COMPOSITE

    def _normalize_ratios(self) -> Tuple[float, ...]:
        min_ratio = min(float(r) for r in self.ratios)
        return tuple(float(r)/min_ratio for r in self.ratios)

    def transform(self, other: 'RationalStructure') -> 'RationalStructure':
        new_nums = tuple(n1 * n2 for n1, n2 in zip(self.numerators, other.numerators))
        new_dens = tuple(d1 * d2 for d1, d2 in zip(self.denominators, other.denominators))
        return RationalStructure(new_nums, new_dens)

    def similarity_to(self, other: 'RationalStructure') -> float:
        return 1 - max(abs(n1 - n2) for n1, n2 in zip(self.normalized_ratios, other.normalized_ratios))

class RationalBasis:
    def __init__(self):
        self.primary = RationalStructure((3, 4, 3), (4, 5, 5))
        self.secondary = RationalStructure((4, 3, 4), (3, 5, 5))
        self.tertiary = RationalStructure((5, 11, 5), (12, 12, 13))
        self.quaternary = RationalStructure((7, 24, 7), (24, 25, 25))
        self.quinary = RationalStructure((8, 15, 8), (15, 17, 17))
        self.fundamental_transforms = self._compute_fundamental_transforms()
        self.base_patterns = [self.primary, self.secondary, self.tertiary,
                            self.quaternary, self.quinary]

    def _compute_fundamental_transforms(self) -> List[RationalStructure]:
        transforms = []
        for p in product(range(1, 9), repeat=3):
            for q in product(range(1, 9), repeat=3):
                transform = RationalStructure(p, q)
                if self._is_valid_transform(transform):
                    transforms.append(transform)
        return transforms

    def _is_valid_transform(self, transform: RationalStructure) -> bool:
        result = self.primary.transform(transform)
        return (max(result.denominators) <= 24 and
                min(result.numerators) >= 1 and
                self._preserves_structure(result))

    def _preserves_structure(self, structure: RationalStructure) -> bool:
        ratios = structure.ratios
        return (ratios[0] * ratios[2] == ratios[1] and
                all(0.1 <= float(r) <= 10 for r in ratios))

class StructuralAnalyzer:
    def __init__(self):
        self.basis = RationalBasis()
        self.structure_graph = nx.DiGraph()
        self.pattern_families: Dict[int, List[RationalStructure]] = defaultdict(list)

    def analyze_solution(self, solution: Solution) -> Dict:
        ratios = self._to_rational_structure(solution)
        family = self._identify_pattern_family(ratios)
        transforms = self._find_generating_transforms(ratios)
        predictions = self._predict_related_solutions(ratios)

        return {
            'family': family,
            'transforms': transforms,
            'predictions': predictions,
            'complexity': ratios.complexity
        }

    def _to_rational_structure(self, solution: Solution) -> RationalStructure:
        ratios = solution.ratios()
        nums, dens = [], []
        for r in ratios:
            frac = Fraction(r).limit_denominator(24)
            nums.append(frac.numerator)
            dens.append(frac.denominator)
        return RationalStructure(tuple(nums), tuple(dens))

    def _identify_pattern_family(self, structure: RationalStructure) -> Optional[int]:
        # Check existing families
        best_similarity = 0
        best_family = None

        for family_id, members in self.pattern_families.items():
            for member in members:
                similarity = structure.similarity_to(member)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_family = family_id

        if best_similarity > 0.75:  # High similarity threshold
            return best_family

        # Check base patterns
        for base in self.basis.base_patterns:
            if structure.similarity_to(base) > 0.5:  # Lower threshold for base patterns
                new_family = len(self.pattern_families)
                self.pattern_families[new_family].append(structure)
                return new_family

        # Create new family
        new_family = len(self.pattern_families)
        self.pattern_families[new_family].append(structure)
        return new_family

    def _find_generating_transforms(self, structure: RationalStructure) -> List[RationalStructure]:
        transforms = []
        for base in self.basis.base_patterns:
            if structure.similarity_to(base) > 0.25:
                transform = self._find_transform(base, structure)
                if transform:
                    transforms.append(transform)
        return transforms

    def _find_transform(self, base: RationalStructure, target: RationalStructure) -> Optional[RationalStructure]:
        x, y, z = symbols('x y z')
        equations = []
        for b, t in zip(base.ratios, target.ratios):
            equations.append(b * x - t * y)

        try:
            solution = solve(equations, [x, y, z])
            if solution:
                nums = tuple(int(sol.numerator) if isinstance(sol, Rational)
                           else int(sol) for sol in solution.values())
                dens = tuple(int(sol.denominator) if isinstance(sol, Rational)
                           else 1 for sol in solution.values())
                return RationalStructure(nums, dens)
        except:
            return None

    def _predict_related_solutions(self, structure: RationalStructure) -> List[RationalStructure]:
        predictions = []
        for transform in self.basis.fundamental_transforms:
            prediction = structure.transform(transform)
            if self._is_valid_prediction(prediction):
                predictions.append(prediction)
        return predictions

    def _is_valid_prediction(self, structure: RationalStructure) -> bool:
        return (max(structure.denominators) <= 24 and
                min(structure.numerators) >= 1 and
                self.basis._preserves_structure(structure))

class SolutionGenerator:
    def __init__(self, analyzer: StructuralAnalyzer):
        self.analyzer = analyzer
        self.known_structures: Set[RationalStructure] = set()
        self.predicted_structures: List[RationalStructure] = []

    def generate_solutions(self, max_denominator: int = 24) -> Generator[Solution, None, None]:
        self._initialize_structures(max_denominator)

        while self.predicted_structures:
            structure = self.predicted_structures.pop(0)
            if structure not in self.known_structures:
                self.known_structures.add(structure)
                solution = self._structure_to_solution(structure)
                if solution:
                    yield solution
                    self._update_predictions(structure)

    def _initialize_structures(self, max_denominator: int):
        self.predicted_structures = self.analyzer.basis.base_patterns.copy()
        self.known_structures = set()

    def _structure_to_solution(self, structure: RationalStructure) -> Optional[Solution]:
        ratios = [float(r) for r in structure.ratios]
        for x in range(1, 1000):
            y = int(round(x / ratios[0]))
            if y <= 0:
                continue
            z = int(round(y / ratios[1]))
            if z <= 0:
                continue
            if abs(x*x + y*y - (z*z + 1)) <= 1e-10:
                return Solution(x, y, z)
        return None

    def _update_predictions(self, structure: RationalStructure):
        new_predictions = self.analyzer._predict_related_solutions(structure)
        self.predicted_structures.extend(new_predictions)
        self.predicted_structures.sort(key=lambda s: s.complexity)

def main():
    structural_analyzer = StructuralAnalyzer()
    generator = SolutionGenerator(structural_analyzer)

    print("Generating and analyzing solutions...")

    solutions = []
    for solution in generator.generate_solutions():
        solutions.append(solution)
        if len(solutions) >= 1000:
            break

    print(f"\nGenerated {len(solutions)} solutions")

    pattern_counts = defaultdict(int)
    complexity_stats = []

    for solution in solutions:
        analysis = structural_analyzer.analyze_solution(solution)
        pattern_counts[analysis['family']] += 1
        complexity_stats.append(analysis['complexity'])

    print("\nPattern Family Distribution:")
    for family, count in sorted(pattern_counts.items()):
        print(f"Family {family}: {count} solutions")

    print("\nComplexity Statistics:")
    print(f"Mean complexity: {np.mean(complexity_stats):.2f}")
    print(f"Std complexity: {np.std(complexity_stats):.2f}")

    print("\nFundamental Rational Structures:")
    for transform in structural_analyzer.basis.fundamental_transforms[:5]:
        print(f"Transform: {transform.numerators}/{transform.denominators}")

    print("\nPredicted Novel Structures:")
    novel_count = 0
    for solution in solutions[-10:]:
        analysis = structural_analyzer.analyze_solution(solution)
        if len(analysis['predictions']) > 0:
            print(f"Solution {solution.x},{solution.y},{solution.z} -> "
                  f"{len(analysis['predictions'])} new predictions")
            novel_count += 1
    print(f"\nTotal solutions with novel predictions: {novel_count}")

if __name__ == "__main__":
    main()
