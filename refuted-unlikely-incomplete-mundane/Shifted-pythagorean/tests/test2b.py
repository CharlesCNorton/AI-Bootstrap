from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict, Optional
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    x: int
    y: int
    z: int

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def verify(self) -> bool:
        return self.x*self.x + self.y*self.y == self.z*self.z + 1

class ZFamily:
    def __init__(self, z: int):
        self.z = z
        self.solutions: Set[Solution] = set()
        self.ratio_patterns: Dict[Tuple[float, float, float], int] = defaultdict(int)

    def add_solution(self, solution: Solution):
        if solution.z != self.z:
            raise ValueError(f"Solution z={solution.z} doesn't match family z={self.z}")
        self.solutions.add(solution)
        ratios = tuple(round(r, 2) for r in solution.ratios())
        self.ratio_patterns[ratios] += 1

    @property
    def size(self) -> int:
        return len(self.solutions)

    def get_structure(self) -> Dict:
        solutions = sorted(self.solutions, key=lambda s: (s.x, s.y))
        return {
            'size': self.size,
            'x_values': [s.x for s in solutions],
            'y_values': [s.y for s in solutions],
            'ratios': [s.x/s.y for s in solutions],
            'normalized_points': [(s.x/self.z, s.y/self.z) for s in solutions]
        }

class ComprehensiveAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions = self._generate_solutions()
        self.z_families = self._organize_families()
        self.max_families = self._find_max_families()
        self.symmetry_data = self._analyze_symmetry()
        self.growth_data = self._analyze_growth()

    def _generate_solutions(self) -> Set[Solution]:
        solutions = set()
        for z in range(2, self.limit):
            for x in range(2, z):
                y_squared = z*z + 1 - x*x
                if y_squared > 0:
                    y = isqrt(y_squared)
                    if (y*y == y_squared and y <= self.limit and
                        y > 1 and x != y and y != z):
                        sol = Solution(min(x, y), max(x, y), z)
                        if sol.verify():
                            solutions.add(sol)
        return solutions

    def _organize_families(self) -> Dict[int, ZFamily]:
        families = defaultdict(lambda: ZFamily(0))
        for sol in self.solutions:
            if sol.z not in families:
                families[sol.z] = ZFamily(sol.z)
            families[sol.z].add_solution(sol)
        return families

    def _find_max_families(self) -> Dict[int, Dict]:
        max_size = max(f.size for f in self.z_families.values())
        return {z: family.get_structure()
                for z, family in self.z_families.items()
                if family.size == max_size}

    def _calculate_symmetry(self, points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        center_x = sum(x for x,y in points) / len(points)
        center_y = sum(y for x,y in points) / len(points)
        distances = [sqrt((x-center_x)**2 + (y-center_y)**2) for x,y in points]
        return np.std(distances) / np.mean(distances) if distances else 0.0

    def _analyze_symmetry(self) -> Dict[int, float]:
        return {z: self._calculate_symmetry(family.get_structure()['normalized_points'])
                for z, family in self.z_families.items()
                if family.size > 5}

    def _analyze_growth(self) -> Dict:
        sizes = sorted(set(f.size for f in self.z_families.values()))
        distribution = Counter(f.size for f in self.z_families.values())

        z_values = sorted(self.z_families.keys())
        cumulative_solutions = np.cumsum([len(self.z_families[z].solutions)
                                        for z in z_values])

        slope, intercept = np.polyfit(z_values, cumulative_solutions, 1)
        r_squared = np.corrcoef(z_values, cumulative_solutions)[0,1]**2

        return {
            'size_distribution': dict(distribution),
            'size_progression': sizes,
            'growth_rate': slope,
            'r_squared': r_squared
        }

class MaxSizeAnalyzer:
    def __init__(self, solutions: List[Solution]):
        self.solutions = solutions
        self.x_values = sorted([s.x for s in solutions])
        self.y_values = sorted([s.y for s in solutions])
        self.z = solutions[0].z if solutions else 0

    def analyze_constraints(self) -> Dict:
        max_possible = self._theoretical_max_solutions()
        primitive_sols = self._count_primitive_solutions()
        perfect_squares = self._count_perfect_square_opportunities()

        return {
            'theoretical_bound': max_possible,
            'primitive_count': primitive_sols,
            'derived_count': len(self.solutions) - primitive_sols,
            'perfect_square_count': perfect_squares,
            'limiting_factor': self._identify_limiting_factor()
        }

    def _theoretical_max_solutions(self) -> int:
        if not self.z:
            return 0
        return isqrt(self.z * 2)

    def _count_primitive_solutions(self) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        return sum(1 for s in self.solutions
                  if gcd(s.x, gcd(s.y, s.z)) == 1)

    def _count_perfect_square_opportunities(self) -> int:
        count = 0
        z_squared_plus_1 = self.z * self.z + 1
        for x in self.x_values:
            y_squared = z_squared_plus_1 - x*x
            if y_squared > 0 and isqrt(y_squared)**2 == y_squared:
                count += 1
        return count

    def _identify_limiting_factor(self) -> str:
        max_theoretical = self._theoretical_max_solutions()
        actual = len(self.solutions)
        perfect_squares = self._count_perfect_square_opportunities()

        if actual == perfect_squares:
            return "Perfect Square Constraint"
        elif actual == max_theoretical:
            return "Theoretical Bound"
        return "Unknown Factor"

class SymmetryTradeoffAnalyzer:
    def __init__(self, z_families: Dict[int, ZFamily]):
        self.families = z_families
        self.size_symmetry_pairs = self._compute_pairs()

    def _compute_pairs(self) -> List[Tuple[int, float]]:
        pairs = []
        for family in self.families.values():
            if family.size >= 2:
                structure = family.get_structure()
                symmetry = self._calculate_symmetry(structure['normalized_points'])
                pairs.append((family.size, symmetry))
        return pairs

    def _calculate_symmetry(self, points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        center_x = sum(x for x,y in points) / len(points)
        center_y = sum(y for x,y in points) / len(points)
        distances = [sqrt((x-center_x)**2 + (y-center_y)**2) for x,y in points]
        return np.std(distances) / np.mean(distances) if distances else 0.0

    def compute_tradeoff_curve(self) -> Dict:
        if not self.size_symmetry_pairs:
            return {
                'correlation': 0.0,
                'pareto_optimal': [],
                'size_range': (0, 0),
                'symmetry_range': (0.0, 0.0)
            }

        sizes = [p[0] for p in self.size_symmetry_pairs]
        symmetries = [p[1] for p in self.size_symmetry_pairs]

        correlation = np.corrcoef(sizes, symmetries)[0,1]

        pareto_optimal = []
        for size, sym in self.size_symmetry_pairs:
            is_pareto = all(not(other_size >= size and other_sym <= sym)
                           for other_size, other_sym in self.size_symmetry_pairs
                           if (other_size, other_sym) != (size, sym))
            if is_pareto:
                pareto_optimal.append((size, sym))

        pareto_optimal.sort(key=lambda x: (-x[0], x[1]))

        return {
            'correlation': correlation,
            'pareto_optimal': pareto_optimal,
            'size_range': (min(sizes), max(sizes)),
            'symmetry_range': (min(symmetries), max(symmetries))
        }

def analyze_and_print_results(limit: int = 10000):
    """Run complete analysis and print results"""
    print(f"\nAnalyzing solutions up to {limit}...")

    analyzer = ComprehensiveAnalyzer(limit)

    print("\n=== COMPREHENSIVE ANALYSIS ===\n")

    print("Maximum Family Analysis:")
    for z, structure in analyzer.max_families.items():
        print(f"\nZ = {z}:")
        x_diffs = np.diff(structure['x_values'])
        y_diffs = np.diff(structure['y_values'])
        ratios = structure['ratios']
        symmetry = analyzer._calculate_symmetry(structure['normalized_points'])

        print(f"Symmetry Score: {symmetry:.3f}")
        print(f"Ratio Range: ({min(ratios)}, {max(ratios)})")
        print(f"X-progression (first 5): {list(x_diffs[:5])}")
        print(f"Y-progression (first 5): {list(y_diffs[:5])}")

    print("\nSymmetry Analysis:")
    most_symmetric = sorted(analyzer.symmetry_data.items(), key=lambda x: x[1])[:3]
    print("Most symmetric families:")
    for z, score in most_symmetric:
        print(f"Z = {z}: {score:.3f}")

    print("\nGrowth Analysis:")
    growth = analyzer.growth_data
    print(f"Family size distribution: {growth['size_distribution']}")
    print(f"Growth rate: {growth['growth_rate']:.3f} (R² = {growth['r_squared']:.3f})")

    print("\n=== ADVANCED THEORETICAL ANALYSIS ===\n")

    max_family = max(analyzer.z_families.values(), key=lambda f: len(f.solutions))
    max_analyzer = MaxSizeAnalyzer(list(max_family.solutions))
    constraints = max_analyzer.analyze_constraints()

    print("23-Solution Limit Analysis:")
    print(f"Theoretical bound: {constraints['theoretical_bound']}")
    print(f"Primitive/Derived ratio: {constraints['primitive_count']}/{constraints['derived_count']}")
    print(f"Perfect square opportunities: {constraints['perfect_square_count']}")
    print(f"Limiting factor: {constraints['limiting_factor']}")

    tradeoff = SymmetryTradeoffAnalyzer(analyzer.z_families)
    tradeoff_data = tradeoff.compute_tradeoff_curve()

    print("\nSymmetry-Size Trade-off:")
    print(f"Correlation: {tradeoff_data['correlation']:.3f}")
    print("Size range:", tradeoff_data['size_range'])
    print("Symmetry range:", tradeoff_data['symmetry_range'])
    print("\nPareto optimal points (size, symmetry):")
    for size, symmetry in tradeoff_data['pareto_optimal'][:5]:
        print(f"  {size}: {symmetry:.3f}")

if __name__ == "__main__":
    analyze_and_print_results(10000)
