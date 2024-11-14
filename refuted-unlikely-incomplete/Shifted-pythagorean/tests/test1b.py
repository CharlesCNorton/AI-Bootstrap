from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict, Optional
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from scipy.stats import linregress

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
        if not points:
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

        growth_rate = linregress(z_values, cumulative_solutions)

        return {
            'size_distribution': dict(distribution),
            'size_progression': sizes,
            'growth_rate': growth_rate.slope,
            'r_squared': growth_rate.rvalue**2
        }

    def analyze_max_family_structure(self) -> Dict:
        """Analyze the structure of maximum-size families"""
        results = {}
        for z, structure in self.max_families.items():
            x_diffs = np.diff(structure['x_values'])
            y_diffs = np.diff(structure['y_values'])
            ratio_diffs = np.diff(structure['ratios'])

            results[z] = {
                'x_progression': list(x_diffs),
                'y_progression': list(y_diffs),
                'ratio_progression': list(ratio_diffs),
                'symmetry_score': self._calculate_symmetry(structure['normalized_points']),
                'ratio_range': (min(structure['ratios']), max(structure['ratios']))
            }
        return results

    def predict_large_families(self) -> Dict:
        """Identify patterns that predict large family sizes"""
        z_features = {}
        for z, family in self.z_families.items():
            if family.size > 1:
                structure = family.get_structure()
                z_features[z] = {
                    'size': family.size,
                    'ratio_range': max(structure['ratios']) - min(structure['ratios']),
                    'symmetry': self._calculate_symmetry(structure['normalized_points']),
                    'density': family.size / (max(structure['x_values']) - min(structure['x_values']))
                }
        return z_features

def print_comprehensive_analysis(analyzer: ComprehensiveAnalyzer):
    print("\n=== COMPREHENSIVE ANALYSIS ===\n")

    print("Maximum Family Analysis:")
    max_structure = analyzer.analyze_max_family_structure()
    for z, data in max_structure.items():
        print(f"\nZ = {z}:")
        print(f"Symmetry Score: {data['symmetry_score']:.3f}")
        print(f"Ratio Range: {data['ratio_range']}")
        print(f"X-progression (first 5): {data['x_progression'][:5]}")
        print(f"Y-progression (first 5): {data['y_progression'][:5]}")

    print("\nSymmetry Analysis:")
    most_symmetric = sorted(analyzer.symmetry_data.items(), key=lambda x: x[1])[:3]
    print("Most symmetric families:")
    for z, score in most_symmetric:
        print(f"Z = {z}: {score:.3f}")

    print("\nGrowth Analysis:")
    growth = analyzer.growth_data
    print(f"Family size distribution: {growth['size_distribution']}")
    print(f"Growth rate: {growth['growth_rate']:.3f} (R² = {growth['r_squared']:.3f})")

    print("\nPredictive Features:")
    features = analyzer.predict_large_families()
    large_z = sorted([(z, f['size']) for z, f in features.items()],
                    key=lambda x: x[1], reverse=True)[:5]
    print("Largest families and their features:")
    for z, size in large_z:
        f = features[z]
        print(f"Z = {z} (size {size}):")
        print(f"  Ratio Range: {f['ratio_range']:.3f}")
        print(f"  Symmetry: {f['symmetry']:.3f}")
        print(f"  Density: {f['density']:.3f}")

if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer(10000)
    print_comprehensive_analysis(analyzer)
