from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple, FrozenSet
from math import gcd, sqrt, isqrt, log, ceil, pi, e
from collections import defaultdict
import time
import numpy as np
from itertools import combinations
from enum import Enum
from scipy.stats import entropy, gaussian_kde
from scipy.ndimage import maximum_filter
from scipy.stats import linregress
import networkx as nx
from fractions import Fraction

class AttractorType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    DERIVED = "derived"

@dataclass(frozen=True)
class Solution:
    x: int
    y: int
    z: int

    def __post_init__(self):
        assert self.x*self.x + self.y*self.y == self.z*self.z + 1

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def normalized_ratios(self) -> Tuple[float, float, float]:
        return tuple(round(r, 4) for r in self.ratios())

    def is_trivial(self) -> bool:
        return self.x == 1 or self.y == 1

    def is_primitive(self) -> bool:
        return gcd(gcd(self.x, self.y), self.z) == 1

@dataclass(frozen=True)
class Attractor:
    center: Tuple[float, float, float]
    solutions: FrozenSet[Solution]
    attractor_type: AttractorType
    basin_size: int = field(init=False)
    stability: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'basin_size', len(self.solutions))
        object.__setattr__(self, 'stability', self._calculate_stability())

    def _calculate_stability(self) -> float:
        if not self.solutions:
            return 0.0
        ratios = np.array([s.ratios() for s in self.solutions])
        return 1 - np.std(ratios, axis=0).mean()

    def __hash__(self):
        return hash(self.center)

class MultiscaleAttractorAnalyzer:
    PRIMARY_RATIOS = (0.75, 0.8, 0.6)
    SECONDARY_RATIOS = (1.3333, 0.6, 0.8)
    TERTIARY_RATIOS = (0.4167, 0.9231, 0.3846)

    def __init__(self):
        self.attractors: Dict[Tuple[float, float, float], Attractor] = {}
        self.transformation_graph = nx.Graph()
        self.attractor_hierarchy = {t: [] for t in AttractorType}
        self.sub_attractors: Dict[Tuple[float, float, float], List[Attractor]] = {}
        self.fractal_dimensions: Dict[str, float] = {}
        self.rational_relations: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []

    def analyze_solutions(self, solutions: List[Solution]) -> Dict:
        self._identify_attractors(solutions)
        self._analyze_transformations()
        self._build_hierarchy()
        self._find_sub_attractors(solutions)
        self._analyze_fractal_structure()
        self._find_rational_relations()
        return self._compile_analysis()

    def _identify_attractors(self, solutions: List[Solution]) -> None:
        ratio_groups = defaultdict(set)
        for sol in solutions:
            closest_center = self._find_closest_center(sol)
            if closest_center:
                ratio_groups[closest_center].add(sol)

        for center, sols in ratio_groups.items():
            if len(sols) >= 10:
                attractor_type = self._determine_attractor_type(center)
                attractor = Attractor(center, frozenset(sols), attractor_type)
                self.attractors[center] = attractor
                self.attractor_hierarchy[attractor_type].append(attractor)

    def _find_closest_center(self, solution: Solution) -> Optional[Tuple[float, float, float]]:
        ratios = solution.normalized_ratios()
        for center in [self.PRIMARY_RATIOS, self.SECONDARY_RATIOS, self.TERTIARY_RATIOS]:
            if self._is_near_center(ratios, center):
                return center
        return None

    def _is_near_center(self, ratios: Tuple[float, float, float],
                       center: Tuple[float, float, float], threshold: float = 0.01) -> bool:
        return sum(abs(r1 - r2) for r1, r2 in zip(ratios, center)) < threshold

    def _determine_attractor_type(self, center: Tuple[float, float, float]) -> AttractorType:
        if center == self.PRIMARY_RATIOS:
            return AttractorType.PRIMARY
        elif center == self.SECONDARY_RATIOS:
            return AttractorType.SECONDARY
        elif center == self.TERTIARY_RATIOS:
            return AttractorType.TERTIARY
        return AttractorType.DERIVED

    def _analyze_transformations(self) -> None:
        for a1, a2 in combinations(self.attractors.values(), 2):
            transform = self._find_transformation(a1.center, a2.center)
            if transform:
                self.transformation_graph.add_edge(a1, a2, transform=transform)

    def _find_transformation(self, center1: Tuple[float, float, float],
                           center2: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        transform = tuple(c2/c1 if abs(c1) > 1e-10 else 0
                        for c1, c2 in zip(center1, center2))
        if all(abs(t - round(t, 4)) < 1e-4 for t in transform):
            return transform
        return None

    def _build_hierarchy(self) -> None:
        for attractor in self.attractors.values():
            parents = []
            for potential_parent in self.attractors.values():
                if (potential_parent.attractor_type in
                    [AttractorType.PRIMARY, AttractorType.SECONDARY] and
                    self._is_derived_from(attractor, potential_parent)):
                    parents.append(potential_parent)

            if parents and attractor.attractor_type == AttractorType.DERIVED:
                self._update_attractor_hierarchy(attractor, parents)

    def _is_derived_from(self, attractor: Attractor, parent: Attractor) -> bool:
        transforms = [(4/3, 3/4, 4/3), (5/12, 11/12, 5/13)]
        for transform in transforms:
            predicted = tuple(p*t for p, t in zip(parent.center, transform))
            if self._is_near_center(attractor.center, predicted):
                return True
        return False

    def _update_attractor_hierarchy(self, attractor: Attractor, parents: List[Attractor]) -> None:
        if len(parents) == 1 and parents[0].attractor_type == AttractorType.PRIMARY:
            new_attractor = Attractor(attractor.center, attractor.solutions, AttractorType.SECONDARY)
            self.attractors[attractor.center] = new_attractor
            self.attractor_hierarchy[AttractorType.SECONDARY].append(new_attractor)
        elif len(parents) >= 2:
            new_attractor = Attractor(attractor.center, attractor.solutions, AttractorType.TERTIARY)
            self.attractors[attractor.center] = new_attractor
            self.attractor_hierarchy[AttractorType.TERTIARY].append(new_attractor)

    def _find_sub_attractors(self, solutions: List[Solution]) -> None:
        main_basins = defaultdict(list)
        for sol in solutions:
            center = self._find_closest_center(sol)
            if center:
                main_basins[center].append(sol)

        for center, basin_sols in main_basins.items():
            sub_centers = self._find_density_peaks(basin_sols)
            self.sub_attractors[center] = []

            for sub_center in sub_centers:
                sub_sols = [s for s in basin_sols if self._is_near_center(s.ratios(), sub_center, threshold=0.001)]
                if len(sub_sols) >= 5:
                    sub_attractor = Attractor(sub_center, frozenset(sub_sols), AttractorType.DERIVED)
                    self.sub_attractors[center].append(sub_attractor)

    def _find_density_peaks(self, solutions: List[Solution]) -> List[Tuple[float, float, float]]:
        if not solutions:
            return []

        ratios = np.array([s.ratios() for s in solutions])
        kde = gaussian_kde(ratios.T)

        grid_points = np.linspace(0.1, 2.0, 50)
        density_map = np.zeros((50, 50, 50))

        # Fix the deprecated behavior
        for i, x in enumerate(grid_points):
            for j, y in enumerate(grid_points):
                for k, z in enumerate(grid_points):
                    point = np.array([[x], [y], [z]])  # Reshape for KDE
                    density_map[i,j,k] = float(kde(point))  # Explicit conversion

        # Use more sophisticated peak finding
        from scipy.signal import find_peaks_cwt
        peaks = []
        for i in range(3):  # For each dimension
            peak_indices = find_peaks_cwt(np.mean(density_map, axis=i), np.array([2, 3, 4]))
            for idx in peak_indices:
                coord = grid_points[idx]
                if 0.2 < coord < 1.8:  # Filter out edge cases
                    peaks.append(coord)

        return list(zip(peaks[::3], peaks[1::3], peaks[2::3]))

    def _analyze_fractal_structure(self) -> None:
        for center, sub_attractors in self.sub_attractors.items():
            if not sub_attractors:
                continue

            points = np.array([s.ratios() for attractor in sub_attractors
                             for s in attractor.solutions])

            scales = np.logspace(-3, 0, 20)
            counts = []

            for scale in scales:
                bins = np.ceil((points.max(axis=0) - points.min(axis=0)) / scale)
                counts.append(len(np.unique(np.floor(points/scale), axis=0)))

            slope, _, _, _, _ = linregress(np.log(scales), np.log(counts))
            self.fractal_dimensions[str(center)] = -slope

    def _find_rational_relations(self) -> None:
        for center1, center2 in combinations(self.attractors.keys(), 2):
            ratios = []
            for r1, r2 in zip(center1, center2):
                if abs(r2) > 1e-10:
                    ratio = r1/r2
                    frac = Fraction(ratio).limit_denominator(12)
                    ratios.append((frac.numerator, frac.denominator))

            if all(r[1] <= 12 for r in ratios):
                self.rational_relations.append((
                    tuple(r[0] for r in ratios),
                    tuple(r[1] for r in ratios)
                ))

    def _compile_analysis(self) -> Dict:
        base = {
            'attractors': {
                'count': len(self.attractors),
                'by_type': {
                    t.name: len(self.attractor_hierarchy[t])
                    for t in AttractorType
                },
                'stability': {
                    str(center): attractor.stability
                    for center, attractor in self.attractors.items()
                },
                'basin_sizes': {
                    str(center): attractor.basin_size
                    for center, attractor in self.attractors.items()
                }
            },
            'transformations': {
                'count': self.transformation_graph.number_of_edges(),
                'types': [
                    {
                        'from': str(e[0].center),
                        'to': str(e[1].center),
                        'transform': str(self.transformation_graph[e[0]][e[1]]['transform'])
                    }
                    for e in self.transformation_graph.edges()
                ]
            },
            'hierarchy': {
                'levels': len(set(nx.shortest_path_length(
                    self.transformation_graph,
                    source=next(iter(self.attractors.values()), None)
                ).values())) if self.attractors else 0,
                'branching_factor': np.mean([
                    d for _, d in self.transformation_graph.degree()
                ]) if self.attractors else 0
            }
        }

        base['multiscale'] = {
            'sub_attractors': {
                str(center): len(attractors)
                for center, attractors in self.sub_attractors.items()
            },
            'fractal_dimensions': self.fractal_dimensions,
            'rational_relations': [
                {
                    'numerators': nums,
                    'denominators': denoms
                }
                for nums, denoms in self.rational_relations
            ]
        }
        return base

class ComprehensiveDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.attractor_analyzer = MultiscaleAttractorAnalyzer()
        self.constant_analyzer = StructuralConstantAnalyzer()
        self.generation_times: Dict[str, float] = {}

    def generate_and_analyze(self):
        start_time = time.time()
        print(f"\nAnalyzing solutions up to {self.limit}...")

        t0 = time.time()
        self._generate_solutions()
        self.generation_times['generation'] = time.time() - t0

        t0 = time.time()
        self._analyze_structure()
        self.generation_times['analysis'] = time.time() - t0

        self.generation_times['total'] = time.time() - start_time

    def _generate_solutions(self):
        sqrt_limit = isqrt(self.limit)
        for x in range(2, self.limit):
            x_squared = x*x
            y_start = x + 1
            y_end = min(self.limit, isqrt(self.limit*self.limit - x_squared + 1))

            for y in range(y_start, y_end + 1):
                z_squared = x_squared + y*y - 1
                z = isqrt(z_squared)
                if z <= self.limit and z*z == z_squared:
                    self._try_add_solution(x, y, z)

    def _try_add_solution(self, x: int, y: int, z: int):
        if max(x, y, z) <= self.limit and min(x, y) > 0:
            try:
                sol = Solution(x, y, z)
                self.solutions.add(sol)
                if sol.is_primitive() and not sol.is_trivial():
                    self.primitive_solutions.append(sol)

                if x != y:
                    sym_sol = Solution(y, x, z)
                    self.solutions.add(sym_sol)
                    if sym_sol.is_primitive() and not sym_sol.is_trivial():
                        self.primitive_solutions.append(sym_sol)
            except AssertionError:
                pass

    def _analyze_structure(self):
        sorted_primitives = sorted(self.primitive_solutions, key=lambda s: (s.z, s.y, s.x))
        self.attractor_analysis = self.attractor_analyzer.analyze_solutions(sorted_primitives)
        self.constant_analysis = self.constant_analyzer.analyze_constants(sorted_primitives)

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'attractor_analysis': self.attractor_analysis,
            'constant_analysis': self.constant_analysis,
            'timing': self.generation_times
        }

class StructuralConstantAnalyzer:
    def __init__(self):
        self.complexity_constant = log(2.86)
        self.ratio_constants = {
            'primary': (3/4, 4/5, 3/5),
            'secondary': (4/3, 3/5, 4/5),
            'tertiary': (5/12, 11/12, 5/13)
        }

    def analyze_constants(self, solutions: List[Solution]) -> Dict:
        return {
            'complexity': self._analyze_complexity(solutions),
            'ratios': self._analyze_ratios(solutions)
        }

    def _analyze_complexity(self, solutions: List[Solution]) -> Dict:
        complexities = []
        for sol in solutions:
            ratios = sol.ratios()
            complexity = entropy(ratios) / log(3)
            complexities.append(complexity)

        return {
            'mean': float(np.mean(complexities)),
            'std': float(np.std(complexities)),
            'theoretical': self.complexity_constant,
            'deviation': float(abs(np.mean(complexities) - self.complexity_constant))
        }

    def _analyze_ratios(self, solutions: List[Solution]) -> Dict:
        ratio_deviations = defaultdict(list)
        for sol in solutions:
            ratios = sol.ratios()
            for const_type, const_ratios in self.ratio_constants.items():
                deviation = sum(abs(r1 - r2) for r1, r2 in zip(ratios, const_ratios))
                ratio_deviations[const_type].append(deviation)

        return {
            const_type: {
                'mean_deviation': float(np.mean(deviations)),
                'std_deviation': float(np.std(deviations))
            }
            for const_type, deviations in ratio_deviations.items()
        }

def main():
    analyzer = ComprehensiveDiophantineAnalyzer(100000)
    analyzer.generate_and_analyze()
    analysis = analyzer.get_analysis()

    print("\nSolution Counts:")
    print(f"Total: {analysis['solution_counts']['total']}")
    print(f"Primitive: {analysis['solution_counts']['primitive']}")

    print("\nAttractor Analysis:")
    attractor_analysis = analysis['attractor_analysis']
    print(f"Number of attractors: {attractor_analysis['attractors']['count']}")
    print("\nBy type:")
    for type_name, count in attractor_analysis['attractors']['by_type'].items():
        print(f"{type_name}: {count}")

    print("\nMultiscale Analysis:")
    if 'multiscale' in attractor_analysis:
        print("\nSub-attractors by main attractor:")
        for center, count in attractor_analysis['multiscale']['sub_attractors'].items():
            print(f"{center}: {count}")

        print("\nFractal Dimensions:")
        for center, dim in attractor_analysis['multiscale']['fractal_dimensions'].items():
            print(f"{center}: {dim:.3f}")

        print("\nRational Relations:")
        for relation in attractor_analysis['multiscale']['rational_relations']:
            print(f"Numerators: {relation['numerators']}")
            print(f"Denominators: {relation['denominators']}")

    print("\nStructural Constants Analysis:")
    constant_analysis = analysis['constant_analysis']
    print("\nComplexity:")
    for key, value in constant_analysis['complexity'].items():
        print(f"{key}: {value:.6f}")

    print("\nRatio Deviations:")
    for const_type, stats in constant_analysis['ratios'].items():
        print(f"\n{const_type}:")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value:.6f}")

    print(f"\nTotal analysis time: {analysis['timing']['total']:.2f} seconds")

if __name__ == "__main__":
    main()
