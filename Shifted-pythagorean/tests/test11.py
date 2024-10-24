from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple, Generator
from math import gcd, sqrt, isqrt, log, ceil, pi, e
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain, product
from enum import Enum, auto
from scipy.stats import linregress, entropy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import networkx as nx

class PatternType(Enum):
    TRIVIAL = auto()
    SYMMETRIC = auto()
    Z_PRESERVING = auto()
    RATIO_LOCKED = auto()
    HIERARCHICAL = auto()
    COMPOSITE = auto()
    FRACTAL = auto()
    ATTRACTOR = auto()

class StructuralProperty(Enum):
    GOLDEN_RATIO = (1 + sqrt(5))/2
    SQRT2 = sqrt(2)
    SQRT3 = sqrt(3)
    PI_RATIO = pi/2
    E_RATIO = e/2

@dataclass(frozen=True)
class Solution:
    x: int
    y: int
    z: int

    def __post_init__(self):
        assert self.x*self.x + self.y*self.y == self.z*self.z + 1

    def is_trivial(self) -> bool:
        return self.x == 1 or self.y == 1

    def is_symmetric(self) -> bool:
        return self.x == self.y

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def normalized(self) -> 'Solution':
        return Solution(min(self.x, self.y), max(self.x, self.y), self.z)

    def is_primitive(self) -> bool:
        return gcd(gcd(self.x, self.y), self.z) == 1

    def vector_form(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def structural_properties(self) -> Dict[StructuralProperty, float]:
        props = {}
        ratios = self.ratios()
        for prop in StructuralProperty:
            props[prop] = min(abs(r - prop.value) for r in ratios)
        return props

@dataclass
class Pattern:
    solutions: List[Solution]
    pattern_type: PatternType
    properties: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._analyze_properties()

    def _analyze_properties(self):
        if not self.solutions:
            return

        vectors = np.array([s.vector_form() for s in self.solutions])
        ratios = np.array([s.ratios() for s in self.solutions])

        self.properties.update({
            'dimension': self._fractal_dimension(vectors),
            'entropy': entropy(np.histogram(ratios.flatten(), bins=20)[0]),
            'stability': 1 - np.std(ratios, axis=0).mean(),
            'symmetry': self._symmetry_score(vectors),
            'complexity': self._complexity_score(vectors, ratios)
        })

    def _fractal_dimension(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        scales = np.log(dists)
        counts = np.log(np.arange(1, len(dists) + 1))
        slope, _, _, _, _ = linregress(scales, counts)
        return abs(slope)

    def _symmetry_score(self, points: np.ndarray) -> float:
        center = points.mean(axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        return 1 - np.std(dists)/np.mean(dists)

    def _complexity_score(self, points: np.ndarray, ratios: np.ndarray) -> float:
        spatial_complexity = np.linalg.matrix_rank(points)/points.shape[1]
        ratio_complexity = entropy(np.histogram(ratios.flatten(), bins=10)[0])
        return (spatial_complexity + ratio_complexity)/2

class MathematicalStructureAnalyzer:
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.ratio_families: Dict[Tuple[float, float, float], List[Solution]] = defaultdict(list)
        self.z_families: Dict[int, List[Solution]] = defaultdict(list)
        self.structural_graph = nx.Graph()
        self.attractor_basins: Dict[Tuple[float, float, float], Set[Solution]] = defaultdict(set)
        self.transformation_groups: List[List[Pattern]] = []

    def analyze_structure(self, solutions: List[Solution]) -> Dict:
        print(f"Analyzing mathematical structure of {len(solutions)} solutions...")

        self._find_patterns(solutions)
        self._analyze_relationships()
        self._identify_attractors()
        self._find_transformation_groups()

        return self._compile_analysis()

    def _find_patterns(self, solutions: List[Solution]) -> None:
        # Group by common properties
        groups = defaultdict(list)

        for sol in solutions:
            # Z-value patterns
            self.z_families[sol.z].append(sol)

            # Ratio patterns
            ratios = tuple(round(r, 4) for r in sol.ratios())
            self.ratio_families[ratios].append(sol)

            # Structural properties
            props = sol.structural_properties()
            min_prop = min(props.items(), key=lambda x: x[1])
            if min_prop[1] < 0.01:
                groups[min_prop[0]].append(sol)

        # Create patterns from groups
        for prop, sols in groups.items():
            if len(sols) >= 3:
                pattern_type = self._determine_pattern_type(sols)
                self.patterns.append(Pattern(sols, pattern_type))

        # Find composite patterns
        self._find_composite_patterns(solutions)

    def _determine_pattern_type(self, solutions: List[Solution]) -> PatternType:
        if all(s.is_trivial() for s in solutions):
            return PatternType.TRIVIAL
        elif all(s.is_symmetric() for s in solutions):
            return PatternType.SYMMETRIC
        elif len(set(s.z for s in solutions)) == 1:
            return PatternType.Z_PRESERVING
        elif self._has_stable_ratios(solutions):
            return PatternType.RATIO_LOCKED
        elif self._has_fractal_structure(solutions):
            return PatternType.FRACTAL
        else:
            return PatternType.COMPOSITE

    def _has_stable_ratios(self, solutions: List[Solution]) -> bool:
        ratios = np.array([s.ratios() for s in solutions])
        return np.std(ratios, axis=0).mean() < 0.01

    def _has_fractal_structure(self, solutions: List[Solution]) -> bool:
        if len(solutions) < 10:
            return False
        points = np.array([s.vector_form() for s in solutions])
        dim = Pattern(solutions, PatternType.FRACTAL).properties['dimension']
        return 1.1 < dim < 2.9

    def _find_composite_patterns(self, solutions: List[Solution]) -> None:
        # Find patterns that combine multiple properties
        for p1, p2 in combinations(self.patterns, 2):
            shared = set(p1.solutions) & set(p2.solutions)
            if len(shared) >= 3:
                self.patterns.append(Pattern(list(shared), PatternType.COMPOSITE))

    def _analyze_relationships(self) -> None:
        # Build relationship graph
        for p1, p2 in combinations(self.patterns, 2):
            similarity = self._pattern_similarity(p1, p2)
            if similarity > 0.5:
                self.structural_graph.add_edge(p1, p2, weight=similarity)

    def _pattern_similarity(self, p1: Pattern, p2: Pattern) -> float:
        prop_sim = sum(abs(p1.properties[k] - p2.properties[k])
                      for k in p1.properties if k in p2.properties)
        sol_sim = len(set(p1.solutions) & set(p2.solutions)) / \
                 len(set(p1.solutions) | set(p2.solutions))
        return (1 - prop_sim/len(p1.properties) + sol_sim)/2

    def _identify_attractors(self) -> None:
        # Find ratio combinations that act as attractors
        for ratios, solutions in self.ratio_families.items():
            if len(solutions) >= 10:
                basin = self._find_attractor_basin(ratios, solutions)
                if basin:
                    self.attractor_basins[ratios] = basin

    def _find_attractor_basin(self, ratios: Tuple[float, float, float],
                            solutions: List[Solution]) -> Optional[Set[Solution]]:
        basin = set(solutions)
        for sol in solutions:
            nearby = self._find_nearby_solutions(sol)
            if nearby:
                basin.update(nearby)
        return basin if len(basin) > len(solutions) else None

    def _find_nearby_solutions(self, sol: Solution) -> Set[Solution]:
        nearby = set()
        for pattern in self.patterns:
            if sol in pattern.solutions:
                for other in pattern.solutions:
                    if self._is_nearby(sol, other):
                        nearby.add(other)
        return nearby

    def _is_nearby(self, s1: Solution, s2: Solution) -> bool:
        return all(abs(r1 - r2) < 0.1 for r1, r2 in zip(s1.ratios(), s2.ratios()))

    def _find_transformation_groups(self) -> None:
        # Find groups of patterns related by transformations
        components = list(nx.connected_components(self.structural_graph))
        for component in components:
            if len(component) >= 3:
                self.transformation_groups.append(list(component))

    def _compile_analysis(self) -> Dict:
        pattern_stats = defaultdict(list)
        for pattern in self.patterns:
            for prop, value in pattern.properties.items():
                pattern_stats[f"{pattern.pattern_type.name}_{prop}"].append(value)

        z_stats = {
            'unique_z': len(self.z_families),
            'max_family': max(len(f) for f in self.z_families.values()),
            'avg_family': np.mean([len(f) for f in self.z_families.values()]),
            'distribution': np.histogram([len(f) for f in self.z_families.values()],
                                      bins=10)[0].tolist()
        }

        ratio_stats = {
            'unique_ratios': len(self.ratio_families),
            'max_family': max(len(f) for f in self.ratio_families.values()),
            'avg_family': np.mean([len(f) for f in self.ratio_families.values()]),
            'dominant': sorted(
                [(r, len(s)) for r, s in self.ratio_families.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

        structure_stats = {
            'pattern_counts': {
                ptype: len([p for p in self.patterns if p.pattern_type == ptype])
                for ptype in PatternType
            },
            'pattern_properties': {
                name: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
                for name, values in pattern_stats.items()
            },
            'attractors': {
                'count': len(self.attractor_basins),
                'avg_basin': np.mean([len(b) for b in self.attractor_basins.values()]),
                'largest_basin': max((len(b) for b in self.attractor_basins.values()),
                                  default=0)
            },
            'transformations': {
                'groups': len(self.transformation_groups),
                'avg_group': np.mean([len(g) for g in self.transformation_groups]),
                'max_group': max((len(g) for g in self.transformation_groups), default=0)
            }
        }

        return {
            'z_families': z_stats,
            'ratio_families': ratio_stats,
            'structure': structure_stats
        }

class ComprehensiveDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.structure_analyzer = MathematicalStructureAnalyzer()
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
        self.structure_analysis = self.structure_analyzer.analyze_structure(sorted_primitives)

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'mathematical_structure': self.structure_analysis,
            'timing': self.generation_times
        }

def analyze_large_scale(limits: List[int] = [1000, 10000, 100000]):
    for limit in limits:
        analyzer = ComprehensiveDiophantineAnalyzer(limit)
        analyzer.generate_and_analyze()
        analysis = analyzer.get_analysis()

        print(f"\n{'='*50}")
        print(f"Analysis for N={limit}")

        print("\nSolution Counts:")
        for count_type, count in analysis['solution_counts'].items():
            print(f"{count_type}: {count}")

        struct = analysis['mathematical_structure']

        print("\nPattern Distribution:")
        for ptype, count in struct['structure']['pattern_counts'].items():
            print(f"{ptype.name}: {count}")

        print("\nPattern Properties:")
        for name, stats in struct['structure']['pattern_properties'].items():
            print(f"\n{name}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.3f}")

        print("\nZ-Family Analysis:")
        z_stats = struct['z_families']
        print(f"Unique z values: {z_stats['unique_z']}")
        print(f"Max family size: {z_stats['max_family']}")
        print(f"Average family size: {z_stats['avg_family']:.2f}")
        print(f"Size distribution: {z_stats['distribution']}")

        print("\nRatio Family Analysis:")
        r_stats = struct['ratio_families']
        print(f"Unique ratio combinations: {r_stats['unique_ratios']}")
        print(f"Max family size: {r_stats['max_family']}")
        print("\nDominant ratios:")
        for ratio, count in r_stats['dominant']:
            print(f"{ratio}: {count} solutions")

        print("\nStructural Features:")
        print(f"Attractor count: {struct['structure']['attractors']['count']}")
        print(f"Average basin size: {struct['structure']['attractors']['avg_basin']:.2f}")
        print(f"Transformation groups: {struct['structure']['transformations']['groups']}")
        print(f"Average group size: {struct['structure']['transformations']['avg_group']:.2f}")

        print(f"\nTotal analysis time: {analysis['timing']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
