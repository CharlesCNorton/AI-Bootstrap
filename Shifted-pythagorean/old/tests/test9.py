from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple
from math import gcd, sqrt, isqrt, log, ceil
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain
from enum import Enum
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class PatternType(Enum):
    MONOTONIC = "monotonic"
    MIXED = "mixed"
    CONSERVED = "conserved"
    Z_PRESERVING = "z_preserving"
    RATIO_LOCKED = "ratio_locked"
    HIERARCHICAL = "hierarchical"
    COMPOSITE = "composite"

class PatternRelation(NamedTuple):
    type1: PatternType
    type2: PatternType
    strength: float
    shared_ratios: List[float]

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

    def scale_to(self, target: 'Solution') -> float:
        v1, v2 = self.vector_form(), target.vector_form()
        scales = v2/v1
        return float(np.median(scales[scales != np.inf]))

@dataclass
class Pattern:
    solutions: List[Solution]
    pattern_type: PatternType
    scale_factors: List[float] = field(default_factory=list)
    ratio_changes: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.scale_factors = [s2.scale_to(s1) for s1, s2 in zip(self.solutions, self.solutions[1:])]
        self.ratio_changes = [tuple(r2 - r1 for r1, r2 in zip(s1.ratios(), s2.ratios()))
                            for s1, s2 in zip(self.solutions, self.solutions[1:])]

    @property
    def z_values(self) -> Set[int]:
        return {s.z for s in self.solutions}

    @property
    def ratio_set(self) -> Set[Tuple[float, float, float]]:
        return {s.ratios() for s in self.solutions}

    @property
    def hierarchical_level(self) -> int:
        if not self.scale_factors:
            return 0
        return int(abs(log(np.mean(self.scale_factors), 2)))

    def similarity_to(self, other: 'Pattern') -> float:
        shared_z = len(self.z_values & other.z_values)
        shared_ratios = len(self.ratio_set & other.ratio_set)
        return (shared_z + shared_ratios) / (len(self.z_values) + len(self.ratio_set))

class HierarchicalAnalyzer:
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.z_hierarchy: Dict[int, List[Pattern]] = defaultdict(list)
        self.ratio_hierarchy: Dict[Tuple[float, float, float], List[Pattern]] = defaultdict(list)
        self.pattern_relations: List[PatternRelation] = []

    def analyze_structure(self, solutions: List[Solution]) -> Dict:
        print(f"Analyzing hierarchical structure of {len(solutions)} solutions...")

        self._find_patterns(solutions)
        self._build_hierarchies()
        self._analyze_relations()
        return self._compile_analysis()

    def _find_patterns(self, solutions: List[Solution]) -> None:
        window_size = min(1000, len(solutions))

        for i in range(0, len(solutions), window_size//2):
            window = solutions[i:i+window_size]
            if len(window) < 3:
                continue

            current_pattern = [window[0]]
            current_type = None

            for s1, s2 in zip(window, window[1:]):
                scale = s2.scale_to(s1)

                if 0.9 < scale < 1.1:
                    if current_type is None:
                        current_type = self._determine_pattern_type(s1, s2)
                    current_pattern.append(s2)
                else:
                    if len(current_pattern) >= 3:
                        pattern = Pattern(current_pattern.copy(), current_type)
                        self.patterns.append(pattern)
                        self._classify_pattern(pattern)
                    current_pattern = [s2]
                    current_type = None

            if len(current_pattern) >= 3:
                pattern = Pattern(current_pattern, current_type)
                self.patterns.append(pattern)
                self._classify_pattern(pattern)

    def _determine_pattern_type(self, s1: Solution, s2: Solution) -> PatternType:
        v1, v2 = s1.vector_form(), s2.vector_form()
        diff = v2 - v1

        if s1.z == s2.z:
            return PatternType.Z_PRESERVING
        elif abs(sum(diff)) < 1e-10:
            return PatternType.CONSERVED
        elif all(abs(r1 - r2) < 0.01 for r1, r2 in zip(s1.ratios(), s2.ratios())):
            return PatternType.RATIO_LOCKED
        elif all(d >= 0 for d in diff):
            return PatternType.MONOTONIC
        else:
            return PatternType.MIXED

    def _classify_pattern(self, pattern: Pattern) -> None:
        for z in pattern.z_values:
            self.z_hierarchy[z].append(pattern)

        for ratios in pattern.ratio_set:
            self.ratio_hierarchy[ratios].append(pattern)

    def _build_hierarchies(self) -> None:
        if not self.patterns:
            return

        # Cluster patterns by similarity
        similarity_matrix = np.zeros((len(self.patterns), len(self.patterns)))
        for i, p1 in enumerate(self.patterns):
            for j, p2 in enumerate(self.patterns[i+1:], i+1):
                sim = p1.similarity_to(p2)
                similarity_matrix[i,j] = similarity_matrix[j,i] = sim

        linkage_matrix = linkage(1 - similarity_matrix, method='ward')
        clusters = fcluster(linkage_matrix, t=0.7, criterion='distance')

        # Identify hierarchical patterns
        for i, cluster_id in enumerate(clusters):
            if sum(clusters == cluster_id) > 1:
                self.patterns[i].pattern_type = PatternType.HIERARCHICAL

    def _analyze_relations(self) -> None:
        for p1, p2 in combinations(self.patterns, 2):
            shared_z = len(p1.z_values & p2.z_values)
            shared_ratios = list(p1.ratio_set & p2.ratio_set)
            if shared_z > 0 or shared_ratios:
                strength = p1.similarity_to(p2)
                relation = PatternRelation(p1.pattern_type, p2.pattern_type,
                                        strength, shared_ratios)
                self.pattern_relations.append(relation)

    def _compile_analysis(self) -> Dict:
        pattern_stats = defaultdict(int)
        for pattern in self.patterns:
            pattern_stats[pattern.pattern_type] += 1

        z_stats = {
            'unique_z': len(self.z_hierarchy),
            'max_patterns_per_z': max(len(patterns) for patterns in self.z_hierarchy.values()),
            'avg_patterns_per_z': np.mean([len(patterns) for patterns in self.z_hierarchy.values()]),
            'hierarchical_levels': max(p.hierarchical_level for p in self.patterns) + 1
        }

        ratio_stats = {
            'unique_ratios': len(self.ratio_hierarchy),
            'max_patterns_per_ratio': max(len(patterns) for patterns in self.ratio_hierarchy.values()),
            'avg_patterns_per_ratio': np.mean([len(patterns) for patterns in self.ratio_hierarchy.values()]),
            'dominant_ratios': sorted(
                [(ratio, len(patterns)) for ratio, patterns in self.ratio_hierarchy.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

        relation_stats = defaultdict(list)
        for relation in self.pattern_relations:
            relation_stats[f"{relation.type1.value}-{relation.type2.value}"].append(relation.strength)

        return {
            'pattern_counts': dict(pattern_stats),
            'z_hierarchy': z_stats,
            'ratio_hierarchy': ratio_stats,
            'pattern_relations': {
                key: np.mean(strengths)
                for key, strengths in relation_stats.items()
            }
        }

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.hierarchical_analyzer = HierarchicalAnalyzer()
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
        self.hierarchical_analysis = self.hierarchical_analyzer.analyze_structure(sorted_primitives)

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'hierarchical_analysis': self.hierarchical_analysis,
            'generation_times': self.generation_times
        }

def analyze_large_scale(limits: List[int] = [1000, 10000, 100000]):
    for limit in limits:
        analyzer = EnhancedDiophantineAnalyzer(limit)
        analyzer.generate_and_analyze()
        analysis = analyzer.get_analysis()

        print(f"\n{'='*50}")
        print(f"Analysis for N={limit}")

        print("\nSolution Counts:")
        for count_type, count in analysis['solution_counts'].items():
            print(f"{count_type}: {count}")

        hier = analysis['hierarchical_analysis']

        print("\nPattern Distribution:")
        for ptype, count in hier['pattern_counts'].items():
            print(f"{ptype}: {count}")

        print("\nZ-Value Hierarchy:")
        for stat, value in hier['z_hierarchy'].items():
            print(f"{stat}: {value}")

        print("\nRatio Hierarchy:")
        print("Dominant ratios:")
        for ratio, count in hier['ratio_hierarchy']['dominant_ratios']:
            print(f"{ratio}: {count} patterns")

        print("\nPattern Relations:")
        for relation, strength in sorted(hier['pattern_relations'].items()):
            print(f"{relation}: {strength:.3f}")

        print(f"\nTotal analysis time: {analysis['generation_times']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
