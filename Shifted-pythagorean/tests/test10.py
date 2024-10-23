from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple
from math import gcd, sqrt, isqrt, log, ceil
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain
from enum import Enum
from scipy.stats import linregress

class PatternType(Enum):
    HIERARCHICAL = "hierarchical"
    Z_PRESERVING = "z_preserving"
    RATIO_LOCKED = "ratio_locked"
    COMPOSITE = "composite"

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

@dataclass
class PatternGroup:
    solutions: List[Solution]
    pattern_type: PatternType
    z_value: Optional[int] = None
    ratio_signature: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if len(self.solutions) > 0:
            self.z_value = self.solutions[0].z
            self.ratio_signature = tuple(
                round(np.mean([s.ratios()[i] for s in self.solutions]), 3)
                for i in range(3)
            )

class StructuralAnalyzer:
    def __init__(self):
        self.pattern_groups: List[PatternGroup] = []
        self.z_families: Dict[int, List[Solution]] = defaultdict(list)
        self.ratio_families: Dict[Tuple[float, float, float], List[Solution]] = defaultdict(list)

    def analyze_structure(self, solutions: List[Solution], prev_analysis: Optional[Dict] = None) -> Dict:
        print(f"Analyzing structural patterns in {len(solutions)} solutions...")

        self._find_patterns(solutions)
        self._classify_patterns()

        return self._compile_analysis()

    def _find_patterns(self, solutions: List[Solution]) -> None:
        # Group by z-value
        z_groups = defaultdict(list)
        for sol in solutions:
            z_groups[sol.z].append(sol)

        # Find patterns within z-groups
        for z, group in z_groups.items():
            if len(group) >= 2:
                self.z_families[z].extend(group)
                self._analyze_z_group(group)

        # Find ratio patterns
        ratio_groups = defaultdict(list)
        for sol in solutions:
            rounded = tuple(round(r, 3) for r in sol.ratios())
            ratio_groups[rounded].append(sol)
            if len(ratio_groups[rounded]) >= 2:
                self.ratio_families[rounded].extend(ratio_groups[rounded])

    def _analyze_z_group(self, group: List[Solution]) -> None:
        ratios = [s.ratios() for s in group]
        ratio_stability = 1 - np.std([r[0]/r[1] for r in ratios])

        if ratio_stability > 0.9:
            pattern_type = PatternType.RATIO_LOCKED
        elif len(set(s.z for s in group)) == 1:
            pattern_type = PatternType.Z_PRESERVING
        else:
            pattern_type = PatternType.COMPOSITE

        self.pattern_groups.append(PatternGroup(group, pattern_type))

    def _classify_patterns(self) -> None:
        # Find hierarchical patterns
        for ratio, solutions in self.ratio_families.items():
            if len(solutions) >= 3:
                self.pattern_groups.append(
                    PatternGroup(solutions, PatternType.HIERARCHICAL)
                )

    def _compile_analysis(self) -> Dict:
        pattern_counts = defaultdict(int)
        for group in self.pattern_groups:
            pattern_counts[group.pattern_type] += 1

        z_stats = {
            'unique_z': len(self.z_families),
            'max_solutions_per_z': max((len(sols) for sols in self.z_families.values()), default=0),
            'avg_solutions_per_z': np.mean([len(sols) for sols in self.z_families.values()]) if self.z_families else 0
        }

        ratio_stats = {
            'unique_ratios': len(self.ratio_families),
            'max_solutions_per_ratio': max((len(sols) for sols in self.ratio_families.values()), default=0),
            'avg_solutions_per_ratio': np.mean([len(sols) for sols in self.ratio_families.values()]) if self.ratio_families else 0,
            'most_common': sorted(
                [(ratio, len(sols)) for ratio, sols in self.ratio_families.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

        return {
            'pattern_counts': dict(pattern_counts),
            'z_families': z_stats,
            'ratio_families': ratio_stats
        }

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.structural_analyzer = StructuralAnalyzer()
        self.generation_times: Dict[str, float] = {}

    def generate_and_analyze(self, prev_analysis: Optional[Dict] = None):
        start_time = time.time()
        print(f"\nAnalyzing solutions up to {self.limit}...")

        t0 = time.time()
        self._generate_solutions()
        self.generation_times['generation'] = time.time() - t0

        t0 = time.time()
        self._analyze_structure(prev_analysis)
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

    def _analyze_structure(self, prev_analysis: Optional[Dict]):
        sorted_primitives = sorted(self.primitive_solutions, key=lambda s: (s.z, s.y, s.x))
        self.structural_analysis = self.structural_analyzer.analyze_structure(
            sorted_primitives, prev_analysis
        )

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'structural_analysis': self.structural_analysis,
            'timing': self.generation_times
        }

def analyze_large_scale(limits: List[int] = [1000, 10000, 100000]):
    prev_analysis = None

    for limit in limits:
        analyzer = EnhancedDiophantineAnalyzer(limit)
        analyzer.generate_and_analyze(prev_analysis)
        analysis = analyzer.get_analysis()

        print(f"\n{'='*50}")
        print(f"Analysis for N={limit}")

        print("\nSolution Counts:")
        for count_type, count in analysis['solution_counts'].items():
            print(f"{count_type}: {count}")

        struct = analysis['structural_analysis']

        print("\nPattern Distribution:")
        for ptype, count in struct['pattern_counts'].items():
            print(f"{ptype.value}: {count}")

        print("\nZ-Family Analysis:")
        z_stats = struct['z_families']
        print(f"Unique z values: {z_stats['unique_z']}")
        print(f"Max solutions per z: {z_stats['max_solutions_per_z']}")
        print(f"Average solutions per z: {z_stats['avg_solutions_per_z']:.2f}")

        print("\nRatio Family Analysis:")
        r_stats = struct['ratio_families']
        print(f"Unique ratio combinations: {r_stats['unique_ratios']}")
        print(f"Max solutions per ratio: {r_stats['max_solutions_per_ratio']}")
        print("\nMost common ratios:")
        for ratio, count in r_stats['most_common']:
            print(f"{ratio}: {count} solutions")

        print(f"\nTotal analysis time: {analysis['timing']['total']:.2f} seconds")

        prev_analysis = analysis

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
