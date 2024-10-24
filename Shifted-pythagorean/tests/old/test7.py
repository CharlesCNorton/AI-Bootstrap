from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, Generator
from math import gcd, sqrt, isqrt, log
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain
from enum import Enum

class PatternType(Enum):
    MONOTONIC = "monotonic"
    MIXED = "mixed"
    CONSERVED = "conserved"
    SCALING = "scaling"
    SYMMETRIC = "symmetric"
    Z_PRESERVING = "z_preserving"

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

    def difference_vector(self, other: 'Solution') -> Tuple[int, int, int]:
        return (other.x - self.x, other.y - self.y, other.z - self.z)

    def scale_factor(self, other: 'Solution') -> float:
        factors = [other.x/self.x if self.x != 0 else 0,
                  other.y/self.y if self.y != 0 else 0,
                  other.z/self.z if self.z != 0 else 0]
        return np.mean([f for f in factors if f != 0])

    def ratio_vector(self) -> Tuple[float, float]:
        """Returns (x/y, y/z) ratio vector"""
        return (self.x/self.y, self.y/self.z)

@dataclass
class Pattern:
    start: Solution
    end: Solution
    pattern_type: PatternType
    diff_vector: Tuple[int, int, int] = field(init=False)
    scale_factor: float = field(init=False)
    ratio_change: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.diff_vector = self.start.difference_vector(self.end)
        self.scale_factor = self.end.scale_factor(self.start)
        start_ratios = self.start.ratio_vector()
        end_ratios = self.end.ratio_vector()
        self.ratio_change = tuple(e - s for s, e in zip(start_ratios, end_ratios))

    @property
    def z_preserved(self) -> bool:
        return self.start.z == self.end.z

    @property
    def normalized_diff(self) -> Tuple[float, float, float]:
        max_diff = max(abs(d) for d in self.diff_vector)
        return tuple(d/max_diff if max_diff > 0 else d for d in self.diff_vector)

@dataclass
class ScalingSequence:
    solutions: List[Solution]
    z_value: int
    ratio_progression: List[Tuple[float, float]]
    scale_factors: List[float]

    @property
    def length(self) -> int:
        return len(self.solutions)

    @property
    def average_scale(self) -> float:
        return np.mean(self.scale_factors)

    @property
    def ratio_stability(self) -> float:
        """Measure how stable the ratios are across the sequence"""
        return np.std([r[0]/r[1] for r in self.ratio_progression])

class PatternAnalyzer:
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.pattern_families: Dict[PatternType, List[Pattern]] = defaultdict(list)
        self.scaling_sequences: List[ScalingSequence] = []
        self.z_preservation_map: Dict[int, List[Solution]] = defaultdict(list)
        self.ratio_clusters: Dict[Tuple[float, float], List[Solution]] = defaultdict(list)

    def analyze_patterns(self, solutions: List[Solution]) -> None:
        """Comprehensive pattern analysis"""
        self._find_basic_patterns(solutions)
        self._find_scaling_sequences(solutions)
        self._analyze_z_preservation(solutions)
        self._analyze_ratio_clusters(solutions)

    def _find_basic_patterns(self, solutions: List[Solution]) -> None:
        for s1, s2 in zip(solutions, solutions[1:]):
            diff = s1.difference_vector(s2)

            if all(d > 0 for d in diff):
                pattern_type = PatternType.MONOTONIC
            elif sum(diff) == 0:
                pattern_type = PatternType.CONSERVED
            else:
                pattern_type = PatternType.MIXED

            pattern = Pattern(s1, s2, pattern_type)
            self.patterns.append(pattern)
            self.pattern_families[pattern_type].append(pattern)

    def _find_scaling_sequences(self, solutions: List[Solution]) -> None:
        current_sequence = [solutions[0]]
        current_ratios = [solutions[0].ratio_vector()]
        current_scales = []

        for s in solutions[1:]:
            prev = current_sequence[-1]
            scale = s.scale_factor(prev)

            if 0.9 < scale < 1.1 and s.z == prev.z:  # Z-preserving scaling
                current_sequence.append(s)
                current_ratios.append(s.ratio_vector())
                current_scales.append(scale)
            else:
                if len(current_sequence) > 2:
                    seq = ScalingSequence(
                        solutions=current_sequence.copy(),
                        z_value=current_sequence[0].z,
                        ratio_progression=current_ratios.copy(),
                        scale_factors=current_scales.copy()
                    )
                    self.scaling_sequences.append(seq)
                current_sequence = [s]
                current_ratios = [s.ratio_vector()]
                current_scales = []

    def _analyze_z_preservation(self, solutions: List[Solution]) -> None:
        for s in solutions:
            self.z_preservation_map[s.z].append(s)

    def _analyze_ratio_clusters(self, solutions: List[Solution]) -> None:
        for s in solutions:
            ratios = tuple(round(r, 3) for r in s.ratio_vector())
            self.ratio_clusters[ratios].append(s)

    def get_pattern_statistics(self) -> Dict:
        stats = {}
        for pattern_type, patterns in self.pattern_families.items():
            if not patterns:
                continue

            diffs = [p.diff_vector for p in patterns]
            avg_diff = tuple(map(lambda x: sum(x)/len(x), zip(*diffs)))

            scales = [p.scale_factor for p in patterns]
            ratio_changes = [p.ratio_change for p in patterns]

            stats[pattern_type] = {
                'count': len(patterns),
                'average_difference': avg_diff,
                'average_scale': sum(scales)/len(scales),
                'average_ratio_change': tuple(map(lambda x: sum(x)/len(x), zip(*ratio_changes))),
                'examples': patterns[:3]
            }

        return stats

    def get_scaling_statistics(self) -> Dict:
        if not self.scaling_sequences:
            return {}

        return {
            'number_of_sequences': len(self.scaling_sequences),
            'longest_sequence': max(s.length for s in self.scaling_sequences),
            'average_sequence_length': np.mean([s.length for s in self.scaling_sequences]),
            'z_value_distribution': len(set(s.z_value for s in self.scaling_sequences)),
            'most_stable_sequence': min(self.scaling_sequences, key=lambda s: s.ratio_stability),
            'largest_scale_sequence': max(self.scaling_sequences, key=lambda s: s.average_scale)
        }

    def get_z_preservation_statistics(self) -> Dict:
        return {
            'unique_z_values': len(self.z_preservation_map),
            'max_solutions_per_z': max(len(sols) for sols in self.z_preservation_map.values()),
            'average_solutions_per_z': np.mean([len(sols) for sols in self.z_preservation_map.values()]),
            'most_common_z': max(self.z_preservation_map.items(), key=lambda x: len(x[1]))[0]
        }

    def get_ratio_statistics(self) -> Dict:
        return {
            'unique_ratio_pairs': len(self.ratio_clusters),
            'most_common_ratios': sorted(
                [(ratios, len(sols)) for ratios, sols in self.ratio_clusters.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.pattern_analyzer = PatternAnalyzer()
        self.generation_times: Dict[str, float] = {}

    def generate_and_analyze(self):
        start_time = time.time()
        print(f"\nAnalyzing solutions up to {self.limit}...")

        t0 = time.time()
        self._generate_solutions()
        self.generation_times['generation'] = time.time() - t0

        t0 = time.time()
        self._analyze_patterns()
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

    def _analyze_patterns(self):
        sorted_primitives = sorted(self.primitive_solutions, key=lambda s: (s.z, s.y, s.x))
        self.pattern_analyzer.analyze_patterns(sorted_primitives)

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'pattern_statistics': self.pattern_analyzer.get_pattern_statistics(),
            'scaling_statistics': self.pattern_analyzer.get_scaling_statistics(),
            'z_preservation': self.pattern_analyzer.get_z_preservation_statistics(),
            'ratio_statistics': self.pattern_analyzer.get_ratio_statistics(),
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

        print("\nPattern Statistics:")
        for pattern_type, stats in analysis['pattern_statistics'].items():
            print(f"\n{pattern_type.value.title()} Patterns ({stats['count']} instances):")
            print(f"Average difference: {tuple(round(x,3) for x in stats['average_difference'])}")
            print(f"Average scale: {stats['average_scale']:.3f}")
            print(f"Average ratio change: {tuple(round(x,3) for x in stats['average_ratio_change'])}")
            print("Examples:")
            for pattern in stats['examples']:
                print(f"  ({pattern.start.x},{pattern.start.y},{pattern.start.z}) -> "
                      f"({pattern.end.x},{pattern.end.y},{pattern.end.z})")
                print(f"  Difference vector: {pattern.diff_vector}")

        print("\nScaling Sequence Statistics:")
        for key, value in analysis['scaling_statistics'].items():
            print(f"{key}: {value}")

        print("\nZ-Value Preservation:")
        for key, value in analysis['z_preservation'].items():
            print(f"{key}: {value}")

        print("\nRatio Statistics:")
        print("Most common ratio pairs (x/y, y/z):")
        for ratios, count in analysis['ratio_statistics']['most_common_ratios']:
            print(f"{ratios}: {count} occurrences")

        print(f"\nTotal analysis time: {analysis['generation_times']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
