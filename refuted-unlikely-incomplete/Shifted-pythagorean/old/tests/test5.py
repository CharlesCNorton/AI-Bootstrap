from dataclasses import dataclass
from typing import Set, List, Tuple, Dict, Optional
from math import gcd, sqrt, isqrt
from collections import defaultdict
import time
import numpy as np
from itertools import combinations

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
        """Calculate approximate scaling factor between solutions"""
        factors = [other.x/self.x if self.x != 0 else 0,
                  other.y/self.y if self.y != 0 else 0,
                  other.z/self.z if self.z != 0 else 0]
        return np.mean([f for f in factors if f != 0])

@dataclass
class Pattern:
    start: Solution
    end: Solution
    diff_vector: Tuple[int, int, int]
    pattern_type: str

    @property
    def scale(self) -> float:
        return self.end.scale_factor(self.start)

    @property
    def ratio_change(self) -> Tuple[float, float, float]:
        start_ratios = self.start.ratios()
        end_ratios = self.end.ratios()
        return tuple(e - s for s, e in zip(start_ratios, end_ratios))

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: Set[Solution] = set()
        self.patterns: List[Pattern] = []
        self.pattern_families: Dict[str, List[Pattern]] = defaultdict(list)
        self.scaling_sequences: List[List[Solution]] = []
        self.generation_times: Dict[str, float] = {}

    def generate_all(self):
        start_time = time.time()
        print(f"\nGenerating solutions up to {self.limit}...")

        t0 = time.time()
        self._generate_solutions_optimized()
        self.generation_times['generation'] = time.time() - t0

        t0 = time.time()
        self._analyze_patterns()
        self.generation_times['pattern_analysis'] = time.time() - t0

        t0 = time.time()
        self._find_scaling_sequences()
        self.generation_times['scaling_analysis'] = time.time() - t0

        self.generation_times['total'] = time.time() - start_time

    def _generate_solutions_optimized(self):
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
                if sol.is_primitive():
                    self.primitive_solutions.add(sol.normalized())

                if x != y:
                    sym_sol = Solution(y, x, z)
                    self.solutions.add(sym_sol)
                    if sym_sol.is_primitive():
                        self.primitive_solutions.add(sym_sol.normalized())
            except AssertionError:
                pass

    def _analyze_patterns(self):
        primitives = sorted([s for s in self.primitive_solutions if not s.is_trivial()],
                          key=lambda s: (s.z, s.y, s.x))

        for i in range(len(primitives)-1):
            s1, s2 = primitives[i], primitives[i+1]
            diff = s1.difference_vector(s2)

            if all(d > 0 for d in diff):
                pattern_type = 'monotonic'
            elif sum(diff) == 0:
                pattern_type = 'conserved'
            else:
                pattern_type = 'mixed'

            pattern = Pattern(s1, s2, diff, pattern_type)
            self.patterns.append(pattern)
            self.pattern_families[pattern_type].append(pattern)

            # Look for similar patterns
            pattern_key = tuple(d/max(abs(d) for d in diff) if max(abs(d) for d in diff) > 0 else d for d in diff)
            self.pattern_families[f"family_{pattern_key}"].append(pattern)

    def _find_scaling_sequences(self):
        """Find sequences of solutions that scale uniformly"""
        primitives = sorted([s for s in self.primitive_solutions if not s.is_trivial()],
                          key=lambda s: (s.z, s.y, s.x))

        current_sequence = [primitives[0]]
        for i in range(1, len(primitives)):
            prev = current_sequence[-1]
            curr = primitives[i]

            # Check if current solution follows scaling pattern
            scale = curr.scale_factor(prev)
            if 0.9 < scale < 1.1:  # Allow 10% variation
                current_sequence.append(curr)
            else:
                if len(current_sequence) > 2:
                    self.scaling_sequences.append(current_sequence)
                current_sequence = [curr]

        if len(current_sequence) > 2:
            self.scaling_sequences.append(current_sequence)

    def analyze(self) -> Dict:
        analysis = {
            'total_solutions': len(self.solutions),
            'primitive_solutions': len(self.primitive_solutions),
            'pattern_counts': {k: len(v) for k, v in self.pattern_families.items()},
            'scaling_sequences': len(self.scaling_sequences),
            'generation_times': self.generation_times
        }

        # Analyze pattern families
        pattern_analysis = defaultdict(list)
        for pattern_type, patterns in self.pattern_families.items():
            if pattern_type.startswith('family_'):
                continue

            diffs = [p.diff_vector for p in patterns]
            if diffs:
                avg_diff = tuple(map(lambda x: sum(x)/len(x), zip(*diffs)))
                scales = [p.scale for p in patterns]
                ratio_changes = [p.ratio_change for p in patterns]

                pattern_analysis[pattern_type] = {
                    'count': len(patterns),
                    'average_difference': avg_diff,
                    'average_scale': sum(scales)/len(scales),
                    'average_ratio_change': tuple(map(lambda x: sum(x)/len(x), zip(*ratio_changes))),
                    'examples': patterns[:3]
                }

        analysis['pattern_analysis'] = dict(pattern_analysis)

        # Analyze scaling sequences
        if self.scaling_sequences:
            longest_sequence = max(self.scaling_sequences, key=len)
            analysis['scaling_analysis'] = {
                'number_of_sequences': len(self.scaling_sequences),
                'longest_sequence_length': len(longest_sequence),
                'example_sequence': longest_sequence[:5]
            }

        return analysis

def analyze_large_scale(limits: List[int] = [1000, 10000, 100000]):
    for limit in limits:
        print(f"\n{'='*50}")
        print(f"Analyzing solutions up to N={limit}")

        analyzer = EnhancedDiophantineAnalyzer(limit)
        analyzer.generate_all()
        analysis = analyzer.analyze()

        print(f"\nGeneration Times:")
        for method, time_taken in analysis['generation_times'].items():
            print(f"{method}: {time_taken:.2f} seconds")

        print(f"\nSolution Counts:")
        print(f"Total solutions: {analysis['total_solutions']}")
        print(f"Primitive solutions: {analysis['primitive_solutions']}")

        print("\nPattern Analysis:")
        for pattern_type, data in analysis['pattern_analysis'].items():
            print(f"\n{pattern_type.title()} Patterns ({data['count']} instances):")
            print(f"Average difference vector: {tuple(round(x,3) for x in data['average_difference'])}")
            print(f"Average scale factor: {data['average_scale']:.3f}")
            print(f"Average ratio change: {tuple(round(x,3) for x in data['average_ratio_change'])}")
            print("Examples:")
            for pattern in data['examples']:
                print(f"  ({pattern.start.x},{pattern.start.y},{pattern.start.z}) -> "
                      f"({pattern.end.x},{pattern.end.y},{pattern.end.z})")
                print(f"  Difference vector: {pattern.diff_vector}")

        if 'scaling_analysis' in analysis:
            print("\nScaling Sequence Analysis:")
            print(f"Number of sequences: {analysis['scaling_analysis']['number_of_sequences']}")
            print(f"Longest sequence length: {analysis['scaling_analysis']['longest_sequence_length']}")
            print("Example sequence:")
            for sol in analysis['scaling_analysis']['example_sequence']:
                print(f"  ({sol.x}, {sol.y}, {sol.z})")

        print(f"\nTotal analysis time: {analysis['generation_times']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
