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
        assert self.x*self.x + self.y*self.y == self.z*self.z + 1  # Fixed z*z instead of z*z

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

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: Set[Solution] = set()
        self.families: Dict[str, Set[Solution]] = defaultdict(set)
        self.generation_times: Dict[str, float] = {}
        self.ratio_solutions: Dict[float, List[Solution]] = defaultdict(list)
        self.consecutive_patterns: List[Tuple[Solution, Solution, Tuple[int, int, int]]] = []

    def generate_all(self):
        start_time = time.time()
        print(f"\nGenerating solutions up to {self.limit}...")

        t0 = time.time()
        self._generate_solutions_optimized()
        self.generation_times['generation'] = time.time() - t0

        t0 = time.time()
        self._analyze_patterns()
        self.generation_times['analysis'] = time.time() - t0

        self.generation_times['total'] = time.time() - start_time

    def _generate_solutions_optimized(self):
        # Generate trivial solutions
        for n in range(1, self.limit + 1):
            self._try_add_solution(1, n, n, "trivial")
            if n > 1:
                self._try_add_solution(n, 1, n, "trivial")

        # Generate non-trivial solutions
        sqrt_limit = isqrt(self.limit)
        for x in range(2, self.limit):
            x_squared = x*x
            y_start = x + 1
            y_end = min(self.limit, isqrt(self.limit*self.limit - x_squared + 1))

            for y in range(y_start, y_end + 1):
                z_squared = x_squared + y*y - 1
                z = isqrt(z_squared)
                if z <= self.limit and z*z == z_squared:
                    self._try_add_solution(x, y, z, "non_trivial")

    def _try_add_solution(self, x: int, y: int, z: int, family: str):
        if max(x, y, z) <= self.limit and min(x, y) > 0:
            try:
                sol = Solution(x, y, z)
                self.solutions.add(sol)
                self.families[family].add(sol)

                # Track ratios
                for ratio in sol.ratios():
                    rounded = round(ratio, 3)
                    self.ratio_solutions[rounded].append(sol)

                if sol.is_primitive():
                    self.primitive_solutions.add(sol.normalized())

                if x != y:
                    sym_sol = Solution(y, x, z)
                    self.solutions.add(sym_sol)
                    self.families[family].add(sym_sol)
                    if sym_sol.is_primitive():
                        self.primitive_solutions.add(sym_sol.normalized())
            except AssertionError:
                pass

    def _analyze_patterns(self):
        primitives = sorted(list(self.primitive_solutions),
                          key=lambda s: (s.z, s.y, s.x))

        # Analyze consecutive solutions
        for i in range(len(primitives)-1):
            s1, s2 = primitives[i], primitives[i+1]
            diff = s1.difference_vector(s2)
            self.consecutive_patterns.append((s1, s2, diff))

    def analyze_dominant_ratios(self) -> Dict:
        ratio_analysis = {}
        # Only analyze non-trivial primitive solutions
        non_trivial_primitives = [s for s in self.primitive_solutions if not s.is_trivial()]

        for ratio, solutions in self.ratio_solutions.items():
            # Filter out trivial solutions
            solutions = [s for s in solutions if not s.is_trivial()]
            if len(solutions) > len(non_trivial_primitives) * 0.15:
                diffs = []
                for s1, s2 in combinations(solutions[:100], 2):
                    diffs.append(s1.difference_vector(s2))

                if diffs:
                    avg_diff = tuple(map(lambda x: sum(x)/len(x), zip(*diffs)))
                    ratio_analysis[ratio] = {
                        'count': len(solutions),
                        'percentage': len(solutions) / len(non_trivial_primitives) * 100,
                        'average_difference': avg_diff,
                        'example_solutions': sorted(solutions, key=lambda s: s.z)[:5]
                    }

        return ratio_analysis

    def analyze_consecutive_patterns(self) -> Dict:
        pattern_analysis = defaultdict(list)
        # Only analyze non-trivial primitive solutions
        primitives = sorted([s for s in self.primitive_solutions if not s.is_trivial()],
                           key=lambda s: (s.z, s.y, s.x))

        for i in range(len(primitives)-1):
            s1, s2 = primitives[i], primitives[i+1]
            diff = s1.difference_vector(s2)
            if all(d > 0 for d in diff):
                pattern_analysis['monotonic'].append((s1, s2, diff))
            elif sum(diff) == 0:
                pattern_analysis['conserved'].append((s1, s2, diff))
            else:
                pattern_analysis['mixed'].append((s1, s2, diff))

        return dict(pattern_analysis)

def analyze_large_scale(limits: List[int] = [1000, 10000, 100000]):
    for limit in limits:
        print(f"\n{'='*50}")
        print(f"Analyzing solutions up to N={limit}")

        t0 = time.time()
        analyzer = EnhancedDiophantineAnalyzer(limit)
        analyzer.generate_all()

        print(f"\nGeneration Times:")
        for method, time_taken in analyzer.generation_times.items():
            print(f"{method}: {time_taken:.2f} seconds")

        print(f"\nSolution Counts:")
        print(f"Total solutions: {len(analyzer.solutions)}")
        print(f"Primitive solutions: {len(analyzer.primitive_solutions)}")

        print("\nDominant Ratio Analysis:")
        ratio_analysis = analyzer.analyze_dominant_ratios()
        for ratio, data in sorted(ratio_analysis.items()):
            print(f"\nRatio {ratio:.3f}:")
            print(f"  Frequency: {data['count']} ({data['percentage']:.1f}%)")
            print(f"  Average difference vector: {data['average_difference']}")
            print("  Example solutions:")
            for sol in data['example_solutions']:
                print(f"    ({sol.x}, {sol.y}, {sol.z})")

        print("\nConsecutive Pattern Analysis:")
        pattern_analysis = analyzer.analyze_consecutive_patterns()
        for pattern, examples in pattern_analysis.items():
            print(f"\n{pattern.title()} Patterns ({len(examples)} instances):")
            for s1, s2, diff in examples[:3]:  # Show first 3 examples
                print(f"  ({s1.x},{s1.y},{s1.z}) -> ({s2.x},{s2.y},{s2.z})")
                print(f"  Difference vector: {diff}")

        print(f"\nTotal analysis time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
