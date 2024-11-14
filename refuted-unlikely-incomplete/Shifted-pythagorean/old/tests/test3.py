from dataclasses import dataclass
from typing import Set, List, Tuple, Dict
from math import gcd, sqrt, isqrt
from collections import defaultdict
import time

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

class DiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: Set[Solution] = set()
        self.families: Dict[str, Set[Solution]] = defaultdict(set)
        self.generation_times: Dict[str, float] = {}

    def generate_all(self):
        start_time = time.time()

        t0 = time.time()
        self._generate_trivial_solutions()
        self.generation_times['trivial'] = time.time() - t0

        t0 = time.time()
        self._generate_square_difference_solutions()
        self.generation_times['square_diff'] = time.time() - t0

        t0 = time.time()
        self._generate_square_progression_solutions()
        self.generation_times['square_prog'] = time.time() - t0

        t0 = time.time()
        self._generate_symmetric_solutions()
        self.generation_times['symmetric'] = time.time() - t0

        t0 = time.time()
        self._classify_solutions()
        self.generation_times['classification'] = time.time() - t0

        self.generation_times['total'] = time.time() - start_time

    def _generate_trivial_solutions(self):
        for n in range(1, self.limit + 1):
            self._try_add_solution(1, n, n, "trivial")
            if n > 1:
                self._try_add_solution(n, 1, n, "trivial")

    def _generate_square_difference_solutions(self):
        sqrt_limit = isqrt(self.limit) + 1
        for m in range(2, sqrt_limit):
            for n in range(1, m):
                if gcd(m, n) == 1:
                    # Basic form
                    x = m*m - n*n
                    y = 2*m*n
                    z = m*m + n*n - 1
                    if max(x, y, z) <= self.limit:
                        self._try_add_solution(x, y, z, "square_diff")

                    # Modified forms
                    x = abs(m*m - n*n - 1)
                    y = 2*m*n
                    z = m*m + n*n - 1
                    if max(x, y, z) <= self.limit:
                        self._try_add_solution(x, y, z, "square_diff")

                    x = m*m + n*n
                    y = abs(m*m - n*n - 1)
                    z = m*m + n*n - 1
                    if max(x, y, z) <= self.limit:
                        self._try_add_solution(x, y, z, "square_diff")

    def _generate_square_progression_solutions(self):
        # Use a more efficient approach for large numbers
        for x in range(2, self.limit):
            x_squared = x*x
            y_start = x + 1
            y_end = min(self.limit, isqrt(self.limit*self.limit - x_squared + 1))

            for y in range(y_start, y_end + 1):
                z_squared = x_squared + y*y - 1
                z = isqrt(z_squared)
                if z <= self.limit and z*z == z_squared:
                    self._try_add_solution(x, y, z, "square_prog")

    def _generate_symmetric_solutions(self):
        for n in range(2, self.limit):
            z = isqrt(2*n*n - 1)
            if z <= self.limit and z*z == 2*n*n - 1:
                self._try_add_solution(n, n, z, "symmetric")

    def _try_add_solution(self, x: int, y: int, z: int, family: str):
        if max(x, y, z) <= self.limit and min(x, y) > 0:
            try:
                sol = Solution(x, y, z)
                self.solutions.add(sol)
                self.families[family].add(sol)
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

    def _classify_solutions(self):
        for sol in self.primitive_solutions:
            if not sol.is_trivial():
                ratios = sol.ratios()
                min_ratio = min(ratios)
                max_ratio = max(ratios)

                if abs(min_ratio - 0.5) < 0.1:
                    self.families["half_ratio"].add(sol)
                elif abs(min_ratio - 0.618) < 0.1:
                    self.families["golden_ratio"].add(sol)
                elif abs(max_ratio - 1.414) < 0.1:
                    self.families["sqrt2"].add(sol)

    def verify_completeness(self) -> bool:
        print("\nVerifying completeness (this may take a while for large N)...")
        t0 = time.time()

        brute_force = set()
        sqrt_limit = isqrt(self.limit)

        for x in range(1, self.limit + 1):
            y_start = x  # Optimization: only need y ≥ x
            y_end = min(self.limit, isqrt(self.limit*self.limit - x*x + 1))

            for y in range(y_start, y_end + 1):
                z_squared = x*x + y*y - 1
                z = isqrt(z_squared)
                if z <= self.limit and z*z == z_squared:
                    brute_force.add(Solution(x, y, z).normalized())
                    if x != y:
                        brute_force.add(Solution(y, x, z).normalized())

        generated = {s.normalized() for s in self.solutions}
        missing = brute_force - generated

        verification_time = time.time() - t0
        print(f"Verification completed in {verification_time:.2f} seconds")

        if missing:
            print(f"\nFound {len(missing)} missing solutions!")
            for sol in sorted(list(missing)[:20]):  # Show first 20 if many missing
                print(f"({sol.x}, {sol.y}, {sol.z})")
                print(f"  Ratios: {[round(r,3) for r in sol.ratios()]}")

        return len(missing) == 0

    def analyze(self) -> Dict:
        non_trivial = {s for s in self.solutions if not s.is_trivial()}
        primitives = {s for s in self.primitive_solutions if not s.is_trivial()}

        ratios = defaultdict(int)
        for sol in primitives:
            for r in sol.ratios():
                ratios[round(r, 2)] += 1

        common_ratios = sorted([(r, c) for r, c in ratios.items() if c > 1],
                             key=lambda x: x[1], reverse=True)[:10]  # Show top 10 ratios

        family_stats = {
            name: len(sols) for name, sols in self.families.items()
            if len(sols) > 0
        }

        return {
            "total_solutions": len(self.solutions),
            "non_trivial": len(non_trivial),
            "primitive": len(primitives),
            "symmetric": len([s for s in primitives if s.is_symmetric()]),
            "common_ratios": common_ratios,
            "family_stats": family_stats,
            "generation_times": self.generation_times
        }

def analyze_solutions(limits: List[int] = [100, 1000, 5000, 10000]):
    base_stats = None

    for limit in limits:
        print(f"\n{'='*50}")
        print(f"Analyzing solutions up to N={limit}")

        t0 = time.time()
        analyzer = DiophantineAnalyzer(limit)
        analyzer.generate_all()
        stats = analyzer.analyze()
        total_time = time.time() - t0

        print(f"\nSolution Analysis (limit={limit}):")
        print(f"Total solutions found: {stats['total_solutions']}")
        print(f"Non-trivial solutions: {stats['non_trivial']}")
        print(f"Primitive solutions: {stats['primitive']}")
        print(f"Symmetric solutions: {stats['symmetric']}")

        print("\nGeneration Times:")
        for method, time_taken in stats['generation_times'].items():
            print(f"{method}: {time_taken:.2f} seconds")

        print("\nSolution Family Distribution:")
        total = sum(stats['family_stats'].values())
        for family, count in sorted(stats['family_stats'].items()):
            percentage = (count/total) * 100
            print(f"{family}: {count} solutions ({percentage:.1f}%)")

        print("\nCommon Ratios (ratio: frequency):")
        for ratio, count in stats['common_ratios']:
            percentage = (count/stats['primitive']) * 100  # Fixed this line
            print(f"{ratio:.2f}: {count} occurrences ({percentage:.1f}% of primitives)")

        if base_stats:
            n_ratio = limit / limits[0]
            expected_quadratic = n_ratio * n_ratio
            actual_ratio = stats['total_solutions'] / base_stats['total_solutions']
            print(f"\nGrowth Analysis (relative to N={limits[0]}):")
            print(f"N ratio: {n_ratio:.2f}x")
            print(f"Expected quadratic growth: {expected_quadratic:.2f}x")
            print(f"Actual growth: {actual_ratio:.2f}x")
            print(f"Growth factor accuracy: {(actual_ratio/expected_quadratic):.3f}")

        completeness = analyzer.verify_completeness()
        print(f"\nCompleteness verified: {completeness}")
        print(f"Total analysis time: {total_time:.2f} seconds")

        base_stats = stats

if __name__ == "__main__":
    analyze_solutions([100, 1000, 5000, 10000])
