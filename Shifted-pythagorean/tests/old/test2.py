from dataclasses import dataclass
from typing import Set, List, Tuple, Dict
from math import gcd, sqrt, isqrt
from collections import defaultdict

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

    def generate_all(self):
        self._generate_trivial_solutions()
        self._generate_square_difference_solutions()
        self._generate_symmetric_solutions()
        self._generate_derived_solutions()
        self._classify_solutions()

    def _generate_trivial_solutions(self):
        """Generate solutions where x=1 or y=1"""
        for n in range(1, self.limit + 1):
            self._try_add_solution(1, n, n, "trivial")
            if n > 1:
                self._try_add_solution(n, 1, n, "trivial")

    def _generate_square_difference_solutions(self):
        """Generate solutions based on square differences"""
        # Method 1: Direct square difference exploration
        for x in range(2, self.limit):
            for d in range(1, (self.limit-x)//2):
                y = x + d
                if y > self.limit:
                    break
                z_squared = x*x + y*y - 1
                if z_squared > 0:
                    z = isqrt(z_squared)
                    if z <= self.limit and z*z == z_squared:
                        self._try_add_solution(x, y, z, "square_diff")

        # Method 2: Parametric exploration
        sqrt_limit = isqrt(self.limit) + 1
        for m in range(2, sqrt_limit):
            for n in range(1, m):
                if gcd(m, n) == 1:
                    # Basic form
                    x = m*m - n*n
                    y = 2*m*n
                    z = m*m + n*n - 1
                    if max(x, y, z) <= self.limit:
                        self._try_add_solution(x, y, z, "parametric")

                    # Modified form
                    x = abs(m*m - n*n - 1)
                    y = 2*m*n
                    z = m*m + n*n - 1
                    if max(x, y, z) <= self.limit:
                        self._try_add_solution(x, y, z, "parametric")

        # Method 3: Square progression exploration
        for x in range(2, self.limit):
            x_squared = x*x
            for y in range(x+1, self.limit):
                y_squared = y*y
                z = isqrt(x_squared + y_squared - 1)
                if z <= self.limit and z*z == x_squared + y_squared - 1:
                    self._try_add_solution(x, y, z, "square_prog")

    def _generate_symmetric_solutions(self):
        """Generate solutions where x=y"""
        for n in range(2, self.limit):
            z = isqrt(2*n*n - 1)
            if z <= self.limit and z*z == 2*n*n - 1:
                self._try_add_solution(n, n, z, "symmetric")

    def _generate_derived_solutions(self):
        """Generate non-primitive solutions from primitive ones"""
        primitives = set(self.primitive_solutions)
        for sol in primitives:
            for k in range(2, self.limit // max(sol.x, sol.y, sol.z) + 1):
                self._try_add_solution(k*sol.x, k*sol.y, k*sol.z, "derived")

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
        """Classify solutions into mathematical families"""
        for sol in self.primitive_solutions:
            if not sol.is_trivial():
                ratios = sol.ratios()
                min_ratio = min(ratios)
                max_ratio = max(ratios)

                # Classify by ratio patterns
                if abs(min_ratio - 0.5) < 0.1:
                    self.families["half_ratio"].add(sol)
                elif abs(min_ratio - 0.618) < 0.1:
                    self.families["golden_ratio"].add(sol)

                # Classify by properties
                if sol.is_symmetric():
                    self.families["symmetric"].add(sol)
                if abs(sol.z - sqrt(2)*max(sol.x, sol.y)) < 1:
                    self.families["sqrt2"].add(sol)

    def verify_completeness(self) -> bool:
        """Verify that we've found all solutions"""
        brute_force = set()
        for x in range(1, self.limit + 1):
            for y in range(x, self.limit + 1):
                z_squared = x*x + y*y - 1
                if z_squared > 0:
                    z = isqrt(z_squared)
                    if z <= self.limit and z*z == z_squared:
                        brute_force.add(Solution(x, y, z).normalized())
                        if x != y:
                            brute_force.add(Solution(y, x, z).normalized())

        generated = {s.normalized() for s in self.solutions}
        missing = brute_force - generated

        if missing:
            print("\nMissing Solutions:")
            for sol in sorted(missing, key=lambda s: (s.x, s.y))[:20]:
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
                             key=lambda x: x[1], reverse=True)[:5]

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
            "family_stats": family_stats
        }

def analyze_solutions(limits: List[int] = [100, 200, 500, 1000]):
    base_stats = None

    for limit in limits:
        analyzer = DiophantineAnalyzer(limit)
        analyzer.generate_all()
        stats = analyzer.analyze()

        print(f"\n{'='*50}")
        print(f"Solution Analysis (limit={limit}):")
        print(f"Total solutions found: {stats['total_solutions']}")
        print(f"Non-trivial solutions: {stats['non_trivial']}")
        print(f"Primitive solutions: {stats['primitive']}")
        print(f"Symmetric solutions: {stats['symmetric']}")

        print("\nCommon Ratios (ratio: frequency):")
        for ratio, count in stats['common_ratios']:
            print(f"{ratio:.2f}: {count}")

        print("\nSolution Families:")
        for family, count in sorted(stats['family_stats'].items()):
            print(f"{family}: {count} solutions")

        if base_stats is None:
            base_stats = stats
        else:
            growth_rate = stats['total_solutions'] / base_stats['total_solutions']
            primitive_growth = stats['primitive'] / base_stats['primitive']
            print(f"\nGrowth Analysis (relative to N={limits[0]}):")
            print(f"Total solution growth rate: {growth_rate:.2f}x")
            print(f"Primitive solution growth rate: {primitive_growth:.2f}x")

        print(f"\nCompleteness verified: {analyzer.verify_completeness()}")

if __name__ == "__main__":
    analyze_solutions([100, 200, 500, 1000])
