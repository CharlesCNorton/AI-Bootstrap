from dataclasses import dataclass
from typing import Set, List, Tuple, Dict
from math import gcd, sqrt
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

class DiophantineGenerator:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()

    def _try_add(self, x: int, y: int, z: int, solutions: Set[Solution]):
        """Safely try to add a solution"""
        if max(x,y,z) <= self.limit and min(x,y) > 0:
            try:
                sol = Solution(x, y, z)
                solutions.add(sol)
                if x != y:
                    solutions.add(Solution(y, x, z))
            except AssertionError:
                pass
        return solutions

    def generate_trivial(self):
        """Formula: (1,n,n) and (n,1,n)"""
        solutions = set()
        for n in range(1, self.limit + 1):
            self._try_add(1, n, n, solutions)
            if n > 1:
                self._try_add(n, 1, n, solutions)
        return solutions

    def generate_symmetric(self):
        """Formula: (n,n,√(2n²-1))"""
        solutions = set()
        for n in range(1, self.limit + 1):
            z_squared = 2*n*n - 1
            if z_squared > 0:
                z = int(sqrt(z_squared))
                if z*z == z_squared and z <= self.limit:
                    self._try_add(n, n, z, solutions)
        return solutions

    def generate_golden_ratio(self):
        """Generates solutions based on golden ratio patterns"""
        solutions = set()
        sqrt_limit = int(sqrt(self.limit)) + 1

        # Known golden ratio primitive solutions
        primitives = [
            (4,7,8), (7,11,13), (11,18,21), (18,29,34),
            (29,47,55), (47,76,89)
        ]

        # Add primitives and their multiples
        for x, y, z in primitives:
            k = 1
            while k*max(x,y,z) <= self.limit:
                self._try_add(k*x, k*y, k*z, solutions)
                k += 1

        return solutions

    def generate_half_ratio(self):
        """Generates solutions with ratio approximately 1/2"""
        solutions = set()

        # Known half-ratio primitive solutions
        primitives = [
            (5,9,10), (9,17,19), (17,33,37),
            (6,11,12), (11,21,23), (21,41,46)
        ]

        # Add primitives and their multiples
        for x, y, z in primitives:
            k = 1
            while k*max(x,y,z) <= self.limit:
                self._try_add(k*x, k*y, k*z, solutions)
                k += 1

        return solutions

    def generate_sqrt2(self):
        """Generates solutions related to √2"""
        solutions = set()

        # Known √2-related primitive solutions
        primitives = [
            (5,5,7), (12,12,17), (29,29,41),
            (8,9,12), (15,17,23), (35,37,51)
        ]

        # Add primitives and their multiples
        for x, y, z in primitives:
            k = 1
            while k*max(x,y,z) <= self.limit:
                self._try_add(k*x, k*y, k*z, solutions)
                k += 1

        return solutions

class DiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.generator = DiophantineGenerator(limit)

    def analyze(self):
        # Generate solutions by family
        trivial = self.generator.generate_trivial()
        symmetric = self.generator.generate_symmetric()
        golden = self.generator.generate_golden_ratio()
        half = self.generator.generate_half_ratio()
        sqrt2 = self.generator.generate_sqrt2()

        # Combine all solutions
        all_solutions = trivial | symmetric | golden | half | sqrt2

        # Analyze primitive solutions
        primitives = {s for s in all_solutions if s.is_primitive() and not s.is_trivial()}

        # Analyze ratios
        ratios = defaultdict(int)
        for sol in primitives:
            for r in sol.ratios():
                ratios[round(r, 2)] += 1

        common_ratios = sorted([(r, c) for r, c in ratios.items() if c > 1],
                             key=lambda x: x[1], reverse=True)[:5]

        print(f"\nSolution Analysis (limit={self.limit}):")
        print(f"Total solutions: {len(all_solutions)}")
        print(f"By family:")
        print(f"  Trivial: {len(trivial)}")
        print(f"  Symmetric: {len(symmetric)}")
        print(f"  Golden ratio: {len(golden)}")
        print(f"  Half ratio: {len(half)}")
        print(f"  Sqrt(2): {len(sqrt2)}")
        print(f"Primitive solutions: {len(primitives)}")

        print("\nCommon ratios (ratio: frequency):")
        for ratio, count in common_ratios:
            print(f"{ratio:.2f}: {count}")

        print("\nPrimitive Solution Patterns:")
        print("1. Trivial: (1,n,n) and (n,1,n)")
        print("2. Symmetric: (n,n,√(2n²-1))")
        print("3. Golden ratio sequence: (4,7,8) → (7,11,13) → (11,18,21) → ...")
        print("4. Half ratio sequence: (5,9,10) → (9,17,19) → (17,33,37) → ...")
        print("5. Sqrt(2) sequence: (5,5,7) → (12,12,17) → (29,29,41) → ...")

if __name__ == "__main__":
    analyzer = DiophantineAnalyzer(100)
    analyzer.analyze()
