from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, Generator
from math import gcd, sqrt, isqrt, log, ceil
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain
from enum import Enum
from scipy.stats import linregress

class PatternType(Enum):
    MONOTONIC = "monotonic"
    MIXED = "mixed"
    CONSERVED = "conserved"
    SCALING = "scaling"
    SYMMETRIC = "symmetric"
    Z_PRESERVING = "z_preserving"
    SELF_SIMILAR = "self_similar"
    RATIO_LOCKED = "ratio_locked"

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
    def fractal_dimension(self) -> float:
        if len(self.solutions) < 2:
            return 0.0

        points = np.array([s.vector_form() for s in self.solutions])
        max_points = 1000

        if len(points) > max_points:
            indices = np.linspace(0, len(points)-1, max_points, dtype=int)
            points = points[indices]

        scales = []
        counts = []

        min_coord = points.min()
        max_coord = points.max()

        for k in range(1, 5):
            scale = (max_coord - min_coord) / (2**k)
            if scale == 0:
                continue
            scaled_points = points / scale
            unique_boxes = set(tuple(map(int, p)) for p in scaled_points)

            scales.append(np.log(1/scale))
            counts.append(np.log(len(unique_boxes)))

        if len(scales) < 2:
            return 0.0

        slope, _, _, _, _ = linregress(scales, counts)
        return abs(slope)

    @property
    def self_similarity_score(self) -> float:
        if len(self.solutions) < 3:
            return 0.0
        ratios = [s.ratios() for s in self.solutions]
        diffs = [tuple(abs(r2[i] - r1[i]) for i in range(3))
                for r1, r2 in zip(ratios, ratios[1:])]
        return 1.0 - np.std([sum(d) for d in diffs])

class FractalAnalyzer:
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.ratio_attractors: Dict[Tuple[float, float, float], int] = defaultdict(int)
        self.z_families: Dict[int, int] = defaultdict(int)
        self.scale_families: Dict[float, List[Pattern]] = defaultdict(list)

    def analyze_fractal_structure(self, solutions: List[Solution]) -> Dict:
        print(f"Analyzing {len(solutions)} solutions...")

        self._find_patterns(solutions)
        self._identify_attractors(solutions)
        self._classify_z_families(solutions)
        self._analyze_scaling_laws()
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
                        self.patterns.append(Pattern(current_pattern.copy(), current_type))
                    current_pattern = [s2]
                    current_type = None

            if len(current_pattern) >= 3:
                self.patterns.append(Pattern(current_pattern, current_type))

    def _determine_pattern_type(self, s1: Solution, s2: Solution) -> PatternType:
        v1, v2 = s1.vector_form(), s2.vector_form()
        diff = v2 - v1

        if all(d >= 0 for d in diff):
            return PatternType.MONOTONIC
        elif abs(sum(diff)) < 1e-10:
            return PatternType.CONSERVED
        elif s1.z == s2.z:
            return PatternType.Z_PRESERVING
        elif abs(s1.ratios()[0] - s2.ratios()[0]) < 0.01:
            return PatternType.RATIO_LOCKED
        else:
            return PatternType.MIXED

    def _identify_attractors(self, solutions: List[Solution]) -> None:
        for sol in solutions:
            ratios = tuple(round(r, 3) for r in sol.ratios())
            self.ratio_attractors[ratios] += 1

    def _classify_z_families(self, solutions: List[Solution]) -> None:
        for sol in solutions:
            self.z_families[sol.z] += 1

    def _analyze_scaling_laws(self) -> None:
        for pattern in self.patterns:
            if not pattern.scale_factors:
                continue
            avg_scale = np.mean(pattern.scale_factors)
            rounded_scale = round(avg_scale, 3)
            self.scale_families[rounded_scale].append(pattern)

    def _compile_analysis(self) -> Dict:
        fractal_dims = defaultdict(list)
        for pattern in self.patterns:
            fractal_dims[pattern.pattern_type].append(pattern.fractal_dimension)

        dominant_attractors = sorted(
            self.ratio_attractors.items(),
            key=lambda x: x[1], reverse=True
        )[:10]

        z_values = list(self.z_families.values())

        scale_distribution = {
            scale: len(patterns)
            for scale, patterns in self.scale_families.items()
        }

        return {
            'pattern_counts': {
                ptype: len([p for p in self.patterns if p.pattern_type == ptype])
                for ptype in PatternType
            },
            'fractal_dimensions': {
                ptype: np.mean(dims) if dims else 0
                for ptype, dims in fractal_dims.items()
            },
            'ratio_attractors': dominant_attractors,
            'z_family_statistics': {
                'unique_families': len(self.z_families),
                'max_family_size': max(z_values) if z_values else 0,
                'avg_family_size': np.mean(z_values) if z_values else 0,
                'family_size_std': np.std(z_values) if z_values else 0
            },
            'scaling_laws': {
                'unique_scales': len(scale_distribution),
                'dominant_scales': sorted(
                    scale_distribution.items(),
                    key=lambda x: x[1], reverse=True
                )[:5],
                'scale_distribution': scale_distribution
            },
            'self_similarity': {
                'average_score': np.mean([p.self_similarity_score for p in self.patterns]) if self.patterns else 0,
                'max_score': max(p.self_similarity_score for p in self.patterns) if self.patterns else 0,
                'score_distribution': np.histogram(
                    [p.self_similarity_score for p in self.patterns],
                    bins=10
                )[0].tolist() if self.patterns else []
            }
        }

class EnhancedDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.fractal_analyzer = FractalAnalyzer()
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
        self.fractal_analysis = self.fractal_analyzer.analyze_fractal_structure(sorted_primitives)

    def get_analysis(self) -> Dict:
        return {
            'solution_counts': {
                'total': len(self.solutions),
                'primitive': len(self.primitive_solutions)
            },
            'fractal_analysis': self.fractal_analysis,
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

        print("\nFractal Structure Analysis:")
        fractal = analysis['fractal_analysis']

        print("\nPattern Type Distribution:")
        for ptype, count in fractal['pattern_counts'].items():
            print(f"{ptype.value}: {count}")

        print("\nFractal Dimensions:")
        for ptype, dim in fractal['fractal_dimensions'].items():
            if dim > 0:
                print(f"{ptype.value}: {dim:.3f}")

        print("\nDominant Ratio Attractors:")
        for ratios, count in fractal['ratio_attractors']:
            print(f"{ratios}: {count} solutions")

        print("\nZ-Family Statistics:")
        for stat, value in fractal['z_family_statistics'].items():
            print(f"{stat}: {value}")

        print("\nScaling Laws:")
        print("Dominant scales:")
        for scale, count in fractal['scaling_laws']['dominant_scales']:
            print(f"scale={scale:.3f}: {count} patterns")

        print("\nSelf-Similarity Analysis:")
        for metric, value in fractal['self_similarity'].items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.3f}")

        print(f"\nTotal analysis time: {analysis['generation_times']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
