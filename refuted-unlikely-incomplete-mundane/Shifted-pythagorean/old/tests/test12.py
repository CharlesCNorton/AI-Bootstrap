from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, NamedTuple
from math import gcd, sqrt, isqrt, log, ceil, pi, e
from collections import defaultdict
import time
import numpy as np
from itertools import combinations, chain, product
from enum import Enum, auto
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

class FundamentalPattern(Enum):
    GOLDEN = auto()  # φ-based ratios
    SQRT2 = auto()   # √2-based ratios
    CUBIC = auto()   # cubic residue patterns
    PRIME = auto()   # prime-related ratios
    HYBRID = auto()  # mixed fundamental ratios

class AttractorType(Enum):
    PRIMARY = auto()    # (0.75, 0.8, 0.6) family
    SECONDARY = auto()  # (1.3333..., 0.6, 0.8) family
    GOLDEN = auto()     # φ-based attractors
    SYMMETRIC = auto()  # Equal ratio attractors
    COMPOSITE = auto()  # Mixed type attractors

@dataclass(frozen=True)
class Solution:
    x: int
    y: int
    z: int

    def __post_init__(self):
        assert self.x*self.x + self.y*self.y == self.z*self.z + 1

    def is_trivial(self) -> bool:
        return self.x == 1 or self.y == 1

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def normalized(self) -> 'Solution':
        return Solution(min(self.x, self.y), max(self.x, self.y), self.z)

    def is_primitive(self) -> bool:
        return gcd(gcd(self.x, self.y), self.z) == 1

    def fundamental_type(self) -> Optional[FundamentalPattern]:
        ratios = self.ratios()
        phi = (1 + sqrt(5))/2

        # Check for golden ratio patterns
        if any(abs(r - 1/phi) < 0.01 for r in ratios):
            return FundamentalPattern.GOLDEN

        # Check for √2 patterns
        if any(abs(r - 1/sqrt(2)) < 0.01 for r in ratios):
            return FundamentalPattern.SQRT2

        # Check for cubic patterns
        if any(abs(r - x/y) < 0.01
              for x, y in [(2,3), (3,4), (4,5)]
              for r in ratios):
            return FundamentalPattern.CUBIC

        # Check for prime-related patterns
        if any(abs(r - p/q) < 0.01
              for p, q in [(2,3), (3,5), (5,7)]
              for r in ratios):
            return FundamentalPattern.PRIME

        # Check for hybrid patterns
        if len(set(round(r, 2) for r in ratios)) == 3:
            return FundamentalPattern.HYBRID

        return None

@dataclass
class AttractorBasin:
    center: Tuple[float, float, float]
    solutions: Set[Solution]
    attractor_type: AttractorType
    stability: float = field(init=False)
    radius: float = field(init=False)

    def __post_init__(self):
        self.stability = self._calculate_stability()
        self.radius = self._calculate_radius()

    def _calculate_stability(self) -> float:
        if not self.solutions:
            return 0.0
        ratios = np.array([s.ratios() for s in self.solutions])
        return 1 - np.std(ratios, axis=0).mean()

    def _calculate_radius(self) -> float:
        if not self.solutions:
            return 0.0
        ratios = np.array([s.ratios() for s in self.solutions])
        center = np.array(self.center)
        distances = np.linalg.norm(ratios - center, axis=1)
        return np.max(distances)

class FundamentalStructureAnalyzer:
    def __init__(self):
        self.fundamental_patterns: Dict[FundamentalPattern, List[Solution]] = \
            defaultdict(list)
        self.attractor_basins: List[AttractorBasin] = []
        self.z_distribution: Dict[int, List[Solution]] = defaultdict(list)
        self.ratio_sequences: Dict[Tuple[float, float, float], List[Solution]] = \
            defaultdict(list)

    def analyze_structure(self, solutions: List[Solution]) -> Dict:
        print(f"Analyzing fundamental structure of {len(solutions)} solutions...")

        self._classify_fundamental_patterns(solutions)
        self._find_attractor_basins(solutions)
        self._analyze_z_distribution(solutions)
        self._find_ratio_sequences(solutions)

        return self._compile_analysis()

    def _classify_fundamental_patterns(self, solutions: List[Solution]) -> None:
        for sol in solutions:
            pattern_type = sol.fundamental_type()
            if pattern_type:
                self.fundamental_patterns[pattern_type].append(sol)

    def _find_attractor_basins(self, solutions: List[Solution]) -> None:
        # Group solutions by similar ratios
        ratio_groups = defaultdict(set)
        for sol in solutions:
            rounded = tuple(round(r, 3) for r in sol.ratios())
            ratio_groups[rounded].add(sol)

        # Identify attractor basins
        for center, sols in ratio_groups.items():
            if len(sols) < 10:
                continue

            # Determine attractor type
            if abs(center[0] - 0.75) < 0.01 and abs(center[1] - 0.8) < 0.01:
                attractor_type = AttractorType.PRIMARY
            elif abs(center[0] - 1.3333) < 0.01 and abs(center[2] - 0.8) < 0.01:
                attractor_type = AttractorType.SECONDARY
            elif any(abs(r - 0.618) < 0.01 for r in center):
                attractor_type = AttractorType.GOLDEN
            elif len(set(round(r, 2) for r in center)) == 1:
                attractor_type = AttractorType.SYMMETRIC
            else:
                attractor_type = AttractorType.COMPOSITE

            self.attractor_basins.append(
                AttractorBasin(center, sols, attractor_type)
            )

    def _analyze_z_distribution(self, solutions: List[Solution]) -> None:
        for sol in solutions:
            self.z_distribution[sol.z].append(sol)

    def _find_ratio_sequences(self, solutions: List[Solution]) -> None:
        # Group solutions by similar ratio progressions
        for sol in solutions:
            ratios = tuple(round(r, 3) for r in sol.ratios())
            self.ratio_sequences[ratios].append(sol)

    def _compile_analysis(self) -> Dict:
        # Analyze fundamental patterns
        pattern_stats = {
            pattern: {
                'count': len(sols),
                'ratio_distribution': self._analyze_ratio_distribution(sols),
                'z_distribution': self._analyze_z_distribution(sols)
            }
            for pattern, sols in self.fundamental_patterns.items()
        }

        # Analyze attractor basins
        basin_stats = {
            'count_by_type': defaultdict(int),
            'avg_stability': defaultdict(float),
            'avg_radius': defaultdict(float),
            'size_distribution': defaultdict(list)
        }

        for basin in self.attractor_basins:
            basin_stats['count_by_type'][basin.attractor_type] += 1
            basin_stats['avg_stability'][basin.attractor_type] += basin.stability
            basin_stats['avg_radius'][basin.attractor_type] += basin.radius
            basin_stats['size_distribution'][basin.attractor_type].append(len(basin.solutions))

        for attractor_type in AttractorType:
            count = basin_stats['count_by_type'][attractor_type]
            if count > 0:
                basin_stats['avg_stability'][attractor_type] /= count
                basin_stats['avg_radius'][attractor_type] /= count

        # Analyze z-value distribution
        z_stats = self._fit_z_distribution()

        # Analyze ratio sequences
        sequence_stats = self._analyze_ratio_sequences()

        return {
            'fundamental_patterns': pattern_stats,
            'attractor_basins': basin_stats,
            'z_distribution': z_stats,
            'ratio_sequences': sequence_stats
        }

    def _analyze_ratio_distribution(self, solutions: List[Solution]) -> Dict:
        if not solutions:
            return {}

        ratios = np.array([s.ratios() for s in solutions])
        return {
            'mean': ratios.mean(axis=0).tolist(),
            'std': ratios.std(axis=0).tolist(),
            'skew': [float(stats.skew(r)) for r in ratios.T]  # Using scipy.stats.skew instead
        }

    def _fit_z_distribution(self) -> Dict:
        sizes = [len(sols) for sols in self.z_distribution.values()]
        if not sizes:
            return {}

        # Fit power law using numpy
        sizes = np.array(sizes)
        log_sizes = np.log(sizes[sizes > 0])

        # Estimate alpha using maximum likelihood
        alpha = 1 + len(log_sizes) / sum(log_sizes - log_sizes.min())

        # Compute KS statistic
        theoretical_cdf = lambda x: 1 - (x/min(sizes))**(-alpha + 1)
        D, _ = stats.kstest(sizes, theoretical_cdf)

        return {
            'alpha': alpha,
            'xmin': min(sizes),
            'D': D,
            'distribution': np.histogram(sizes, bins='auto')[0].tolist()
        }

    def _analyze_ratio_sequences(self) -> Dict:
        if not self.ratio_sequences:
            return {}

        sequence_lengths = [len(sols) for sols in self.ratio_sequences.values()]

        return {
            'unique_sequences': len(self.ratio_sequences),
            'max_length': max(sequence_lengths),
            'avg_length': np.mean(sequence_lengths),
            'length_distribution': np.histogram(sequence_lengths, bins='auto')[0].tolist()
        }

class ComprehensiveDiophantineAnalyzer:
    def __init__(self, limit: int):
        self.limit = limit
        self.solutions: Set[Solution] = set()
        self.primitive_solutions: List[Solution] = []
        self.structure_analyzer = FundamentalStructureAnalyzer()
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
                    if sym_sol.is_primitive() and not sol.is_trivial():
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
            'fundamental_structure': self.structure_analysis,
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

        struct = analysis['fundamental_structure']

        print("\nFundamental Pattern Distribution:")
        for pattern, stats in struct['fundamental_patterns'].items():
            print(f"\n{pattern.name}:")
            print(f"Count: {stats['count']}")
            if 'ratio_distribution' in stats:
                print("Ratio Distribution:")
                print(f"  Mean: {[round(x,3) for x in stats['ratio_distribution']['mean']]}")
                print(f"  Std: {[round(x,3) for x in stats['ratio_distribution']['std']]}")
                print(f"  Power Law α: {[round(x,3) for x in stats['ratio_distribution']['skew']]}")

        print("\nAttractor Basin Analysis:")
        basin_stats = struct['attractor_basins']
        for attractor_type in AttractorType:
            count = basin_stats['count_by_type'][attractor_type]
            if count > 0:
                print(f"\n{attractor_type.name}:")
                print(f"Count: {count}")
                print(f"Average Stability: {basin_stats['avg_stability'][attractor_type]:.3f}")
                print(f"Average Radius: {basin_stats['avg_radius'][attractor_type]:.3f}")
                print(f"Size Distribution: {basin_stats['size_distribution'][attractor_type]}")

        print("\nZ-Value Distribution Analysis:")
        z_stats = struct['z_distribution']
        if z_stats:
            print(f"Power Law α: {z_stats['alpha']:.3f}")
            print(f"Minimum x: {z_stats['xmin']}")
            print(f"KS Distance: {z_stats['D']:.3f}")
            print(f"Distribution: {z_stats['distribution']}")

        print("\nRatio Sequence Analysis:")
        seq_stats = struct['ratio_sequences']
        if seq_stats:
            print(f"Unique Sequences: {seq_stats['unique_sequences']}")
            print(f"Max Length: {seq_stats['max_length']}")
            print(f"Average Length: {seq_stats['avg_length']:.2f}")
            print(f"Length Distribution: {seq_stats['length_distribution']}")

        print(f"\nTotal analysis time: {analysis['timing']['total']:.2f} seconds")

if __name__ == "__main__":
    analyze_large_scale([1000, 10000, 100000])
