from math import sqrt, isqrt
from typing import Set, Tuple, List, Dict, Optional, Generator
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from itertools import count

@dataclass
class Solution:
    x: int
    y: int
    z: int

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x/self.y, self.y/self.z, self.x/self.z)

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def verify(self) -> bool:
        return self.x*self.x + self.y*self.y == self.z*self.z + 1

    def is_primitive(self) -> bool:
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
        return gcd(self.x, gcd(self.y, self.z)) == 1

class InfiniteSolutionGenerator:
    """Generates solutions indefinitely using optimized search"""

    def __init__(self):
        self.known_max_families: Dict[int, Set[Solution]] = {}
        self.symmetry_scores: Dict[int, float] = {}

    def generate(self) -> Generator[Solution, None, None]:
        """Generate solutions indefinitely"""
        for z in count(2):
            x_limit = isqrt(z*z)
            for x in range(2, x_limit):
                y_squared = z*z + 1 - x*x
                if y_squared > 0:
                    y = isqrt(y_squared)
                    if y*y == y_squared and y > x:
                        sol = Solution(x, y, z)
                        if sol.verify():
                            yield sol

    def find_next_max_family(self, current_max: int = 23) -> Optional[Dict]:
        """Search for a family larger than current_max"""
        z_families = defaultdict(set)  # Fixed: properly initialize defaultdict

        try:
            for sol in self.generate():
                z_families[sol.z].add(sol)

                # Check if we've found a larger family
                family_size = len(z_families[sol.z])
                if family_size > current_max:
                    return {
                        'z': sol.z,
                        'size': family_size,
                        'solutions': z_families[sol.z],
                        'primitive_count': sum(1 for s in z_families[sol.z] if s.is_primitive())
                    }

                # Add a z-value limit to prevent infinite search
                if sol.z > 100000:  # Reasonable upper limit
                    return None

                # Periodic cleanup of small families
                if len(z_families) > 1000:
                    z_families = defaultdict(set, {
                        z: sols for z, sols in z_families.items()
                        if len(sols) >= current_max - 5
                    })

        except Exception as e:
            print(f"Search terminated due to: {e}")
            return None

class PerfectSquareAnalyzer:
    """Analyzes the perfect square constraint in detail"""

    def __init__(self, z: int):
        self.z = z
        self.z_squared_plus_1 = z*z + 1

    def count_perfect_squares(self) -> Dict:
        """Count and categorize perfect square opportunities"""
        results = {
            'total': 0,
            'primitive': 0,
            'derived': 0,
            'square_factors': set(),
            'x_values': []
        }

        for x in range(1, self.z):
            y_squared = self.z_squared_plus_1 - x*x
            if y_squared <= 0:
                break

            y = isqrt(y_squared)
            if y*y == y_squared:
                results['total'] += 1
                results['x_values'].append(x)

                # Analyze primitivity
                if Solution(x, y, self.z).is_primitive():
                    results['primitive'] += 1
                else:
                    results['derived'] += 1

                # Factor analysis
                factors = self._prime_factors(y_squared)
                results['square_factors'].update(factors)

        return results

    def _prime_factors(self, n: int) -> Set[int]:
        """Find prime factors that appear in pairs (square factors)"""
        factors = Counter()
        d = 2
        while n > 1:
            while n % d == 0:
                factors[d] += 1
                n //= d
            d += 1 if d == 2 else 2
        return {p for p, count in factors.items() if count >= 2}

    def theoretical_bound(self) -> int:
        """Calculate theoretical upper bound for solutions"""
        return 2 * isqrt(self.z)

    def analyze_constraints(self) -> Dict:
        """Comprehensive analysis of what limits solution count"""
        ps_count = self.count_perfect_squares()
        theo_bound = self.theoretical_bound()

        return {
            'theoretical_bound': theo_bound,
            'perfect_square_count': ps_count['total'],
            'primitive_count': ps_count['primitive'],
            'derived_count': ps_count['derived'],
            'square_factors': sorted(ps_count['square_factors']),
            'limiting_factor': 'Perfect Square' if ps_count['total'] < theo_bound else 'Theoretical Bound',
            'x_distribution': self._analyze_x_distribution(ps_count['x_values'])
        }

    def _analyze_x_distribution(self, x_values: List[int]) -> Dict:
        """Analyze distribution of x values that yield perfect squares"""
        if not x_values:
            return {}

        gaps = np.diff(x_values)
        return {
            'min_gap': int(min(gaps)) if len(gaps) > 0 else 0,
            'max_gap': int(max(gaps)) if len(gaps) > 0 else 0,
            'mean_gap': float(np.mean(gaps)) if len(gaps) > 0 else 0,
            'std_gap': float(np.std(gaps)) if len(gaps) > 0 else 0
        }

class SymmetryAnalyzer:
    """Advanced symmetry analysis with geometric insights"""

    def __init__(self, solutions: Set[Solution]):
        self.solutions = solutions
        self.z = next(iter(solutions)).z if solutions else 0
        self.normalized_points = self._normalize_points()

    def _normalize_points(self) -> List[Tuple[float, float]]:
        """Convert solutions to normalized (x/z, y/z) coordinates"""
        return [(s.x/s.z, s.y/s.z) for s in self.solutions]

    def analyze_symmetry(self) -> Dict:
        """Comprehensive symmetry analysis"""
        if len(self.solutions) < 2:
            return {'score': 0.0, 'properties': {}}

        center = self._compute_center()
        distances = self._compute_distances(center)
        angles = self._compute_angles(center)

        return {
            'score': self._calculate_symmetry_score(distances),
            'properties': {
                'center': center,
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'angular_uniformity': self._analyze_angular_distribution(angles),
                'radial_symmetry': self._analyze_radial_symmetry(distances),
                'reflection_axes': self._find_reflection_axes()
            }
        }

    def _compute_center(self) -> Tuple[float, float]:
        """Compute geometric center of normalized points"""
        x_mean = np.mean([x for x, y in self.normalized_points])
        y_mean = np.mean([y for x, y in self.normalized_points])
        return (x_mean, y_mean)

    def _compute_distances(self, center: Tuple[float, float]) -> np.ndarray:
        """Compute distances from center to all points"""
        return np.array([
            sqrt((x - center[0])**2 + (y - center[1])**2)
            for x, y in self.normalized_points
        ])

    def _compute_angles(self, center: Tuple[float, float]) -> np.ndarray:
        """Compute angles of all points relative to center"""
        return np.array([
            np.arctan2(y - center[1], x - center[0])
            for x, y in self.normalized_points
        ])

    def _calculate_symmetry_score(self, distances: np.ndarray) -> float:
        """Calculate symmetry score based on distance distribution"""
        if len(distances) < 2:
            return 0.0
        return float(np.std(distances) / np.mean(distances))

    def _analyze_angular_distribution(self, angles: np.ndarray) -> float:
        """Analyze uniformity of angular distribution"""
        if len(angles) < 2:
            return 0.0
        # Convert to unit circle and measure clustering
        points = np.exp(1j * angles)
        mean_vector = np.mean(points)
        return float(abs(mean_vector))  # 0 = uniform, 1 = clustered

    def _analyze_radial_symmetry(self, distances: np.ndarray) -> float:
        """Analyze radial symmetry based on distance distribution"""
        if len(distances) < 2:
            return 0.0
        return float(1 - np.std(distances) / np.mean(distances))

    def _find_reflection_axes(self) -> List[float]:
        """Find potential axes of reflection symmetry"""
        axes = []
        points = np.array(self.normalized_points)
        center = self._compute_center()

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # Check if line through points i,j could be reflection axis
                midpoint = (points[i] + points[j]) / 2
                direction = points[j] - points[i]
                angle = np.arctan2(direction[1], direction[0])

                # Count points with reflections
                reflected_count = 0
                for k in range(len(points)):
                    if k != i and k != j:
                        # Check if point has matching reflection
                        reflected = self._reflect_point(points[k], midpoint, angle)
                        if self._has_matching_point(reflected, points, tolerance=1e-6):
                            reflected_count += 1

                if reflected_count >= len(points) - 2:  # Allow for numerical error
                    axes.append(float(angle))

        return sorted(set(axes))  # Remove duplicates

    def _reflect_point(self, point: np.ndarray, axis_point: np.ndarray,
                      axis_angle: float) -> np.ndarray:
        """Reflect a point across an axis"""
        # Translate to origin
        translated = point - axis_point
        # Rotate to align axis with x-axis
        c, s = np.cos(-axis_angle), np.sin(-axis_angle)
        rotated = np.array([[c, -s], [s, c]]) @ translated
        # Reflect across x-axis
        reflected = np.array([rotated[0], -rotated[1]])
        # Rotate back
        c, s = np.cos(axis_angle), np.sin(axis_angle)
        rotated_back = np.array([[c, -s], [s, c]]) @ reflected
        # Translate back
        return rotated_back + axis_point

    def _has_matching_point(self, point: np.ndarray, points: np.ndarray,
                          tolerance: float) -> bool:
        """Check if point matches any in points within tolerance"""
        return any(np.linalg.norm(p - point) < tolerance for p in points)

def run_definitive_analysis(initial_limit: int = 10000,
                          search_beyond: bool = True) -> Dict:
    """Run comprehensive analysis to settle all open questions"""

    print(f"\nInitiating definitive analysis (initial limit: {initial_limit})...")

    # Phase 1: Analyze known solution space
    generator = InfiniteSolutionGenerator()
    solutions = set()
    z_families: Dict[int, Set[Solution]] = defaultdict(set)

    # Collect solutions up to initial limit
    for sol in generator.generate():
        if sol.z > initial_limit:
            break
        solutions.add(sol)
        z_families[sol.z].add(sol)

    # Find largest families
    max_size = max(len(family) for family in z_families.values())
    max_families = {z: family for z, family in z_families.items()
                   if len(family) == max_size}

    # Analyze perfect square constraint for max families
    ps_analysis = {z: PerfectSquareAnalyzer(z).analyze_constraints()
                  for z in max_families.keys()}

    # Analyze symmetry for all significant families
    symmetry_analysis = {
        z: SymmetryAnalyzer(family).analyze_symmetry()
        for z, family in z_families.items()
        if len(family) > max_size // 2
    }

    # Phase 2: Search beyond (if requested)
    beyond_search = None
    if search_beyond:
        print("\nSearching for larger families beyond initial limit...")
        beyond_search = generator.find_next_max_family(max_size)

    return {
        'initial_analysis': {
            'total_solutions': len(solutions),
            'unique_z_values': len(z_families),
            'max_family_size': max_size,
            'max_family_count': len(max_families),
            'max_z_values': sorted(max_families.keys())
        },
        'perfect_square_analysis': ps_analysis,
        'symmetry_analysis': symmetry_analysis,
        'beyond_search': beyond_search,
        'theoretical_implications': {
            'max_size_bound': min(
                max(a['theoretical_bound'] for a in ps_analysis.values()),
                max(a['perfect_square_count'] for a in ps_analysis.values())
            ),
            'symmetry_bound': min(
                data['score'] for data in symmetry_analysis.values()
            ) if symmetry_analysis else None,
            'limiting_factors': set(
                a['limiting_factor'] for a in ps_analysis.values()
            )
        }
    }

if __name__ == "__main__":
    results = run_definitive_analysis(10000, search_beyond=True)

    print("\n=== DEFINITIVE ANALYSIS RESULTS ===\n")

    print("Initial Analysis:")
    for key, value in results['initial_analysis'].items():
        print(f"{key}: {value}")

    print("\nPerfect Square Analysis:")
    for z, analysis in results['perfect_square_analysis'].items():
        print(f"\nZ = {z}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    print("\nSymmetry Analysis (top 3 most symmetric):")
    sorted_symmetry = sorted(
        results['symmetry_analysis'].items(),
        key=lambda x: x[1]['score']
    )[:3]
    for z, analysis in sorted_symmetry:
        print(f"\nZ = {z}:")
        print(f"  Score: {analysis['score']:.3f}")
        print(f"  Properties: {analysis['properties']}")

    if results['beyond_search']:
        print("\nBeyond Search Results:")
        print(f"Found larger family: Z = {results['beyond_search']['z']}")
        print(f"Size: {results['beyond_search']['size']}")
        print(f"Primitive solutions: {results['beyond_search']['primitive_count']}")

    print("\nTheoretical Implications:")
    for key, value in results['theoretical_implications'].items():
        print(f"{key}: {value}")
