import math
import logging
from typing import Set, List, Tuple, Dict, Callable
from collections import defaultdict
import numpy as np

# ----------------------- Setup Logging -----------------------

logging.basicConfig(filename='detailed_pattern_analysis.log', level=logging.INFO)

# ----------------------- Solution Class -----------------------

class Solution:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Solution(x={self.x}, y={self.y}, z={self.z})"

    def ratios(self) -> Tuple[float, float, float]:
        return (self.x / self.y, self.y / self.z, self.x / self.z)

    def normalized_ratios(self) -> Tuple[float, float, float]:
        """Normalize ratios for pattern classification."""
        return tuple(round(r, 4) for r in self.ratios())

# ----------------------- Pattern Detection -----------------------

class PatternType:
    Z_PRESERVING = 'z_preserving'
    RATIO_LOCKED = 'ratio_locked'
    COMPOSITE = 'composite'
    UNCLASSIFIED = 'unclassified'

THRESHOLDS = {
    PatternType.Z_PRESERVING: 0.01,
    PatternType.RATIO_LOCKED: 0.05,
    PatternType.COMPOSITE: 0.10
}

def classify_pattern(solution_ratios: List[Tuple[float, float, float]]) -> str:
    """Classify patterns based on solution ratios."""
    x_y_ratios, y_z_ratios, x_z_ratios = zip(*solution_ratios)

    # Check for Z-preserving pattern (constant y/z ratios)
    if np.std(y_z_ratios) < THRESHOLDS[PatternType.Z_PRESERVING]:
        return PatternType.Z_PRESERVING

    # Check for Ratio-locked pattern (constant x/y ratios)
    if np.std(x_y_ratios) < THRESHOLDS[PatternType.RATIO_LOCKED]:
        return PatternType.RATIO_LOCKED

    # Check for composite pattern (default if neither above is satisfied)
    if np.std(x_z_ratios) < THRESHOLDS[PatternType.COMPOSITE]:
        return PatternType.COMPOSITE

    # Return as unclassified if it doesn't fit any known pattern
    return PatternType.UNCLASSIFIED

# ----------------------- Enhanced Pattern Analysis -----------------------

def analyze_patterns(solutions: List[Solution]) -> Dict[str, List[Solution]]:
    """Detect and categorize patterns within the solution set."""
    patterns = defaultdict(list)

    for solution in solutions:
        ratios = solution.ratios()
        pattern_type = classify_pattern([ratios])
        patterns[pattern_type].append(solution)

    return dict(patterns)

def log_patterns(patterns: Dict[str, List[Solution]]):
    """Log the number of patterns found in each category."""
    for pattern_type, solution_list in patterns.items():
        logging.info(f"Pattern Type: {pattern_type}, Count: {len(solution_list)}")
        for sol in solution_list[:5]:  # Log a few examples for clarity
            logging.info(f"    Example: {sol}")

# ----------------------- Solution Generation -----------------------

def generate_solutions(limit: int) -> Set[Solution]:
    """Generate all solutions (x, y, z) such that x^2 + y^2 = z^2 + 1."""
    solutions = set()

    for x in range(2, limit):
        for y in range(x + 1, limit):
            z_squared = x * x + y * y - 1
            z = int(math.isqrt(z_squared))
            if z * z == z_squared and z < limit:
                sol = Solution(x, y, z)
                solutions.add(sol)

    return solutions

# ----------------------- Growth Analysis and Validation -----------------------

def fit_power_law(ns: List[int], counts: List[int]) -> Tuple[float, float]:
    """Fit a power law to growth data."""
    logs_ns = np.log(ns)
    logs_counts = np.log(counts)
    slope, intercept = np.polyfit(logs_ns, logs_counts, 1)
    return slope, intercept

def analyze_growth(solutions_by_n: Dict[int, Set[Solution]]) -> Dict[str, Tuple[float, float]]:
    """Analyze the growth of solutions and pattern types."""
    growth_rates = {}

    ns = sorted(solutions_by_n.keys())
    total_counts = [len(solutions_by_n[n]) for n in ns]

    # Analyze total solution growth
    growth_rates['total'] = fit_power_law(ns, total_counts)

    return growth_rates

def validate_growth(growth_data: Dict[str, Tuple[float, float]], expected_slope: float):
    """Validate growth rates based on power-law fit."""
    actual_slope = growth_data['total'][0]
    if abs(actual_slope - expected_slope) > 0.05:
        logging.warning(f"Growth rate deviation: Actual slope = {actual_slope}, Expected = {expected_slope}")
    else:
        logging.info(f"Growth rate matches expected slope: {actual_slope}")

# ----------------------- Testing Functions -----------------------

def run_tests():
    """Run a series of tests on the solution generation and pattern classification."""

    # Test small-scale limits to verify correctness of solution generation
    test_limits = [10, 100, 1000]
    for limit in test_limits:
        solutions = generate_solutions(limit)
        logging.info(f"Generated {len(solutions)} solutions for limit {limit}")

    # Verify pattern classification on known solutions
    solutions_1000 = generate_solutions(1000)
    patterns = analyze_patterns(list(solutions_1000))
    log_patterns(patterns)

    # Test growth validation
    solutions_by_n = {
        1000: generate_solutions(1000),
        10000: generate_solutions(10000),
        100000: generate_solutions(100000)
    }
    growth_data = analyze_growth(solutions_by_n)
    validate_growth(growth_data, expected_slope=0.5)  # Example expected slope

# ----------------------- Main Execution -----------------------

def main():
    """Main function to run the solution generation, pattern analysis, and growth validation."""
    logging.info("Starting comprehensive analysis")

    # Generate and analyze solutions for large-scale limits
    test_limits = [1000, 10000, 100000, 1000000]
    for limit in test_limits:
        solutions = generate_solutions(limit)
        patterns = analyze_patterns(list(solutions))

        # Log the number of solutions and patterns found
        logging.info(f"Limit: {limit}, Total Solutions: {len(solutions)}")
        log_patterns(patterns)

    # Validate growth for larger limits
    solutions_by_n = {
        1000: generate_solutions(1000),
        10000: generate_solutions(10000),
        100000: generate_solutions(100000)
    }
    growth_data = analyze_growth(solutions_by_n)
    validate_growth(growth_data, expected_slope=0.5)  # Adjust expected slope as necessary

if __name__ == "__main__":
    main()
    run_tests()
