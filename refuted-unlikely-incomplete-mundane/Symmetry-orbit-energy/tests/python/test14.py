import numpy as np
from scipy.integrate import nquad
from scipy.stats import sem, ttest_ind
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, NamedTuple
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor
import time
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LieGroupType(Enum):
    CLASSICAL = "classical"
    EXCEPTIONAL = "exceptional"

@dataclass
class LieGroup:
    name: str
    rank: int
    dimension: int
    type: LieGroupType
    root_system: str

    def __post_init__(self):
        self.validate()
        self.weyl_order = self.get_weyl_group_order()
        self.root_complexity = self.compute_root_system_complexity()

    def validate(self):
        """Validate group parameters match known values"""
        if self.type == LieGroupType.EXCEPTIONAL:
            expected_dims = {
                "G2": 14,
                "F4": 52,
                "E6": 78,
                "E7": 133,
                "E8": 248
            }
            if self.name in expected_dims:
                assert self.dimension == expected_dims[self.name], f"Dimension mismatch for {self.name}"

    def get_weyl_group_order(self) -> int:
        """Calculate order of Weyl group"""
        weyl_orders = {
            "A": lambda n: np.math.factorial(n + 1),
            "B": lambda n: 2**n * np.math.factorial(n),
            "C": lambda n: 2**n * np.math.factorial(n),
            "D": lambda n: 2**(n-1) * np.math.factorial(n),
            "G2": 12,
            "F4": 1152,
            "E6": 51840,
            "E7": 2903040,
            "E8": 696729600
        }
        if self.type == LieGroupType.EXCEPTIONAL:
            return weyl_orders[self.name]
        return weyl_orders[self.root_system](self.rank)

    def compute_root_system_complexity(self) -> int:
        """Calculate number of positive roots"""
        positive_roots = {
            "A": lambda n: n*(n+1)//2,
            "B": lambda n: n**2,
            "C": lambda n: n**2,
            "D": lambda n: n*(n-1),
            "G2": 6,
            "F4": 24,
            "E6": 36,
            "E7": 63,
            "E8": 120
        }
        if self.type == LieGroupType.EXCEPTIONAL:
            return positive_roots[self.name]
        return positive_roots[self.root_system](self.rank)

class GroupMetrics(NamedTuple):
    entropy: float
    entropy_error: float
    normalized_entropy: float
    complexity_metric: float
    weyl_normalized_entropy: float

def create_lie_group_catalog() -> Dict[str, LieGroup]:
    """Create catalog of all relevant Lie groups with exact parameters"""
    catalog = {}

    # Classical Series (up to rank 8)
    for n in range(1, 9):
        # A_n series (SU(n+1))
        catalog[f"A{n}"] = LieGroup(
            name=f"A{n}",
            rank=n,
            dimension=n*(n+2),
            type=LieGroupType.CLASSICAL,
            root_system="A"
        )

        # B_n series (SO(2n+1))
        if n >= 2:
            catalog[f"B{n}"] = LieGroup(
                name=f"B{n}",
                rank=n,
                dimension=n*(2*n+1),
                type=LieGroupType.CLASSICAL,
                root_system="B"
            )

        # C_n series (Sp(n))
        if n >= 3:
            catalog[f"C{n}"] = LieGroup(
                name=f"C{n}",
                rank=n,
                dimension=n*(2*n+1),
                type=LieGroupType.CLASSICAL,
                root_system="C"
            )

        # D_n series (SO(2n))
        if n >= 4:
            catalog[f"D{n}"] = LieGroup(
                name=f"D{n}",
                rank=n,
                dimension=n*(2*n-1),
                type=LieGroupType.CLASSICAL,
                root_system="D"
            )

    # Exceptional Groups
    exceptional_groups = [
        ("G2", 2, 14),
        ("F4", 4, 52),
        ("E6", 6, 78),
        ("E7", 7, 133),
        ("E8", 8, 248)
    ]

    for name, rank, dim in exceptional_groups:
        catalog[name] = LieGroup(
            name=name,
            rank=rank,
            dimension=dim,
            type=LieGroupType.EXCEPTIONAL,
            root_system=name[0]
        )

    return catalog

def compute_root_system_density(group: LieGroup, x: np.ndarray) -> np.ndarray:
    """
    Compute density based on actual root system structure
    Includes Weyl group and root system complexity
    """
    norm = np.sqrt(np.sum(x**2, axis=-1))

    # Base density incorporating root system complexity
    complexity_factor = np.sqrt(group.root_complexity / group.rank)

    if group.type == LieGroupType.EXCEPTIONAL:
        if group.name == "E8":
            return np.exp(-norm/8) * (1 + 0.5*np.cos(norm*np.pi/2)) * complexity_factor
        elif group.name == "E7":
            return np.exp(-norm/7) * (1 + 0.4*np.cos(norm*np.pi/2)) * complexity_factor
        elif group.name == "E6":
            return np.exp(-norm/6) * (1 + 0.3*np.cos(norm*np.pi/2)) * complexity_factor
        elif group.name == "F4":
            return np.exp(-norm/4) * (1 + 0.25*np.cos(norm*np.pi/2)) * complexity_factor
        elif group.name == "G2":
            return np.exp(-norm/2) * (1 + 0.2*np.cos(norm*np.pi/2)) * complexity_factor
    else:
        weyl_factor = np.log(group.weyl_order) / group.rank
        if group.root_system == "A":
            return np.exp(-norm/group.rank) * weyl_factor
        elif group.root_system == "B":
            return np.exp(-norm/group.rank) * (1 + 0.1*np.cos(norm*np.pi/2)) * weyl_factor
        elif group.root_system == "C":
            return np.exp(-norm/group.rank) * (1 + 0.15*np.cos(norm*np.pi/2)) * weyl_factor
        elif group.root_system == "D":
            return np.exp(-norm/group.rank) * (1 + 0.2*np.cos(norm*np.pi/2)) * weyl_factor

    raise ValueError(f"Unknown root system for group {group.name}")

def compute_metrics(group: LieGroup, num_samples: int = 10000, num_bootstrap: int = 100) -> GroupMetrics:
    """
    Compute all metrics for a given Lie group
    """
    try:
        rng = np.random.default_rng()
        points = rng.normal(0, 1, (num_samples, group.rank))

        # Compute densities
        densities = compute_root_system_density(group, points)
        densities = densities / np.sum(densities)

        # Bootstrap for entropy estimation
        entropies = []
        for _ in range(num_bootstrap):
            idx = rng.integers(0, num_samples, num_samples)
            boot_densities = densities[idx]
            boot_densities = boot_densities / np.sum(boot_densities)
            entropy = -np.sum(boot_densities * np.log(boot_densities + 1e-10))
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)
        entropy_error = np.std(entropies)

        # Compute normalized metrics
        normalized_entropy = mean_entropy / np.log(group.dimension)
        complexity_metric = group.root_complexity * np.log(group.weyl_order)
        weyl_normalized_entropy = mean_entropy / np.log(group.weyl_order)

        return GroupMetrics(
            entropy=mean_entropy,
            entropy_error=entropy_error,
            normalized_entropy=normalized_entropy,
            complexity_metric=complexity_metric,
            weyl_normalized_entropy=weyl_normalized_entropy
        )

    except Exception as e:
        logger.error(f"Error computing metrics for {group.name}: {str(e)}")
        raise

def run_comprehensive_test():
    """
    Run comprehensive test across all Lie groups with enhanced metrics
    """
    try:
        catalog = create_lie_group_catalog()
        results = {}

        # Compute metrics for all groups
        for name, group in catalog.items():
            logger.info(f"Computing metrics for {name}")
            metrics = compute_metrics(group)
            results[name] = metrics

            logger.info(f"{name}:")
            logger.info(f"  Entropy: {metrics.entropy:.4f} ± {metrics.entropy_error:.4f}")
            logger.info(f"  Normalized Entropy: {metrics.normalized_entropy:.4f}")
            logger.info(f"  Complexity Metric: {metrics.complexity_metric:.4f}")
            logger.info(f"  Weyl-Normalized Entropy: {metrics.weyl_normalized_entropy:.4f}")

        # Create DataFrame for analysis
        df = pd.DataFrame({
            'Group': list(results.keys()),
            'Type': [catalog[name].type.value for name in results.keys()],
            'Rank': [catalog[name].rank for name in results.keys()],
            'Dimension': [catalog[name].dimension for name in results.keys()],
            'Entropy': [results[name].entropy for name in results.keys()],
            'Normalized_Entropy': [results[name].normalized_entropy for name in results.keys()],
            'Complexity_Metric': [results[name].complexity_metric for name in results.keys()],
            'Weyl_Normalized_Entropy': [results[name].weyl_normalized_entropy for name in results.keys()]
        })

        # Create visualizations
        fig = plt.figure(figsize=(20, 15))

        # 1. Entropy vs Rank
        ax1 = fig.add_subplot(221)
        sns.scatterplot(data=df, x='Rank', y='Entropy', hue='Type', style='Type', s=100, ax=ax1)
        ax1.set_title('Entropy vs Rank')

        # 2. Normalized Entropy vs Dimension
        ax2 = fig.add_subplot(222)
        sns.scatterplot(data=df, x='Dimension', y='Normalized_Entropy', hue='Type', style='Type', s=100, ax=ax2)
        ax2.set_title('Normalized Entropy vs Dimension')
        ax2.set_xscale('log')

        # 3. Complexity Metric vs Rank
        ax3 = fig.add_subplot(223)
        sns.scatterplot(data=df, x='Rank', y='Complexity_Metric', hue='Type', style='Type', s=100, ax=ax3)
        ax3.set_title('Complexity Metric vs Rank')

        # 4. Weyl-Normalized Entropy vs Complexity Metric
        ax4 = fig.add_subplot(224)
        sns.scatterplot(data=df, x='Complexity_Metric', y='Weyl_Normalized_Entropy', hue='Type', style='Type', s=100, ax=ax4)
        ax4.set_title('Weyl-Normalized Entropy vs Complexity Metric')
        ax4.set_xscale('log')

        plt.tight_layout()
        plt.show()

        # Statistical analysis
        metrics = ['Entropy', 'Normalized_Entropy', 'Complexity_Metric', 'Weyl_Normalized_Entropy']
        stats_results = {}

        for metric in metrics:
            classical_values = df[df['Type'] == 'classical'][metric]
            exceptional_values = df[df['Type'] == 'exceptional'][metric]

            t_stat, p_value = ttest_ind(classical_values, exceptional_values)
            stats_results[metric] = {'t_statistic': t_stat, 'p_value': p_value}

        return {
            'results': results,
            'stats': stats_results,
            'dataframe': df
        }

    except Exception as e:
        logger.error(f"Error in comprehensive test: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    try:
        results = run_comprehensive_test()
        end_time = time.time()

        logger.info(f"\nTest completed in {end_time - start_time:.2f} seconds")

        # Print statistical results
        logger.info("\nStatistical Analysis Results:")
        for metric, stats in results['stats'].items():
            logger.info(f"\n{metric}:")
            logger.info(f"  T-statistic: {stats['t_statistic']:.4f}")
            logger.info(f"  P-value: {stats['p_value']:.4f}")

        # Save results to CSV
        results['dataframe'].to_csv('lie_group_metrics.csv', index=False)

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
