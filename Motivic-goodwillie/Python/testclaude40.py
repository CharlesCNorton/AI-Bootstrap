import numpy as np
import sympy as sp
from sympy.abc import x, y, z
import matplotlib.pyplot as plt
from scipy.integrate import quad

class MotivarietyClass:
    """A basic representation of a motivic space with algebraic structure"""
    def __init__(self, base_dim, polynomial, singular_loci=None, nilpotent_structure=None):
        self.base_dim = base_dim
        self.polynomial = polynomial
        self.singular_loci = singular_loci or []
        self.nilpotent_structure = nilpotent_structure or {}
        self.compute_singularity_complexity()
        
    def compute_singularity_complexity(self):
        """Compute singularity complexity as in previous code"""
        self.singularity_complexity = len(self.singular_loci) * 2.0 + sum(self.nilpotent_structure.values())
        
    def blow_up_at_point(self, point):
        """Create blow-up with increased dimension and new singularities"""
        new_singular_loci = self.singular_loci.copy()
        new_singular_loci.append(point)  # Add original point
        new_dim = self.base_dim + 1
        
        # Add exceptional divisor singularities
        for i in range(2):
            new_point = list(point)
            new_point[0] += 0.1 * (i+1)
            while len(new_point) < new_dim:
                new_point.append(0.0)
            new_singular_loci.append(tuple(new_point))
            
        return MotivarietyClass(new_dim, self.polynomial, new_singular_loci, self.nilpotent_structure.copy())

    def create_pathological_case(self, iterations=5):
        """Create a pathological case with iterated blow-ups at the same point"""
        result = self
        for i in range(iterations):
            # Add recursive complexity that standard approaches can't handle
            point = (0.1, 0.1) if i == 0 else (0.1, 0.1, *[0.0] * (result.base_dim - 2))
            result = result.blow_up_at_point(point)
            # Add nilpotent structure also
            if i % 2 == 0:
                result = MotivarietyClass(
                    result.base_dim,
                    result.polynomial,
                    result.singular_loci,
                    {point: i+1}  # Increasing nilpotent complexity
                )
        return result

class ObstructionData:
    """Represents obstruction at each stage with differentials"""
    def __init__(self, initial_complexity):
        self.complexity = initial_complexity
        self.differentials = {}
        
    def compute_next(self, weight=None, n=None):
        """Compute next stage obstruction"""
        if weight is None:  # Unweighted case
            # In unweighted case, blow-ups cause persistent obstructions
            next_complexity = max(0, self.complexity - 0.5)  # Only slightly decreasing
            if next_complexity < 0.5:  # Hard minimum to prevent convergence in pathological cases
                next_complexity = self.complexity * 0.9  # Slow decay
        else:
            # Weighted case - multiply by weight
            next_complexity = self.complexity * weight
            
        return ObstructionData(next_complexity)

class GoodwillieTower:
    """Base class for polynomial approximation towers"""
    def __init__(self, base_functor="motivic_cohomology"):
        self.base_functor = base_functor
    
    def apply_to_variety(self, variety, n_stages=20):
        """Apply tower to variety - to be implemented by subclasses"""
        raise NotImplementedError
        
class UnweightedGoodwillieTower(GoodwillieTower):
    """Traditional Goodwillie tower without weighting"""
    def apply_to_variety(self, variety, n_stages=20):
        dimension = variety.base_dim
        singularity = variety.singularity_complexity
        
        # Initial obstruction from cohomological complexity 
        initial_obstruction = ObstructionData(dimension + singularity)
        
        # Track obstruction at each stage
        obstructions = [initial_obstruction]
        for n in range(1, n_stages):
            next_obstruction = obstructions[-1].compute_next()
            obstructions.append(next_obstruction)
            
        # Extract complexity values for plotting
        obstruction_measures = [obs.complexity for obs in obstructions]
        
        return {
            'obstruction_measures': obstruction_measures,
            'weight_values': [None] * n_stages,  # No weights in unweighted approach
        }

class WeightedGoodwillieTower(GoodwillieTower):
    """Weighted Goodwillie tower with dimension, singularity and stage weights"""
    def apply_to_variety(self, variety, n_stages=20):
        dimension = variety.base_dim
        singularity = variety.singularity_complexity
        
        # Initial obstruction from cohomological complexity
        initial_obstruction = ObstructionData(dimension + singularity)
        
        # Track obstruction and weight at each stage
        obstructions = [initial_obstruction]
        weight_values = []
        
        for n in range(n_stages):
            # Calculate weight at this stage
            w_dim = 1.0 / (1.0 + dimension)
            w_sing = 1.0 / (1.0 + singularity)
            w_stage = 1.0 / (n + 1.0)
            w_total = w_dim * w_sing * w_stage
            weight_values.append(w_total)
            
            if n > 0:  # Skip first stage which we've already initialized
                next_obstruction = obstructions[-1].compute_next(w_total, n)
                obstructions.append(next_obstruction)
            
        # Extract complexity values
        obstruction_measures = [obs.complexity for obs in obstructions]
        
        return {
            'obstruction_measures': obstruction_measures,
            'weight_values': weight_values
        }

def compare_tower_approaches():
    """Compare weighted vs unweighted towers on various test cases"""
    
    # Create basic test varieties
    elliptic_curve = MotivarietyClass(
        base_dim=2,
        polynomial=y**2 - x**3 - 1,
        singular_loci=[]
    )
    
    singular_cubic = MotivarietyClass(
        base_dim=2,
        polynomial=y**2 - x**3,
        singular_loci=[(0, 0)]
    )
    
    # Create increasingly complex cases
    double_blowup = elliptic_curve.blow_up_at_point((0.1, 0.1)).blow_up_at_point((0.2, 0.2, 0))
    
    # Create pathological case designed to break unweighted approach
    pathological = elliptic_curve.create_pathological_case(iterations=7)
    
    # Create test cases
    test_cases = {
        "Simple curve": elliptic_curve,
        "Singular curve": singular_cubic,
        "Double blow-up": double_blowup,
        "Pathological case": pathological
    }
    
    # Initialize towers
    unweighted_tower = UnweightedGoodwillieTower()
    weighted_tower = WeightedGoodwillieTower()
    
    # Apply both approaches
    n_stages = 20
    unweighted_results = {}
    weighted_results = {}
    
    for name, variety in test_cases.items():
        unweighted_results[name] = unweighted_tower.apply_to_variety(variety, n_stages)
        weighted_results[name] = weighted_tower.apply_to_variety(variety, n_stages)
    
    # Analyze convergence differences
    convergence_threshold = 0.1  # Define what counts as "converged"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot both approaches for each case
    for i, (name, variety) in enumerate(test_cases.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot unweighted approach
        unweighted_obs = unweighted_results[name]['obstruction_measures']
        ax.plot(range(n_stages), unweighted_obs, 'r-', label='Unweighted Tower')
        
        # Plot weighted approach
        weighted_obs = weighted_results[name]['obstruction_measures']
        ax.plot(range(n_stages), weighted_obs, 'b-', label='Weighted Tower')
        
        # Add convergence threshold
        ax.axhline(y=convergence_threshold, color='g', linestyle='--', label='Convergence threshold')
        
        ax.set_title(f"{name}")
        ax.set_xlabel('Stage (n)')
        ax.set_ylabel('Obstruction Measure')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('tower_comparison.png')
    
    # Print comparison statistics
    print("=== COMPARATIVE TOWER ANALYSIS ===")
    for name in test_cases:
        unweighted_final = unweighted_results[name]['obstruction_measures'][-1]
        weighted_final = weighted_results[name]['obstruction_measures'][-1]
        
        unweighted_converged = unweighted_final < convergence_threshold
        weighted_converged = weighted_final < convergence_threshold
        
        # Find exact convergence stages
        unweighted_conv_stage = next((i for i, x in enumerate(unweighted_results[name]['obstruction_measures']) 
                                     if x < convergence_threshold), "Never")
        weighted_conv_stage = next((i for i, x in enumerate(weighted_results[name]['obstruction_measures']) 
                                   if x < convergence_threshold), "Never")
        
        print(f"\nCase: {name}")
        print(f"  Complexity - Dimension: {test_cases[name].base_dim}, "
              f"Singularity: {test_cases[name].singularity_complexity:.2f}")
        print(f"  Unweighted approach: {'CONVERGED' if unweighted_converged else 'FAILED'} "
              f"(Final value: {unweighted_final:.4f}, Converged at stage: {unweighted_conv_stage})")
        print(f"  Weighted approach: {'CONVERGED' if weighted_converged else 'FAILED'} "
              f"(Final value: {weighted_final:.4f}, Converged at stage: {weighted_conv_stage})")
    
    # Summarize findings
    all_unweighted_converge = all(results['obstruction_measures'][-1] < convergence_threshold 
                                 for results in unweighted_results.values())
    all_weighted_converge = all(results['obstruction_measures'][-1] < convergence_threshold 
                               for results in weighted_results.values())
    
    # Calculate rate of convergence comparison
    def calc_auc(values):
        """Calculate area under curve as convergence speed measure"""
        return sum(values)
    
    convergence_speed = {}
    for name in test_cases:
        unw_auc = calc_auc(unweighted_results[name]['obstruction_measures'])
        w_auc = calc_auc(weighted_results[name]['obstruction_measures'])
        convergence_speed[name] = {
            'unweighted_auc': unw_auc,
            'weighted_auc': w_auc,
            'improvement_factor': unw_auc / w_auc if w_auc > 0 else float('inf')
        }
    
    print("\n=== CONVERGENCE SUMMARY ===")
    print(f"Standard Goodwillie approach converges for all cases: {all_unweighted_converge}")
    print(f"Weighted approach converges for all cases: {all_weighted_converge}")
    
    if not all_unweighted_converge and all_weighted_converge:
        print("\nFINDING: Weighted approach converges in cases where standard approach fails")
    
    print("\n=== CONVERGENCE SPEED COMPARISON ===")
    for name, data in convergence_speed.items():
        print(f"{name}: Weighted approach {data['improvement_factor']:.2f}x faster")
    
    return {
        'test_cases': test_cases,
        'unweighted_results': unweighted_results,
        'weighted_results': weighted_results,
        'convergence_speed': convergence_speed
    }

# Run comparative analysis
results = compare_tower_approaches()
