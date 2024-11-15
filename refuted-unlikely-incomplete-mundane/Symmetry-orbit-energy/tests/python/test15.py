import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_by_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Detailed analysis of each classical series and exceptional groups
    """
    series_stats = {}

    # Analyze classical series separately
    series = ['A', 'B', 'C', 'D']
    metrics = ['Entropy', 'Normalized_Entropy', 'Complexity_Metric', 'Weyl_Normalized_Entropy']

    logger.info("\nDetailed Series Analysis:")

    for s in series:
        series_data = df[df['Group'].str.startswith(s)]
        series_stats[s] = {
            'summary': series_data[metrics].describe(),
            'rank_correlation': {metric: series_data[['Rank', metric]].corr().iloc[0,1]
                               for metric in metrics}
        }

        logger.info(f"\nSeries {s}:")
        logger.info("\nSummary Statistics:")
        logger.info(series_stats[s]['summary'])
        logger.info("\nRank Correlations:")
        for metric, corr in series_stats[s]['rank_correlation'].items():
            logger.info(f"{metric}: {corr:.4f}")

    # Analyze exceptional series
    exceptional_data = df[df['Type'] == 'exceptional']
    series_stats['Exceptional'] = {
        'summary': exceptional_data[metrics].describe(),
        'rank_correlation': {metric: exceptional_data[['Rank', metric]].corr().iloc[0,1]
                           for metric in metrics}
    }

    logger.info("\nExceptional Series:")
    logger.info("\nSummary Statistics:")
    logger.info(series_stats['Exceptional']['summary'])
    logger.info("\nRank Correlations:")
    for metric, corr in series_stats['Exceptional']['rank_correlation'].items():
        logger.info(f"{metric}: {corr:.4f}")

    return series_stats

def analyze_root_systems(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Detailed analysis of root system properties
    """
    logger.info("\nRoot System Analysis:")

    # Compute root system metrics
    root_metrics = {
        'complexity_by_type': df.groupby('Type')['Complexity_Metric'].agg(['mean', 'std', 'min', 'max']),
        'entropy_by_type': df.groupby('Type')['Entropy'].agg(['mean', 'std', 'min', 'max']),
        'normalized_by_type': df.groupby('Type')['Normalized_Entropy'].agg(['mean', 'std', 'min', 'max']),
        'weyl_by_type': df.groupby('Type')['Weyl_Normalized_Entropy'].agg(['mean', 'std', 'min', 'max'])
    }

    for metric_name, metric_data in root_metrics.items():
        logger.info(f"\n{metric_name}:")
        logger.info(metric_data)

    return root_metrics

def perform_statistical_tests(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive statistical testing
    """
    metrics = ['Entropy', 'Normalized_Entropy', 'Complexity_Metric', 'Weyl_Normalized_Entropy']
    tests = {}

    logger.info("\nStatistical Tests:")

    for metric in metrics:
        # Split data by type
        classical = df[df['Type'] == 'classical'][metric]
        exceptional = df[df['Type'] == 'exceptional'][metric]

        # Perform t-test
        t_stat, t_pval = ttest_ind(classical, exceptional)

        # Perform one-way ANOVA across all series
        series_groups = [df[df['Group'].str.startswith(s)][metric] for s in ['A', 'B', 'C', 'D']]
        series_groups.append(exceptional)
        f_stat, f_pval = f_oneway(*series_groups)

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(df[metric], df['Group'].str[0])

        tests[metric] = {
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'tukey_results': tukey
        }

        logger.info(f"\n{metric}:")
        logger.info(f"T-test (Classical vs Exceptional):")
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {t_pval:.4f}")
        logger.info(f"One-way ANOVA:")
        logger.info(f"  F-statistic: {f_stat:.4f}")
        logger.info(f"  p-value: {f_pval:.4f}")
        logger.info("\nTukey's HSD Results:")
        logger.info(tukey)

    return tests

def create_visualizations(df: pd.DataFrame) -> None:
    """
    Create comprehensive visualizations
    """
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [15, 10]

    # 1. Multiple metric comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Entropy vs Rank with series identification
    sns.scatterplot(data=df, x='Rank', y='Entropy', hue='Group', style='Type', s=100, ax=ax1)
    ax1.set_title('Entropy vs Rank by Series')

    # Normalized Entropy vs Dimension
    sns.scatterplot(data=df, x='Dimension', y='Normalized_Entropy', hue='Group', style='Type', s=100, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_title('Normalized Entropy vs Dimension')

    # Complexity Metric vs Rank
    sns.scatterplot(data=df, x='Rank', y='Complexity_Metric', hue='Group', style='Type', s=100, ax=ax3)
    ax3.set_yscale('log')
    ax3.set_title('Complexity Metric vs Rank')

    # Weyl-Normalized Entropy vs Complexity
    sns.scatterplot(data=df, x='Complexity_Metric', y='Weyl_Normalized_Entropy', hue='Group', style='Type', s=100, ax=ax4)
    ax4.set_xscale('log')
    ax4.set_title('Weyl-Normalized Entropy vs Complexity')

    plt.tight_layout()
    plt.savefig('lie_group_metrics.png')
    plt.close()

    # 2. Distribution plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    sns.boxplot(data=df, x='Type', y='Entropy', ax=ax1)
    ax1.set_title('Entropy Distribution by Type')

    sns.boxplot(data=df, x='Type', y='Normalized_Entropy', ax=ax2)
    ax2.set_title('Normalized Entropy Distribution by Type')

    sns.boxplot(data=df, x='Type', y='Complexity_Metric', ax=ax3)
    ax3.set_yscale('log')
    ax3.set_title('Complexity Metric Distribution by Type')

    sns.boxplot(data=df, x='Type', y='Weyl_Normalized_Entropy', ax=ax4)
    ax4.set_title('Weyl-Normalized Entropy Distribution by Type')

    plt.tight_layout()
    plt.savefig('lie_group_distributions.png')
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[['Rank', 'Dimension', 'Entropy', 'Normalized_Entropy',
                           'Complexity_Metric', 'Weyl_Normalized_Entropy']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Metrics')
    plt.savefig('correlation_matrix.png')
    plt.close()

def run_enhanced_analysis(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive enhanced analysis
    """
    results = {}

    # 1. Series Analysis
    results['series_analysis'] = analyze_by_series(df)

    # 2. Root System Analysis
    results['root_analysis'] = analyze_root_systems(df)

    # 3. Statistical Tests
    results['statistical_tests'] = perform_statistical_tests(df)

    # 4. Create Visualizations
    create_visualizations(df)

    # 5. Additional Metrics
    results['rank_effects'] = {
        'entropy_rank_corr': df['Entropy'].corr(df['Rank']),
        'complexity_rank_corr': df['Complexity_Metric'].corr(df['Rank']),
        'dimension_rank_corr': df['Dimension'].corr(df['Rank'])
    }

    # 6. Exceptional Group Analysis
    exceptional_df = df[df['Type'] == 'exceptional']
    results['exceptional_analysis'] = {
        'entropy_rank_corr': exceptional_df['Entropy'].corr(exceptional_df['Rank']),
        'complexity_rank_corr': exceptional_df['Complexity_Metric'].corr(exceptional_df['Rank']),
        'dimension_rank_corr': exceptional_df['Dimension'].corr(exceptional_df['Rank'])
    }

    return results

if __name__ == "__main__":
    # Load the data from the previous run
    df = pd.read_csv('lie_group_metrics.csv')

    # Run enhanced analysis
    results = run_enhanced_analysis(df)

    # Save detailed results to file
    with open('detailed_analysis.txt', 'w') as f:
        f.write("Enhanced Analysis Results\n")
        f.write("========================\n\n")

        for key, value in results.items():
            f.write(f"\n{key}:\n")
            f.write("-------------------\n")
            f.write(str(value))
            f.write("\n\n")
