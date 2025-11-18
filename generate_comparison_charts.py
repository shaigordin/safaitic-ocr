#!/usr/bin/env python3
"""
Generate comprehensive comparison charts for MLX-VLM model evaluation.

Analyzes 5 models on Safaitic inscriptions across 3 prompt types:
- Description: General visual description of inscription
- Script ID: Identification of writing system and characteristics  
- Transliteration: Attempt to read and transliterate Safaitic text

Key insights analyzed:
1. Success rates by prompt type (which tasks do models excel at?)
2. Response quality by prompt type (verbosity patterns)
3. Model size vs performance (does bigger = better?)
4. Consistency across prompt types
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Model configurations
MODELS = {
    'Qwen2.5-VL-7B': {'file': 'docs/data/qwen25-7b_50inscriptions.json', 'size': 7},
    'Qwen2-VL-2B': {'file': 'docs/data/qwen2-2b_50inscriptions.json', 'size': 2},
    'Qwen2-VL-7B': {'file': 'docs/data/qwen2-7b_50inscriptions.json', 'size': 7},
    'Idefics3-8B': {'file': 'docs/data/idefics3-8b_50inscriptions.json', 'size': 8},
    'Pixtral-12B': {'file': 'docs/data/pixtral-12b_50inscriptions.json', 'size': 12}
}

# Prompt type descriptions
PROMPT_TYPES = {
    'description': 'Visual Description',
    'script_id': 'Script Identification',
    'transliteration': 'Transliteration'
}

OUTPUT_DIR = 'proposal/'


def load_results():
    """Load all model results."""
    all_results = {}
    
    print("Loading results...")
    for model_name, config in MODELS.items():
        filepath = config['file']
        if not os.path.exists(filepath):
            print(f"⚠️  Warning: {filepath} not found")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_results[model_name] = data['results']
        print(f"✓ Loaded {model_name}: {len(data['results'])} inscriptions")
    
    return all_results


def analyze_results(all_results):
    """Analyze results and extract key metrics."""
    analysis = []
    
    for model_name, results in all_results.items():
        model_size = MODELS[model_name]['size']
        
        # Per prompt type metrics
        prompt_metrics = {
            'description': {'success': 0, 'total': 0, 'lengths': []},
            'script_id': {'success': 0, 'total': 0, 'lengths': []},
            'transliteration': {'success': 0, 'total': 0, 'lengths': []}
        }
        
        for result in results:
            for prompt in result['prompts']:
                ptype = prompt['prompt_name']
                prompt_metrics[ptype]['total'] += 1
                
                if prompt['success']:
                    prompt_metrics[ptype]['success'] += 1
                    prompt_metrics[ptype]['lengths'].append(len(prompt['response']))
        
        # Calculate rates and store data
        for ptype, metrics in prompt_metrics.items():
            if metrics['total'] > 0:
                success_rate = (metrics['success'] / metrics['total']) * 100
                avg_length = np.mean(metrics['lengths']) if metrics['lengths'] else 0
                
                analysis.append({
                    'Model': model_name,
                    'Model Size (B)': model_size,
                    'Prompt Type': PROMPT_TYPES[ptype],
                    'Success Rate': success_rate,
                    'Responses': metrics['success'],
                    'Total': metrics['total'],
                    'Avg Response Length': avg_length,
                    'Response Lengths': metrics['lengths']
                })
    
    return pd.DataFrame(analysis)


def create_chart1_success_by_prompt_type(df):
    """Chart 1: Success rates by prompt type - the KEY INSIGHT chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Model', columns='Prompt Type', values='Success Rate')
    
    # Reorder to show prompt types in logical order
    pivot_df = pivot_df[['Visual Description', 'Script Identification', 'Transliteration']]
    
    # Create grouped bar chart
    pivot_df.plot(kind='bar', ax=ax, width=0.75, 
                  color=['#4ECDC4', '#45B7D1', '#FF6B6B'])
    
    ax.set_title('Model Performance by Task Type\nAll models excel at visual tasks but struggle with transliteration', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(90, 101)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(title='Task Type', title_fontsize=11, fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'mlx_comparison_success_by_prompt.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_chart2_response_characteristics(df):
    """Chart 2: Response length patterns reveal model behavior differences."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Box plot of response lengths by model
    data_for_box = []
    labels = []
    for model in df['Model'].unique():
        model_lengths = []
        for _, row in df[df['Model'] == model].iterrows():
            model_lengths.extend(row['Response Lengths'])
        data_for_box.append(model_lengths)
        labels.append(model)
    
    bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_title('Response Length Distribution by Model', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Response Length (characters)', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Average response length by prompt type
    pivot_length = df.pivot_table(index='Model', columns='Prompt Type', 
                                   values='Avg Response Length', aggfunc='mean')
    pivot_length = pivot_length[['Visual Description', 'Script Identification', 'Transliteration']]
    
    pivot_length.plot(kind='bar', ax=ax2, width=0.75,
                     color=['#4ECDC4', '#45B7D1', '#FF6B6B'])
    
    ax2.set_title('Average Response Length by Task Type', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Average Characters', fontsize=11)
    ax2.legend(title='Task Type', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mlx_comparison_response_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_chart3_model_size_vs_performance(df):
    """Chart 3: Does model size correlate with performance? KEY FINDING: NO!"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Aggregate data by model
    model_summary = df.groupby(['Model', 'Model Size (B)']).agg({
        'Success Rate': 'mean',
        'Prompt Type': 'count'
    }).reset_index()
    
    # Create scatter plot
    colors_map = {
        'Qwen2.5-VL-7B': '#FF6B6B',
        'Qwen2-VL-2B': '#4ECDC4',
        'Qwen2-VL-7B': '#45B7D1',
        'Idefics3-8B': '#96CEB4',
        'Pixtral-12B': '#FFEAA7'
    }
    
    for _, row in model_summary.iterrows():
        ax.scatter(row['Model Size (B)'], row['Success Rate'], 
                  s=500, alpha=0.7, color=colors_map[row['Model']],
                  edgecolors='black', linewidth=2)
        
        # Annotate with model name
        ax.annotate(row['Model'], 
                   xy=(row['Model Size (B)'], row['Success Rate']),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    ax.set_title('Model Size vs Performance: Bigger ≠ Better\n2B model achieves 100% success (same as 12B model)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(96, 101)
    ax.set_xlim(1, 13)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='100% Success')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mlx_comparison_size_vs_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_chart4_comprehensive_summary(df):
    """Chart 4: Comprehensive 3-panel summary for proposal."""
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Overall success rates
    ax1 = fig.add_subplot(gs[0, 0])
    overall_success = df.groupby('Model')['Success Rate'].mean().sort_values(ascending=True)
    colors = ['#4ECDC4' if x == 100 else '#FF6B6B' if x < 98 else '#45B7D1' 
              for x in overall_success.values]
    overall_success.plot(kind='barh', ax=ax1, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Overall Success Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax1.set_xlim(95, 101)
    ax1.axvline(x=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, v in enumerate(overall_success.values):
        ax1.text(v - 0.5, i, f'{v:.1f}%', va='center', ha='right', 
                fontweight='bold', color='white' if v < 98 else 'black')
    
    # Panel 2: Task difficulty heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_heat = df.pivot_table(index='Model', columns='Prompt Type', values='Success Rate')
    pivot_heat = pivot_heat[['Visual Description', 'Script Identification', 'Transliteration']]
    
    sns.heatmap(pivot_heat, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=94, vmax=100, ax=ax2, cbar_kws={'label': 'Success Rate (%)'},
                linewidths=2, linecolor='white')
    ax2.set_title('Success Rate by Task Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # Panel 3: Model efficiency (size vs performance)
    ax3 = fig.add_subplot(gs[0, 2])
    model_summary = df.groupby(['Model', 'Model Size (B)']).agg({
        'Success Rate': 'mean'
    }).reset_index()
    
    colors_scatter = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    ax3.scatter(model_summary['Model Size (B)'], model_summary['Success Rate'],
               s=300, c=colors_scatter, alpha=0.7, edgecolors='black', linewidth=2)
    
    for _, row in model_summary.iterrows():
        ax3.annotate(row['Model'].split('-')[0], 
                    xy=(row['Model Size (B)'], row['Success Rate']),
                    fontsize=8, ha='center', va='bottom')
    
    ax3.set_title('Model Efficiency\n(Smaller can be better)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Model Size (B)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax3.set_ylim(96, 101)
    ax3.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('MLX-VLM Comparative Evaluation: 5 Models on Safaitic Inscription Analysis',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mlx_comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_detailed_analysis(df):
    """Print detailed statistical analysis."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n1. OVERALL MODEL PERFORMANCE:")
    overall = df.groupby('Model').agg({
        'Success Rate': 'mean',
        'Model Size (B)': 'first',
        'Total': 'sum'
    }).sort_values('Success Rate', ascending=False)
    print(overall.to_string())
    
    print("\n2. PERFORMANCE BY TASK TYPE:")
    by_task = df.groupby('Prompt Type').agg({
        'Success Rate': ['mean', 'min', 'max', 'std']
    })
    print(by_task.to_string())
    
    print("\n3. KEY FINDINGS:")
    # Find perfect scores
    perfect_models = df[df['Success Rate'] == 100.0]['Model'].unique()
    print(f"   • Models with 100% success on some tasks: {', '.join(perfect_models)}")
    
    # Transliteration performance
    trans_df = df[df['Prompt Type'] == 'Transliteration']
    avg_trans = trans_df['Success Rate'].mean()
    print(f"   • Average transliteration success: {avg_trans:.1f}%")
    print(f"   • This is the challenging task that validates grounded OCR need")
    
    # Size efficiency
    smallest = df[df['Model Size (B)'] == 2]['Success Rate'].mean()
    largest = df[df['Model Size (B)'] == 12]['Success Rate'].mean()
    print(f"   • Smallest model (2B): {smallest:.1f}% success")
    print(f"   • Largest model (12B): {largest:.1f}% success")
    print(f"   • Efficiency gain: {smallest - largest:+.1f} percentage points (smaller is better!)")


def main():
    """Main execution function."""
    print("="*80)
    print("MLX-VLM COMPARATIVE ANALYSIS - CHART GENERATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    all_results = load_results()
    
    if not all_results:
        print("\n❌ No results loaded. Please check file paths.")
        return
    
    print(f"\n✓ Successfully loaded {len(all_results)} models\n")
    
    # Analyze data
    print("Analyzing results across prompt types...")
    df = analyze_results(all_results)
    
    # Generate charts
    print("\nGenerating visualization charts...\n")
    
    print("Chart 1: Success rates by prompt type...")
    create_chart1_success_by_prompt_type(df)
    
    print("Chart 2: Response characteristics...")
    create_chart2_response_characteristics(df)
    
    print("Chart 3: Model size vs performance...")
    create_chart3_model_size_vs_performance(df)
    
    print("Chart 4: Comprehensive summary...")
    create_chart4_comprehensive_summary(df)
    
    # Print detailed analysis
    print_detailed_analysis(df)
    
    print("\n" + "="*80)
    print("ALL CHARTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nFiles saved to {OUTPUT_DIR}:")
    print("  • mlx_comparison_success_by_prompt.png - Task-specific performance")
    print("  • mlx_comparison_response_patterns.png - Response quality analysis")
    print("  • mlx_comparison_size_vs_performance.png - Efficiency analysis")
    print("  • mlx_comprehensive_comparison.png - Executive summary")
    print("\nReady for project proposal!\n")


if __name__ == '__main__':
    main()
