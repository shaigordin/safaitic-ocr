#!/usr/bin/env python3
"""
Generate comparison charts for MLX-VLM model analysis
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Define model configurations
models = {
    'Qwen2.5-VL-7B': {
        'file': 'docs/data/qwen25-7b_50inscriptions.json',
        'size': '7B',
        'family': 'Qwen2.5',
        'color': '#FF6B6B'
    },
    'Qwen2-VL-2B': {
        'file': 'docs/data/qwen2-2b_50inscriptions.json',
        'size': '2B',
        'family': 'Qwen2',
        'color': '#4ECDC4'
    },
    'Qwen2-VL-7B': {
        'file': 'docs/data/qwen2-7b_50inscriptions.json',
        'size': '7B',
        'family': 'Qwen2',
        'color': '#45B7D1'
    },
    'Idefics3-8B': {
        'file': 'docs/data/idefics3-8b_50inscriptions.json',
        'size': '8B',
        'family': 'Idefics3',
        'color': '#96CEB4'
    },
    'Pixtral-12B': {
        'file': 'docs/data/pixtral-12b_50inscriptions.json',
        'size': '12B',
        'family': 'Pixtral',
        'color': '#FFEAA7'
    }
}

print("Loading results...")
# Load all results
results = {}
for model_name, config in models.items():
    try:
        with open(config['file'], 'r', encoding='utf-8') as f:
            results[model_name] = json.load(f)
        print(f"✓ Loaded {model_name}: {len(results[model_name]['results'])} inscriptions")
    except FileNotFoundError:
        print(f"✗ File not found: {config['file']}")
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")

print(f"\nSuccessfully loaded {len(results)} models\n")

# Calculate success rates
print("Calculating success rates...")
success_data = []

for model_name, data in results.items():
    total_prompts = 0
    successful_prompts = 0
    total_inscriptions = len(data['results'])
    successful_inscriptions = 0
    
    for inscription in data['results']:
        inscription_success = True
        for prompt_result in inscription['prompts']:
            total_prompts += 1
            if prompt_result.get('success', False):
                successful_prompts += 1
            else:
                inscription_success = False
        
        if inscription_success:
            successful_inscriptions += 1
    
    success_data.append({
        'Model': model_name,
        'Total Inscriptions': total_inscriptions,
        'Successful Inscriptions': successful_inscriptions,
        'Inscription Success Rate': (successful_inscriptions / total_inscriptions * 100),
        'Total Prompts': total_prompts,
        'Successful Prompts': successful_prompts,
        'Prompt Success Rate': (successful_prompts / total_prompts * 100),
        'Model Size': models[model_name]['size'],
        'Family': models[model_name]['family']
    })

df_success = pd.DataFrame(success_data)
df_success = df_success.sort_values('Prompt Success Rate', ascending=False)

print("SUCCESS RATE COMPARISON:")
print(df_success[['Model', 'Prompt Success Rate', 'Inscription Success Rate']].to_string(index=False))

# Chart 1: Success Rate Comparison
print("\nGenerating Chart 1: Success Rates...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Prompt-level success rates
ax1 = axes[0]
colors = [models[m]['color'] for m in df_success['Model']]
bars1 = ax1.barh(df_success['Model'], df_success['Prompt Success Rate'], color=colors, alpha=0.8)
ax1.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Prompt-Level Success Rates\n(150 prompts per model)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 105)
ax1.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center', fontweight='bold')

# Plot 2: Inscription-level success rates
ax2 = axes[1]
df_inscr = df_success.sort_values('Inscription Success Rate', ascending=False)
colors2 = [models[m]['color'] for m in df_inscr['Model']]
bars2 = ax2.barh(df_inscr['Model'], df_inscr['Inscription Success Rate'], color=colors2, alpha=0.8)
ax2.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Inscription-Level Success Rates\n(All 3 prompts must succeed)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('proposal/mlx_comparison_success_rates.png', dpi=300, bbox_inches='tight')
print("✓ Saved: proposal/mlx_comparison_success_rates.png")

# Collect response data for length analysis
print("\nAnalyzing response quality...")
response_data = []

for model_name, data in results.items():
    for inscription in data['results']:
        for prompt_result in inscription['prompts']:
            if prompt_result.get('success', False):
                response = prompt_result.get('response', '')
                response_data.append({
                    'Model': model_name,
                    'Prompt Type': prompt_result.get('prompt_type', 'unknown'),
                    'Response Length': len(response),
                    'Word Count': len(response.split()),
                    'Inscription': inscription['inscription_siglum']
                })

df_responses = pd.DataFrame(response_data)

# Chart 2: Response Length Distribution
print("Generating Chart 2: Response Lengths...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Box plot by model
ax1 = axes[0]
model_order = df_success['Model'].tolist()
colors_ordered = [models[m]['color'] for m in model_order]
bp1 = ax1.boxplot([df_responses[df_responses['Model'] == m]['Response Length'] for m in model_order],
                   labels=model_order, patch_artist=True, showmeans=True)

for patch, color in zip(bp1['boxes'], colors_ordered):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Response Length (characters)', fontsize=12, fontweight='bold')
ax1.set_title('Response Length Distribution by Model', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=15)

# Plot 2: Box plot by prompt type
ax2 = axes[1]
prompt_types = df_responses['Prompt Type'].unique()
bp2 = ax2.boxplot([df_responses[df_responses['Prompt Type'] == pt]['Response Length'] for pt in prompt_types],
                   labels=prompt_types, patch_artist=True, showmeans=True)

for patch in bp2['boxes']:
    patch.set_facecolor('#9B59B6')
    patch.set_alpha(0.7)

ax2.set_ylabel('Response Length (characters)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
ax2.set_title('Response Length Distribution by Prompt Type', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('proposal/mlx_comparison_response_lengths.png', dpi=300, bbox_inches='tight')
print("✓ Saved: proposal/mlx_comparison_response_lengths.png")

# Chart 3: Performance Heatmap
print("Generating Chart 3: Performance Heatmap...")
heatmap_data = []

for model_name, data in results.items():
    prompt_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    
    for inscription in data['results']:
        for prompt_result in inscription['prompts']:
            prompt_type = prompt_result.get('prompt_type', 'unknown')
            prompt_stats[prompt_type]['total'] += 1
            if prompt_result.get('success', False):
                prompt_stats[prompt_type]['success'] += 1
    
    for prompt_type, stats in prompt_stats.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        heatmap_data.append({
            'Model': model_name,
            'Prompt Type': prompt_type,
            'Success Rate': success_rate
        })

df_heatmap = pd.DataFrame(heatmap_data)
pivot_table = df_heatmap.pivot(index='Model', columns='Prompt Type', values='Success Rate')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
            cbar_kws={'label': 'Success Rate (%)'}, linewidths=0.5)
plt.title('Model Performance Heatmap by Prompt Type', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Prompt Type', fontsize=12, fontweight='bold')
plt.ylabel('Model', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('proposal/mlx_comparison_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: proposal/mlx_comparison_heatmap.png")

# Chart 4: Comprehensive Comparison
print("Generating Chart 4: Comprehensive Comparison...")
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Chart 1: Success Rates
ax1 = fig.add_subplot(gs[0, :2])
model_names = df_success['Model']
colors = [models[m]['color'] for m in model_names]
x_pos = np.arange(len(model_names))
bars = ax1.bar(x_pos, df_success['Prompt Success Rate'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Success Rates on 50 Safaitic Inscriptions', fontsize=15, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Score')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Chart 2: Model Sizes
ax2 = fig.add_subplot(gs[0, 2])
sizes = {'2B': 1, '7B': 2, '8B': 2.5, '12B': 3}
model_size_vals = [sizes[models[m]['size']] for m in model_names]
ax2.barh(model_names, model_size_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Relative Size', fontsize=12, fontweight='bold')
ax2.set_title('Model Sizes', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 3.5)

for i, (name, val) in enumerate(zip(model_names, model_names)):
    size = models[val]['size']
    ax2.text(sizes[size] + 0.1, i, size, va='center', fontweight='bold')

# Chart 3: Average Response Length
ax3 = fig.add_subplot(gs[1, :])
response_stats = df_responses.groupby('Model')['Response Length'].agg(['mean', 'std']).reindex(model_names)
x_pos = np.arange(len(model_names))
ax3.bar(x_pos, response_stats['mean'], yerr=response_stats['std'], 
        color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5,
        error_kw={'linewidth': 2, 'ecolor': 'gray'})
ax3.set_xlabel('Model', fontsize=13, fontweight='bold')
ax3.set_ylabel('Response Length (characters)', fontsize=13, fontweight='bold')
ax3.set_title('Average Response Length ± Std Dev', fontsize=15, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('MLX-VLM Comparative Analysis: 5 Models on Safaitic Ancient Inscriptions',
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('proposal/mlx_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: proposal/mlx_comprehensive_comparison.png")

print("\n" + "="*80)
print("ALL CHARTS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nFiles saved to proposal/:")
print("  • mlx_comparison_success_rates.png")
print("  • mlx_comparison_response_lengths.png") 
print("  • mlx_comparison_heatmap.png")
print("  • mlx_comprehensive_comparison.png")
print("\nReady for project proposal!")
