import wandb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Connect to your sweep
api = wandb.Api()
sweep_path = "christianz97-national-tsing-hua-university/2025-Fall-NLP-HW2/qxipmncl"
sweep = api.sweep(sweep_path)

print(f"Analyzing Sweep: {sweep.name}")
print(f"Total runs: {len(sweep.runs)}")
print("-" * 80)

# Extract data from all runs
results = []
for run in sweep.runs:
    # 明確轉換為字典
    config_dict = dict(run.config) if run.config else {}
    summary_dict = dict(run.summary) if run.summary else {}
    
    results.append({
        'run_name': run.name,
        'state': run.state,
        'lr': config_dict.get('lr', np.nan),
        'weight_decay': config_dict.get('weight_decay', np.nan),
        'rnn_type': config_dict.get('rnn_type', 'Unknown'),
        'val_accuracy': summary_dict.get('val_accuracy', 0.0),
        'best_val_accuracy': summary_dict.get('best_val_accuracy', 0.0),
    })

df = pd.DataFrame(results)
print(f"\nSuccessfully loaded {len(df)} runs")

# Remove rows with missing data
df = df.dropna(subset=['lr', 'weight_decay'])
print(f"After filtering: {len(df)} runs with complete data")

if len(df) == 0:
    print("\nNo valid runs found. Please check wandb dashboard manually.")
    exit()

# Filter by state
df_finished = df[df['state'] == 'finished'].copy()
df_crashed = df[df['state'] == 'crashed'].copy()

print(f"\nRun Statistics:")
print(f"  Finished: {len(df_finished)}")
print(f"  Crashed:  {len(df_crashed)}")

if len(df_finished) == 0:
    print("\nNo finished runs. Analyzing all runs with data...")
    df_finished = df.copy()

# Sort by accuracy
df_finished = df_finished.sort_values('best_val_accuracy', ascending=False)

print("\n" + "=" * 80)
print("TOP 10 CONFIGURATIONS")
print("=" * 80)

for i, (idx, row) in enumerate(df_finished.head(10).iterrows(), 1):
    print(f"\nRank {i}: {row['run_name']}")
    print(f"  Accuracy:      {row['best_val_accuracy']:.4f}")
    print(f"  LR:            {row['lr']:.6f}")
    print(f"  Weight Decay:  {row['weight_decay']:.6f}")
    print(f"  RNN Type:      {row['rnn_type']}")
    print(f"  State:         {row['state']}")

# Statistics
top10 = df_finished.head(10)

print("\n" + "=" * 80)
print("PARAMETER STATISTICS (Top 10)")
print("=" * 80)

print(f"\nLearning Rate:")
print(f"  Mean:   {top10['lr'].mean():.6f}")
print(f"  Median: {top10['lr'].median():.6f}")
print(f"  Range:  [{top10['lr'].min():.6f}, {top10['lr'].max():.6f}]")

print(f"\nWeight Decay:")
print(f"  Mean:   {top10['weight_decay'].mean():.6f}")
print(f"  Median: {top10['weight_decay'].median():.6f}")
print(f"  Range:  [{top10['weight_decay'].min():.6f}, {top10['weight_decay'].max():.6f}]")

print(f"\nRNN Type Distribution:")
for rnn_type, count in top10['rnn_type'].value_counts().items():
    print(f"  {rnn_type}: {count}")

# By RNN type
print("\n" + "=" * 80)
print("PERFORMANCE BY RNN TYPE")
print("=" * 80)

for rnn_type in df_finished['rnn_type'].unique():
    subset = df_finished[df_finished['rnn_type'] == rnn_type]
    print(f"\n{rnn_type}:")
    print(f"  Runs:     {len(subset)}")
    print(f"  Mean:     {subset['best_val_accuracy'].mean():.4f}")
    print(f"  Max:      {subset['best_val_accuracy'].max():.4f}")
    print(f"  Std:      {subset['best_val_accuracy'].std():.4f}")

# Export
df_finished.to_csv('sweep_results.csv', index=False)
print(f"\n" + "=" * 80)
print("Results saved to: sweep_results.csv")

# Refined config
print("\n" + "=" * 80)
print("SUGGESTED REFINED CONFIG")
print("=" * 80)

q1_lr = top10['lr'].quantile(0.25)
q3_lr = top10['lr'].quantile(0.75)
q1_wd = top10['weight_decay'].quantile(0.25)
q3_wd = top10['weight_decay'].quantile(0.75)
best_rnn = top10['rnn_type'].value_counts().index[0]

print(f"""
program: nlp_hw2.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: {q1_lr:.6f}
    max: {q3_lr:.6f}
  weight_decay:
    distribution: log_uniform_values
    min: {q1_wd:.6f}
    max: {q3_wd:.6f}
  rnn_type:
    value: {best_rnn}
""")

print("=" * 80)
best = df_finished.iloc[0]
print(f"\nBest Run: {best['run_name']}")
print(f"  Accuracy: {best['best_val_accuracy']:.4f}")
print(f"  LR: {best['lr']:.6f}, WD: {best['weight_decay']:.6f}, Type: {best['rnn_type']}")
