import wandb
import pandas as pd
import numpy as np

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
    try:
        # Handle config as dict
        if hasattr(run.config, 'items'):
            config_dict = dict(run.config)
        else:
            config_dict = {}
        
        summary_dict = dict(run.summary) if hasattr(run.summary, 'items') else {}
        
        results.append({
            'run_name': run.name,
            'state': run.state,
            'lr': config_dict.get('lr', np.nan),
            'weight_decay': config_dict.get('weight_decay', np.nan),
            'rnn_type': config_dict.get('rnn_type', 'Unknown'),
            'val_accuracy': summary_dict.get('val_accuracy', 0),
            'best_val_accuracy': summary_dict.get('best_val_accuracy', 0),
            'train_loss': summary_dict.get('train_loss', np.nan),
            'total_params': summary_dict.get('total_params', 0),
        })
    except Exception as e:
        print(f"Warning: Skipping run {run.name} due to error: {e}")
        continue

df = pd.DataFrame(results)

# Remove rows with missing critical data
df = df.dropna(subset=['lr', 'weight_decay'])

# Filter completed runs only
df_finished = df[df['state'] == 'finished'].copy()
df_crashed = df[df['state'] == 'crashed'].copy()

print(f"\nRun Statistics:")
print(f"  Finished: {len(df_finished)}")
print(f"  Crashed: {len(df_crashed)}")
print(f"  Other: {len(df) - len(df_finished) - len(df_crashed)}")

if len(df_finished) == 0:
    print("\nNo finished runs found. Check if runs completed successfully.")
    exit()

# Sort by best validation accuracy
df_finished = df_finished.sort_values('best_val_accuracy', ascending=False)

print("\n" + "=" * 80)
print("TOP 10 CONFIGURATIONS")
print("=" * 80)

for idx, row in df_finished.head(10).iterrows():
    rank = list(df_finished.index).index(idx) + 1
    print(f"\nRank {rank}: {row['run_name']}")
    print(f"  Val Accuracy:  {row['best_val_accuracy']:.4f}")
    print(f"  LR:            {row['lr']:.6f}")
    print(f"  Weight Decay:  {row['weight_decay']:.6f}")
    print(f"  RNN Type:      {row['rnn_type']}")

# Statistical analysis
print("\n" + "=" * 80)
print("PARAMETER STATISTICS (Top 10 runs)")
print("=" * 80)

top10 = df_finished.head(10)

print(f"\nLearning Rate:")
print(f"  Mean:   {top10['lr'].mean():.6f}")
print(f"  Median: {top10['lr'].median():.6f}")
print(f"  Min:    {top10['lr'].min():.6f}")
print(f"  Max:    {top10['lr'].max():.6f}")

print(f"\nWeight Decay:")
print(f"  Mean:   {top10['weight_decay'].mean():.6f}")
print(f"  Median: {top10['weight_decay'].median():.6f}")
print(f"  Min:    {top10['weight_decay'].min():.6f}")
print(f"  Max:    {top10['weight_decay'].max():.6f}")

print(f"\nRNN Type Distribution (Top 10):")
rnn_counts = top10['rnn_type'].value_counts()
for rnn_type, count in rnn_counts.items():
    print(f"  {rnn_type}: {count}")

# Accuracy by RNN type
print("\n" + "=" * 80)
print("ACCURACY BY RNN TYPE (All finished runs)")
print("=" * 80)

for rnn_type in df_finished['rnn_type'].unique():
    subset = df_finished[df_finished['rnn_type'] == rnn_type]
    print(f"\n{rnn_type}:")
    print(f"  Count:      {len(subset)}")
    print(f"  Mean Acc:   {subset['best_val_accuracy'].mean():.4f}")
    print(f"  Max Acc:    {subset['best_val_accuracy'].max():.4f}")
    print(f"  Std Dev:    {subset['best_val_accuracy'].std():.4f}")

# Export results
output_file = 'sweep_results_analysis.csv'
df_finished.to_csv(output_file, index=False)
print(f"\n" + "=" * 80)
print(f"Full results exported to: {output_file}")

# Generate refined config suggestion
print("\n" + "=" * 80)
print("SUGGESTED REFINED SWEEP CONFIG")
print("=" * 80)

lr_min = max(top10['lr'].quantile(0.25), 0.0001)  # Ensure reasonable minimum
lr_max = min(top10['lr'].quantile(0.75), 0.1)     # Ensure reasonable maximum
wd_min = max(top10['weight_decay'].quantile(0.25), 0.00001)
wd_max = min(top10['weight_decay'].quantile(0.75), 0.1)
best_rnn = top10['rnn_type'].mode()[0] if len(top10['rnn_type'].mode()) > 0 else 'LSTM'

print(f"""
program: nlp_hw2.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 3
parameters:
  lr:
    distribution: log_uniform_values
    min: {lr_min:.6f}
    max: {lr_max:.6f}
  weight_decay:
    distribution: log_uniform_values
    min: {wd_min:.6f}
    max: {wd_max:.6f}
  rnn_type:
    value: {best_rnn}
""")

print("=" * 80)
print("Analysis complete.")
print(f"\nBest configuration:")
best = df_finished.iloc[0]
print(f"  Run: {best['run_name']}")
print(f"  Accuracy: {best['best_val_accuracy']:.4f}")
print(f"  LR: {best['lr']:.6f}")
print(f"  Weight Decay: {best['weight_decay']:.6f}")
print(f"  RNN Type: {best['rnn_type']}")
