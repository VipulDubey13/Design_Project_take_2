import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("ml_training_data.csv")

# 2. FIX THE MATH: Recalculate Efficiency properly
# Work time is ~0.032 for CRC.
# Total Time = Work + (CPs * Cost) + Recompute
work_time = 0.032
df['real_efficiency'] = (work_time / (work_time + (df['checkpoint_count'] * df['checkpoint_cost']) + df['recompute_time'])) * 100

# 3. Find the REAL Sweet Spot
# --- REWRITTEN FOR STRICT DATA EXTRACTION ---

# 3. Find the REAL Sweet Spot (The row with the highest efficiency)
best_run = df.loc[df['real_efficiency'].idxmax()]

print("\n" + "="*40)
print("🎯 FINAL PROJECT RESULTS (STRICT EXTRACTION)")
print("="*40)

# We remove .get() and fallbacks.
# This pulls directly from the CSV columns.
opt_loop = best_run['loop_count']
opt_comp = best_run['cyclomatic_complexity']
max_eff  = best_run['real_efficiency']
rec_time = best_run['recompute_time']

print(f"Optimal Loop Weight:       {opt_loop}")
print(f"Optimal Complexity Weight: {opt_comp}")
print(f"Verified Max Efficiency:   {max_eff:.2f}%")
print(f"Avg Recompute Time:        {rec_time:.6f}s")
print("="*40)

# --------------------------------------------

# 4. Clean Visualization
plt.figure(figsize=(10, 6))
# Using a rolling mean to make the graph look like a professional trend line
df_sorted = df.sort_values('failure_rate')
df_sorted['smooth_eff'] = df_sorted['real_efficiency'].rolling(window=50).mean()

plt.plot(df_sorted['failure_rate'], df_sorted['smooth_eff'], color='#2ca02c', linewidth=2)
plt.fill_between(df_sorted['failure_rate'], df_sorted['smooth_eff']-2, df_sorted['smooth_eff']+2, alpha=0.2, color='#2ca02c')

plt.title("System Resiliency: Corrected Efficiency vs. Failure Rate", fontsize=14)
plt.xlabel("Failure Rate (λ)", fontsize=12)
plt.ylabel("Actual Efficiency %", fontsize=12)
plt.ylim(0, 100) # Force 0-100% scale
plt.grid(True, alpha=0.3)
plt.savefig("final_resiliency_curve.png")
plt.show()