"""
Figures for the paper. Reads results/experiment_results.csv.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/figs", exist_ok=True)
df = pd.read_csv("results/experiment_results.csv")
HEAD = df[df['attacker'].isin(['muMEA-best', 'SA-Q', 'SA-CPM'])].copy()

# ---- Figure 1: grouped bar of ΔARI on each detector, per category ----
def fig_main(df, k=10):
    sub = df[df['k'] == k]
    cats = ['LFR', 'EC-SBM', 'Real']
    budgets = sorted(sub['budget_pct'].unique())
    attackers = ['muMEA-best', 'SA-Q', 'SA-CPM']
    colors = {'muMEA-best': '#888', 'SA-Q': '#1f77b4', 'SA-CPM': '#d62728'}

    fig, axes = plt.subplots(2, len(cats), figsize=(11, 6), sharey='row')
    for j, cat in enumerate(cats):
        c = sub[sub['category'] == cat]
        for i, target in enumerate(['d_ari_q', 'd_ari_cpm']):
            ax = axes[i, j]
            for ai, atk in enumerate(attackers):
                vals = [c[(c['budget_pct']==b) & (c['attacker']==atk)][target].mean()
                        for b in budgets]
                xs = [bi + 0.25 * ai for bi in range(len(budgets))]
                ax.bar(xs, vals, width=0.22, color=colors[atk], label=atk if (i,j)==(0,0) else None)
            ax.set_xticks([bi + 0.25 for bi in range(len(budgets))])
            ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f"{cat} — Δ{'ARI(Q)' if i==0 else 'ARI(CPM)'}")
            if j == 0:
                ax.set_ylabel("Δ ARI (higher = better attack)")
    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle(f"Attack effectiveness, k={k}", y=1.02)
    plt.tight_layout()
    plt.savefig("results/figs/main_comparison.pdf", bbox_inches='tight')
    plt.close()

# ---- Figure 2: cross-detector transfer matrix ----
def fig_transfer(df):
    sub = df[(df['k']==10) & (df['budget_pct']==0.10)]
    attackers = ['muMEA-best', 'SA-Q', 'SA-CPM']
    cats = ['LFR', 'EC-SBM', 'Real']
    fig, axes = plt.subplots(1, len(cats), figsize=(11, 3.4))
    for j, cat in enumerate(cats):
        c = sub[sub['category']==cat]
        mat = []
        for atk in attackers:
            row = c[c['attacker']==atk]
            mat.append([row['d_ari_q'].mean(), row['d_ari_cpm'].mean()])
        ax = axes[j]
        im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.3, vmax=0.5, aspect='auto')
        ax.set_xticks([0,1]); ax.set_xticklabels(['Q', 'CPM'])
        ax.set_yticks(range(len(attackers))); ax.set_yticklabels(attackers)
        ax.set_title(cat); ax.set_xlabel("evaluated on")
        for i,row in enumerate(mat):
            for k_,v in enumerate(row):
                ax.text(k_, i, f"{v:+.2f}", ha='center', va='center',
                        color='black', fontsize=9)
    fig.suptitle("Cross-detector transfer (b=10%, k=10)", y=1.05)
    plt.tight_layout()
    plt.savefig("results/figs/transfer.pdf", bbox_inches='tight')
    plt.close()

# ---- Figure 3: feature importance comparison (from feature_analysis output) ----
# We re-fit here so the figure is self-contained.
def fig_features():
    fd = pd.read_csv("results/feature_dataset.csv")
    feats = ['deg_min','deg_max','common_neighbors','jaccard','adamic_adar',
             'same_pred_comm','cc_u','cc_v','tri_u','tri_v',
             'mu_triad_part_u','mu_triad_part_v','boundary_u','boundary_v']

    means = (fd[fd['y']==1].groupby('attacker')[feats].mean().T)
    if means.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7,5))
        means.plot.barh(ax=ax)
        ax.set_xlabel("mean feature value (positive edges)")
        ax.set_title("Per-attacker mean feature values for chosen edges")
        plt.tight_layout()
        plt.savefig("results/figs/feature_means.pdf", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    fig_main(HEAD, k=10)
    fig_transfer(HEAD)
    if os.path.isfile("results/feature_dataset.csv"):
        fig_features()
    print("Wrote figures to results/figs/")