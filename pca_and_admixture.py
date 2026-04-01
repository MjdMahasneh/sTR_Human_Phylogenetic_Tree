"""
PCA (Principal Component Analysis) and STRUCTURE/ADMIXTURE Analysis for Y-STR Data
=====================================================================================

HOW TO INTERPRET THE PCA PLOT:
--------------------------------
  • Each dot = one sample (person/family).
  • The X-axis (PC1) explains the MOST genetic variation in your data.
    PC2 explains the second most, etc.
  • Samples that are CLOSE together on the plot are genetically SIMILAR.
  • Samples that are FAR apart are genetically DIFFERENT.
  • Clusters of dots = groups of people who share similar STR profiles
    (likely related or from the same haplogroup/population).
  • The % in axis labels (e.g. "PC1 (38.5%)") tells you how much of the
    total variation that component captures. Higher % = more informative axis.
  • If PC1 + PC2 together explain >50% of variance, the 2D plot is a good
    representation of your data.
  • Outliers far from all clusters may represent unique/rare lineages.

HOW TO INTERPRET THE ADMIXTURE/STRUCTURE PLOT:
-----------------------------------------------
  • Each VERTICAL BAR = one sample.
  • Each COLOUR in a bar = one inferred ancestral component (K).
  • A bar that is 100% one colour = that sample is "pure" from that component.
  • A bar that is MIXED colours = that sample has mixed ancestry from multiple
    components (admixed individual).
  • K (number of components) is chosen by you. Try K=2,3,4... and see which
    grouping is most biologically meaningful.
  • Since true STRUCTURE software needs MCMC, we approximate it here using
    NMF (Non-negative Matrix Factorisation) which gives very similar bar plots
    and is standard in many genomics workflows.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF


# =============================================================================
# CONFIG  — change the file path and K values here
# =============================================================================
FILE_PATH = r"C:\Users\maj\Desktop\Samples_STR\tableConvert.com_nsp9tx.csv"

# K values to try for the ADMIXTURE / STRUCTURE bar chart
# Each K will produce its own saved image.
K_VALUES = [2, 3, 4, 5]

# Output filenames
PCA_OUTPUT    = "PCA_plot.png"
ADM_OUTPUT    = "Admixture_K{k}.png"   # {k} is replaced automatically
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = pd.read_csv(FILE_PATH)

label_col = 'Family'
if label_col in data.columns:
    raw_labels = data[label_col].astype(str).tolist()
    data_samples = data.drop(columns=[label_col])
else:
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    raw_labels = (data[non_numeric[0]].astype(str).tolist()
                  if non_numeric else [str(i + 1) for i in range(len(data))])
    data_samples = data.select_dtypes(include=[np.number])

data_samples = data_samples.apply(pd.to_numeric, errors='coerce').fillna(0)


def shorten_label(lbl):
    m = re.search(r'(\d+)', lbl)
    return f'F{m.group(1)}' if m else lbl

labels      = [shorten_label(l) for l in raw_labels]
n_samples   = data_samples.shape[0]
n_markers   = data_samples.shape[1]

print(f"Loaded {n_samples} samples × {n_markers} STR markers.")
print(f"Labels: {labels}")


# ---------------------------------------------------------------------------
# 2.  Colour palette — one colour per unique label
# ---------------------------------------------------------------------------
unique_labels = list(dict.fromkeys(labels))   # preserve order, deduplicate
cmap = plt.get_cmap('tab20', len(unique_labels))
label_colour  = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
point_colours = [label_colour[l] for l in labels]


# ---------------------------------------------------------------------------
# 3. PCA
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_samples)

n_components = min(n_samples, n_markers, 10)   # up to 10 PCs
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100   # in percent

# --- Plot 1: PC1 vs PC2 scatter ---
fig, ax = plt.subplots(figsize=(10, 7))

for i, (x, y, lbl) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], labels)):
    ax.scatter(x, y, color=label_colour[lbl], s=120, zorder=3,
               edgecolors='black', linewidths=0.5)
    ax.text(x, y + 0.015 * (X_pca[:, 1].max() - X_pca[:, 1].min()),
            lbl, ha='center', va='bottom', fontsize=8)

ax.set_xlabel(f'PC1  ({explained[0]:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2  ({explained[1]:.1f}% variance)', fontsize=12)
ax.set_title('PCA of Y-STR Profiles', fontsize=14)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
ax.grid(True, linestyle=':', alpha=0.4)

# Legend
patches = [mpatches.Patch(color=label_colour[l], label=f'{l} = {raw_labels[labels.index(l)]}')
           for l in unique_labels]
ax.legend(handles=patches, fontsize=7, loc='best',
          title='Samples', title_fontsize=8,
          framealpha=0.8, ncol=max(1, len(unique_labels) // 12))

plt.tight_layout()
plt.savefig(PCA_OUTPUT, dpi=150, bbox_inches='tight')
print(f"Saved PCA plot → {PCA_OUTPUT}")
plt.show()

# --- Plot 2: Scree plot (how much variance each PC explains) ---
fig2, ax2 = plt.subplots(figsize=(8, 4))
pc_nums = np.arange(1, len(explained) + 1)
ax2.bar(pc_nums, explained, color='steelblue', edgecolor='white')
ax2.plot(pc_nums, np.cumsum(explained), 'r-o', markersize=5, label='Cumulative')
ax2.set_xlabel('Principal Component', fontsize=11)
ax2.set_ylabel('Variance Explained (%)', fontsize=11)
ax2.set_title('PCA Scree Plot', fontsize=13)
ax2.set_xticks(pc_nums)
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
scree_out = "PCA_scree.png"
plt.savefig(scree_out, dpi=150, bbox_inches='tight')
print(f"Saved Scree plot  → {scree_out}")
plt.show()


# ---------------------------------------------------------------------------
# 4. ADMIXTURE / STRUCTURE  (NMF approximation)
# ---------------------------------------------------------------------------
# NMF requires non-negative values. StandardScaler can give negatives,
# so we use MinMax scaling (shift to [0,1]) instead.
from sklearn.preprocessing import MinMaxScaler
X_nn = MinMaxScaler().fit_transform(data_samples)

# Sort samples by label so the bar chart looks tidy
sort_idx  = sorted(range(n_samples), key=lambda i: labels[i])
sorted_labels = [labels[i] for i in sort_idx]
X_sorted  = X_nn[sort_idx]

for K in K_VALUES:
    nmf = NMF(n_components=K, init='nndsvda', max_iter=1000, random_state=42)
    W   = nmf.fit_transform(X_sorted)   # shape: (n_samples, K)

    # Normalise rows so each bar sums to 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid division by zero
    W_norm   = W / row_sums

    # Choose K colours from a qualitative palette
    if K <= 9:
        colours = [plt.get_cmap('Set1')(i / 9.0) for i in range(K)]
    else:
        colours = [plt.get_cmap('tab20')(i / K) for i in range(K)]

    fig3, ax3 = plt.subplots(figsize=(max(12, int(n_samples * 0.5)), 4))

    bottoms = np.zeros(n_samples)
    for k in range(K):
        ax3.bar(range(n_samples), W_norm[:, k], bottom=bottoms,
                color=colours[k], width=0.85,
                label=f'Component {k + 1}')
        bottoms += W_norm[:, k]

    ax3.set_xticks(range(n_samples))
    ax3.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=9)
    ax3.set_xlim(-0.5, n_samples - 0.5)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Ancestry Proportion', fontsize=11)
    ax3.set_xlabel('Sample', fontsize=11)
    ax3.set_title(f'ADMIXTURE / STRUCTURE plot  (K = {K})\n'
                  f'Each colour = one inferred ancestral component', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8, ncol=K,
               framealpha=0.8, title='Ancestral components')

    out_name = ADM_OUTPUT.format(k=K)
    plt.tight_layout()
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"Saved Admixture K={K} → {out_name}")
    plt.show()

print("\nAll done!")

