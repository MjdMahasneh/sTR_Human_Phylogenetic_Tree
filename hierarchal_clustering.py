import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import re



# Load your genetic data from a CSV file
file_path = r"synthatic_sTR.csv"
data = pd.read_csv(file_path)


# Shorten labels: "family-1" -> "F1", "family-2" -> "F2", etc.
def shorten_label(lbl):
    m = re.search(r'(\d+)', lbl)
    if m:
        return f'F{m.group(1)}'
    return lbl  # fallback: keep as-is if no number found


# Separate the Family label column from numeric STR marker columns
label_col = 'Family'
if label_col in data.columns:
    raw_labels = data[label_col].astype(str).tolist()
    data_samples = data.drop(columns=[label_col])
else:
    # Fall back: use any non-numeric columns as labels, drop them from data
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    raw_labels = data[non_numeric[0]].astype(str).tolist() if non_numeric else [str(i + 1) for i in range(len(data))]
    data_samples = data.select_dtypes(include=[np.number])


labels = [shorten_label(l) for l in raw_labels]

# Ensure all values are numeric; fill any unexpected NaN with 0
data_samples = data_samples.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Clustering {data_samples.shape[0]} samples across {data_samples.shape[1]} STR markers.")
print(f"Sample labels: {labels}")

# Perform hierarchical clustering using Ward's method
dist = pdist(data_samples, metric='euclidean')
linked = linkage(dist, method='ward')  # Ward + Euclidean is the most common combo for STR
# Set color threshold manually — branches below this distance get unique colors
color_thresh = linked[:, 2].max() * 0.5 # cut at 50% of max distance, adjust as needed to get desired number of clusters

# Plot the dendrogram
fig, ax = plt.subplots(figsize=(18, 9))
dendrogram_obj = dendrogram(linked,
                            orientation='top',
                            labels=labels,
                            distance_sort='descending',
                            show_leaf_counts=True,
                            color_threshold=color_thresh,  # <-- controls coloring
                            ax=ax)

# Draw a horizontal line showing where the cut is
ax.axhline(y=color_thresh, color='red', linestyle='--', linewidth=1.5, label=f'Cut threshold: {color_thresh:.1f}')
ax.legend(fontsize=9)
ax.set_title('Hierarchical Clustering Dendrogram (Between People)', fontsize=13)
ax.set_xlabel('Samples', fontsize=11)
ax.set_ylabel('Distance', fontsize=11)

# Rotate x-axis labels and reduce font size so they don't overlap
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

# Annotate branches with their merge distances
for i, d in zip(dendrogram_obj['icoord'], dendrogram_obj['dcoord']):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    ax.text(x, y, f'{y:.2f}', va='bottom', ha='center', fontsize=7)

# Add a legend mapping short labels (F1, F2, ...) back to original family names
label_map = {shorten_label(orig): orig for orig in raw_labels}
legend_text = '\n'.join(f'{short} = {orig}' for short, orig in sorted(label_map.items(),
                         key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0))
fig.text(0.98, 0.5, legend_text, transform=fig.transFigure,
         fontsize=8, verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room for legend on the right
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
