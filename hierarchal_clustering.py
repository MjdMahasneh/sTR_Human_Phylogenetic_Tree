import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns




# Ward's Method:
#
# Ward's method minimizes the variance within clusters. It's an agglomerative clustering technique that merges clusters with the smallest increase in total within-cluster variance.
#
# Distance in Ward's Method:
#
# The distance values in your dendrogram represent the Euclidean distance between clusters. As the algorithm progresses, it merges clusters that are closest in terms of this distance.
# The height of the branches in the dendrogram shows the dissimilarity (distance) between merged clusters. The larger the height, the more different the clusters are.


# Load your genetic data from a CSV file
file_path = 'synthatic_sTR.csv'  # Replace with your file path
data = pd.read_csv(file_path)  # Reads from the first row as headers

# Check for non-finite values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isfinite(data.iloc[i, j]):
            print(f"Non-finite value found at row {i}, column {j}: {data.iloc[i, j]}")

# Drop rows with missing values
data_samples = data.iloc[:, :].dropna()

# Generate labels that match the original row numbers from the CSV (Python starts from 0)
labels = [str(i+1) for i in range(data_samples.shape[0])]

# Perform hierarchical clustering (using the 'ward' method)
linked = linkage(data_samples, method='ward')

# Plot the dendrogram with labels
plt.figure(figsize=(15, 7))
dendrogram_obj = dendrogram(linked,
                            orientation='top',
                            labels=labels,  # Add labels to match CSV row numbers
                            distance_sort='descending',
                            show_leaf_counts=True)

plt.title('Hierarchical Clustering Dendrogram (Between People)')
plt.xlabel('People (Samples)')
plt.ylabel('Distance')

# Annotate the plot with the distances between clusters (similarity)
for i, d, c in zip(dendrogram_obj['icoord'], dendrogram_obj['dcoord'], dendrogram_obj['ivl']):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    plt.text(x, y, f'{y:.2f}', va='bottom', ha='center')

# Save the plot
plt.savefig('gene_hierarchy_with_distances.png')
plt.tight_layout()
plt.show()
