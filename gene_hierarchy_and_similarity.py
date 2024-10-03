
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pprint


# Load your genetic data from a CSV file
file_path = 'synthatic_sTR.csv'  # Replace with your file path
data = pd.read_csv(file_path, header=0)  # Reads from the first row as headers

# get data
data_samples = data.iloc[:, :]
pprint.pprint(data_samples)

# 1. Cosine Similarity Matrix (for comparing people based on their genetic markers)
cosine_sim_matrix = cosine_similarity(data_samples)

# 2. Euclidean Distance Matrix (for measuring distance between people's genetic profiles)
euclidean_dist_matrix = squareform(pdist(data_samples, metric='euclidean'))

# 3. Pearson Correlation Matrix (to check linear relationships between people)
pearson_corr_matrix = np.corrcoef(data_samples)

# 4. Spearman Correlation Matrix (for non-linear relationships between people)
# Transpose the data to get correlations between rows
spearman_corr_matrix = data_samples.T.corr(method='spearman')





# Visualize and save each matrix separately

# Cosine Similarity
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, cmap='coolwarm', cbar=True)
plt.title('Cosine Similarity Matrix')
plt.tight_layout()
plt.savefig('Cosine_Similarity_Matrix.png')
plt.show()
plt.close()  # Close the plot to avoid overlap

# Euclidean Distance
plt.figure(figsize=(10, 8))
sns.heatmap(euclidean_dist_matrix, cmap='viridis', cbar=True)
plt.title('Euclidean Distance Matrix')
plt.tight_layout()
plt.savefig('Euclidean_Distance_Matrix.png')
plt.show()
plt.close()  # Close the plot to avoid overlap

# Pearson Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr_matrix, cmap='coolwarm', cbar=True)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('Pearson_Correlation_Matrix.png')
plt.show()
plt.close()  # Close the plot to avoid overlap

# Spearman Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, cmap='coolwarm', cbar=True)
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
plt.savefig('Spearman_Correlation_Matrix.png')
plt.show()
plt.close()  # Close the plot to avoid overlap
