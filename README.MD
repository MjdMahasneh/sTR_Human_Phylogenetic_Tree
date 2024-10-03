# Human Phylogenetic Tree and Genetic Similarity Analysis using Short Tandem Repeats (STR)

This repository combines two key analyses on genetic data using Short Tandem Repeats (STR):
1. **Ward Linkage Hierarchical Clustering**: Constructs a phylogenetic tree to visualize genetic relationships between individuals.
2. **Genetic Similarity & Distance Matrices**: Calculates and visualizes multiple similarity and distance metrics between individuals based on their genetic profiles.

## Description

- **Hierarchical Clustering (Ward Linkage)**: Uses Ward's method to perform agglomerative clustering, minimizing variance within clusters. The result is displayed as a dendrogram, where the distance between branches reflects genetic dissimilarity.
- **Similarity/Distance Metrics**: Computes and visualizes the following matrices:
  - **Cosine Similarity Matrix**
  - **Euclidean Distance Matrix**
  - **Pearson Correlation Matrix**
  - **Spearman Correlation Matrix**

Both analyses are designed to infer genetic relationships, visualize similarity, and highlight genetic variance between samples.

## Features

- **Hierarchical Clustering**: Constructs a dendrogram annotated with genetic distances between individuals.
- **Similarity & Distance Matrices**: Generates and visualizes key metrics such as cosine similarity, Euclidean distance, and Pearson/Spearman correlations.

## Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`

## Usage

1. **CSV File**: Ensure your genetic data is in a CSV file format.
   - Example : `synthatic_sTR.csv`. Note that this contains synthatic sTR data for demonstration purposes.
2. **Change File Paths**: Update the file paths in the `.py` scripts to match the location of your `.csv` files.

```python
file_path = 'your_file.csv'  # Replace with your file path
```



## Usage
```bash
python clustering_script.py
```
### Running the Clustering Script

```bash
python clustering_script.py
```

This will generate a dendrogram saved as `gene_hierarchy_with_distances.png`.


## Running the Similarity Analysis Script
```bash
python genetic_similarity_analysis.py
```


This will generate the following visualizations:

- **Cosine Similarity Matrix:** `Cosine_Similarity_Matrix.png`
- **Euclidean Distance Matrix:** `Euclidean_Distance_Matrix.png`
- **Pearson Correlation Matrix:** `Pearson_Correlation_Matrix.png`
- **Spearman Correlation Matrix:** `Spearman_Correlation_Matrix.png`


## Output
Dendrogram for hierarchical clustering saved as `gene_hierarchy_with_distances.png`.
Heatmaps for similarity and distance metrics saved as separate PNG files.