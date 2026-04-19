# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# %%
# Load the data
data = pd.read_csv(
    '../Data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    '../Data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())
# %%
# Explore the data

print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata

print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
cancer_type = 'LUAD'  # Lung cancer

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
LUAD_data = data[cancer_samples]

# %%
# Subset by index (genes)
desired_gene_list = ['TP53', 'BRAF', 'KRAS', 'EGFR', 'MYC','PIK3CA','AKT1', 'RB1', 'CDKN2A', 'PTEN','SMAD4','APC','ZEB1','SNAIL','TWIST','SLUG','ZEB2','SOX2','OCT4','KLF4']
gene_list = [gene for gene in desired_gene_list if gene in LUAD_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

LUAD_gene_data = LUAD_data.loc[gene_list]
print(LUAD_gene_data.head())

# %%
# Clustering Data
# KMeans Clustering
X = gene_list
y = LUAD_gene_data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

model = KMeans(n_clusters=20, random_state=0)
model.fit(X)
y_pred = model.predict(X)
plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="Set2", s=100)
plt.xlabel('Genes')
plt.ylabel('LUAD Patients')
plt.title("KMeans Clustering")

plt.show()

# Hierarchical Clustering


# DBSCAN
# %%
