import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data (Parameters from your GBM file) ---
data = pd.read_csv(
    r"C:\Users\dance\OneDrive - University of Virginia\Computational BME\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv", 
    index_col=0, header=0)
metadata_df = pd.read_csv(
    r"C:\Users\dance\OneDrive - University of Virginia\Computational BME\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv", 
    index_col=0, header=0)

# Filter for GBM samples
cancer_type = 'GBM' 
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
GBM_data = data[cancer_samples]

# Define the 12 genes
desired_gene_list = ['STAT1', 'CXCR4', 'PTPN6', 'RHOA', 'STAT3', 'LCK', 'CD86', 'MAPK14', 'HCK', 'PTK2', 'HIF1A', 'PRF1']
gene_list = [gene for gene in desired_gene_list if gene in GBM_data.index]

# --- 2. PREPARATION (Observations = Genes) ---
# We keep Genes as Rows (12 rows) and Samples as Columns (features)
X = GBM_data.loc[gene_list] 
y = gene_list # Our "Target" labels are the Gene Names themselves

# --- 3. PAIR GRID (First 4 Samples as features) ---
# This shows how the 12 genes correlate across the first few patients
df_pair = pd.DataFrame(X.iloc[:, :4]) 
df_pair['Gene'] = y
g = sns.PairGrid(df_pair, hue="Gene", palette="tab20")
g.map(sns.scatterplot)
plt.show()

# --- 4. PCA VISUALIZATION (With 12 Colors) ---
# Step A: Scaling (As seen in the UMAP section of the in-class notebook)
# This prevents one patient with high values from dominating the PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step B: PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
# We use 'hue=y' to make every gene a different color
sns.scatterplot(x=X_pca[:, 0], 
                y=X_pca[:, 1], 
                hue=y, 
                palette="tab20", 
                s=200) # Larger dots to see colors clearly

plt.title("PCA: 12 Genes as Individual Observations")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genes")
plt.show()

# --- 5. K-MEANS CLUSTERING (On the Genes) ---
# This groups the 12 genes into 3 clusters based on expression similarity
model = KMeans(n_clusters=3, random_state=0)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="Set2", s=150)
plt.title("KMeans Clustering of Genes")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# --- 6. HIERARCHICAL CLUSTERING ---
# Using the same plot_dendrogram logic or linkage from class
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z, labels=gene_list) # Label the tree with Gene Names
plt.title("Hierarchical Clustering of 12 Genes")
plt.show()

# --- 7. DBSCAN CLUSTERING ---
dbscan = DBSCAN(eps=5.0, min_samples=2) # EPS adjusted for small sample size (12 points)
y_dbscan = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap="Set2", s=150)
plt.title("DBSCAN Clustering of 12 Genes")
plt.show()
