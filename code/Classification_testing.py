# Import packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# %%
# Load the data
####################################################
## data = pd.read_csv(
##    'C:\\Users\\yqr8pz\\Documents\\BME 2315\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
## metadata_df = pd.read_csv(
 ##   'C:\\Users\\yqr8pz\\Documents\\BME 2315\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
data = pd.read_csv(
    'C:\\Users\\Jmarc\\Desktop\\Comp BME\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    'C:\\Users\\Jmarc\\Desktop\\Comp BME\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_type = 'LUAD'

cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
LUAD_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################
desired_gene_list = ['EGFR', 'JAK1', 'JAK2', 'MTOR', 'PIK3CA', 'PIK3CB']
gene_list = [gene for gene in desired_gene_list if gene in LUAD_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

LUAD_gene_data = LUAD_data.loc[gene_list]
print(LUAD_gene_data.head())

gene_summary = pd.DataFrame({
    "non_missing_count": LUAD_gene_data.count(axis=1),
    "missing_count": LUAD_gene_data.isna().sum(axis=1)
})

# %%
# Basic statistics on the subsetted data
####################################################
print(LUAD_gene_data.describe())
print(LUAD_gene_data.var(axis=1))
print(LUAD_gene_data.mean(axis=1))
print(LUAD_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
print(metadata_df.groupby('cancer_type')["ajcc_pathologic_tumor_stage"].value_counts())

metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby('cancer_type')["age_at_diagnosis"].mean())

# %%
# Merging datasets
####################################################
LUAD_metadata = metadata_df.loc[cancer_samples]
LUAD_merged = LUAD_gene_data.T.merge(
    LUAD_metadata, left_index=True, right_index=True)
print(LUAD_merged.head())


# %%
## Generative Ai was used to help write the following code. (Claude, 2026)
# Load gene list from hallmarks file
####################################################
##gene_list = pd.read_csv(
##    'C:\\Users\\yqr8pz\\Documents\\BME 2315\\Module-4-Cancer\\Menyhart_JPA_CancerHallmarks_core.txt',
##    sep='\t', header=None, index_col=0)
gene_list = pd.read_csv(r'C:\Users\Jmarc\Desktop\Comp BME\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt', sep='\t', header=None, index_col=0)
print(gene_list)

immune_list = list(gene_list.loc['EVADING IMMUNE DESTRUCTION'])
angio_list  = list(gene_list.loc['SUSTAINED ANGIOGENESIS'])

immune_list = [g for g in immune_list if pd.notna(g)]
angio_list  = [g for g in angio_list  if pd.notna(g)]

all_genes = immune_list + angio_list
print(f"Immune genes: {immune_list}")
print(f"Angiogenesis genes: {angio_list}")

# %%
# Subset to LUAD + filter to genes present in dataset
####################################################
cancer_samples = metadata_df[metadata_df['cancer_type'] == 'LUAD'].index
LUAD_data      = data[cancer_samples]
LUAD_metadata  = metadata_df.loc[cancer_samples].copy()

gene_list_found = [g for g in all_genes if g in LUAD_data.index]
missing_genes   = [g for g in all_genes if g not in LUAD_data.index]
print(f"\nGenes found in dataset: {len(gene_list_found)}")
print(f"Genes missing from dataset: {missing_genes}")

def simplify_stage(stage):
    if pd.isna(stage): return None
    s = str(stage)
    if 'IV'  in s: return 'Stage IV'
    if 'III' in s: return 'Stage III'
    if 'II'  in s: return 'Stage II'
    if 'I'   in s: return 'Stage I'
    return None

LUAD_metadata['simple_stage'] = LUAD_metadata['ajcc_pathologic_tumor_stage'].apply(simplify_stage)
LUAD_metadata_clean = LUAD_metadata.dropna(subset=['simple_stage'])
clean_samples = LUAD_metadata_clean.index

LUAD_gene_data = LUAD_data.loc[gene_list_found, clean_samples]
LUAD_merged    = LUAD_gene_data.T.merge(
    LUAD_metadata_clean[['simple_stage']], left_index=True, right_index=True)

X = LUAD_merged[gene_list_found].values
X = SimpleImputer(strategy='mean').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)

stage_labels = LUAD_merged['simple_stage'].values
stage_order  = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
palette      = {'Stage I': '#4CAF50', 'Stage II': '#2196F3',
                'Stage III': '#FF9800', 'Stage IV': '#F44336'}

# %%
# PCA — Combined Immune Evasion + Angiogenesis Genes
####################################################
# PCA reduces our many genes down to 2 principal components so we can plot and visually
# inspect whether samples with similar gene expression group together by tumor stage.
# PC1 and PC2 capture the directions of greatest variance in the data.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Print variance explained by each component
print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=[list(palette.values())[stage_order.index(s)] for s in stage_labels],
            s=60, alpha=0.8)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette[s], label=s) for s in stage_order]


# %%
# PCA — Immune Evasion Genes Only
####################################################
immune_found = [g for g in immune_list if g in gene_list_found]
X_immune = LUAD_merged[immune_found].values
X_immune = SimpleImputer(strategy='mean').fit_transform(X_immune)
X_immune_scaled = StandardScaler().fit_transform(X_immune)

# Limit components to what's feasible given the number of genes
n_components_immune = min(2, len(immune_found))
pca_immune = PCA(n_components=n_components_immune, random_state=42)
X_pca_immune = pca_immune.fit_transform(X_immune_scaled)



# %%
# PCA — Angiogenesis Genes Only
####################################################
angio_found = [g for g in angio_list if g in gene_list_found]
X_angio = LUAD_merged[angio_found].values
X_angio = SimpleImputer(strategy='mean').fit_transform(X_angio)
X_angio_scaled = StandardScaler().fit_transform(X_angio)

n_components_angio = min(2, len(angio_found))
pca_angio = PCA(n_components=n_components_angio, random_state=42)
X_pca_angio = pca_angio.fit_transform(X_angio_scaled)


# %%
# Logistic Regression — Combined Immune + Angiogenesis Genes
####################################################
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Encode stage labels to integers (Stage I=0, II=1, III=2, IV=3)
le = LabelEncoder()
le.fit(stage_order)  # use stage_order so encoding is always consistent
y = le.transform(stage_labels)

# Train/test split (stratify keeps class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# %%
# BUILD MODEL
####################################################
lr_model = LogisticRegression(
    penalty=None,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=1000
).fit(X_train, y_train)

# %%
# EVALUATE
####################################################
# Cross-validation on full dataset
cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Detailed per-class report on test set
print("\nClassification Report:")
print(classification_report(y_test, lr_model.predict(X_test),
                             target_names=le.classes_))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    lr_model, X_test, y_test,
    display_labels=le.classes_, ax=ax, colorbar=False)
plt.title("Logistic Regression — Confusion Matrix\n(Immune + Angiogenesis Genes)")
plt.tight_layout()
plt.savefig('confusion_matrix_lr.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# FEATURE IMPORTANCE (coefficients per stage)
####################################################
# %%
# FEATURE IMPORTANCE (coefficients per stage)
####################################################

# Use the actual number of features the model was trained on — not gene_list_found
n_features = lr_model.coef_.shape[1]
print(f"Model trained on {n_features} features")
print(f"gene_list_found has {len(gene_list_found)} genes")

# Rebuild the correct gene list from the merged dataframe columns
# X_scaled was built from LUAD_merged[gene_list_found], but imputer may have changed shape
# So we just label by index if there's a mismatch
if n_features == len(gene_list_found):
    feature_names = gene_list_found
else:
    # Fallback: get column names directly from the merged dataframe
    feature_names = [col for col in LUAD_merged.columns if col in gene_list_found]
    if len(feature_names) != n_features:
        # Last resort: just number them
        feature_names = [f"gene_{i}" for i in range(n_features)]
        print("Warning: could not match gene names to coefficients, using indices")

coef_df = pd.DataFrame(
    lr_model.coef_,
    index=le.classes_,
    columns=feature_names
)

# Plot top 10 most influential genes per stage
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, stage in zip(axes.flatten(), le.classes_):
    top_genes = coef_df.loc[stage].abs().nlargest(10).index
    vals = coef_df.loc[stage, top_genes]
    colors = ['#F44336' if v > 0 else '#2196F3' for v in vals]
    ax.barh(top_genes, vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"Top Genes — {stage}")
    ax.set_xlabel("Coefficient")

plt.suptitle("Logistic Regression Feature Importance by Stage", fontsize=13)
plt.tight_layout()
plt.savefig('feature_importance_lr.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# DECISION BOUNDARY via PCA (visualize on 2D PCA space)
####################################################
# Train a NEW logistic regression on the 2D PCA projection for visualization only
lr_2d = LogisticRegression(
    penalty=None, solver='lbfgs', multi_class='multinomial', max_iter=1000
).fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.25,
             colors=[palette[s] for s in stage_order])
plt.contour(xx, yy, Z, colors='black', linewidths=0.8)

for stage in stage_order:
    mask = stage_labels == stage
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=stage, color=palette[stage],
                s=50, alpha=0.8, edgecolors='k', linewidths=0.4)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Logistic Regression Decision Boundary\n(Visualized in PCA Space)")
plt.legend(title='Tumor Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('decision_boundary_lr.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
