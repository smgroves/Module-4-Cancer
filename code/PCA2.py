import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def main():
    # For Jupyter, this works better than __file__
    script_dir = Path().resolve()
    data_dir = script_dir / 'data'

    expr_path = data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv'
    metadata_path = data_dir / 'TRAINING_SET_GSE62944_metadata.csv'

    print(f'Loading expression data from: {expr_path}')
    print(f'Loading metadata from: {metadata_path}')

    # Load data
    data = pd.read_csv(expr_path, index_col=0, header=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0, header=0)

    print('\nExpression data shape:', data.shape)
    print('Metadata shape:', metadata_df.shape)

    # -----------------------------
    # Step 1: Filter to LUSC samples
    # -----------------------------
    cancer_type = 'LUSC'
    cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
    shared_samples = cancer_samples.intersection(data.columns)

    if len(shared_samples) == 0:
        raise ValueError('No matching sample IDs found between metadata and expression data.')

    LUSC_data = data.loc[:, shared_samples]
    print(f'\nNumber of shared LUSC samples: {len(shared_samples)}')
    print('LUSC expression matrix shape:', LUSC_data.shape)

    # -----------------------------
    # Step 2: Use top variable genes
    # -----------------------------
    n_top_genes = 500

    # Drop genes with all missing values before variance calculation
    LUSC_data_clean = LUSC_data.dropna(axis=0, how='all')

    # Variance across samples for each gene
    gene_variances = LUSC_data_clean.var(axis=1)

    # Drop genes where variance could not be computed
    gene_variances = gene_variances.dropna()

    # Select top variable genes
    top_genes = gene_variances.sort_values(ascending=False).head(n_top_genes).index.tolist()

    if len(top_genes) < 2:
        raise ValueError('Not enough variable genes found for PCA.')

    print(f'\nUsing top {len(top_genes)} most variable genes for PCA/clustering')
    print('First 10 genes:', top_genes[:10])

    # Subset expression matrix to top variable genes
    LUSC_gene_data = LUSC_data_clean.loc[top_genes]

    # -----------------------------
    # Step 3: Merge metadata
    # -----------------------------
    LUSC_metadata = metadata_df.loc[shared_samples].copy()

    LUSC_merged = LUSC_gene_data.T.merge(
        LUSC_metadata,
        left_index=True,
        right_index=True
    )

    print('\nMerged data shape:', LUSC_merged.shape)

    # -----------------------------
    # Step 4: Prepare PCA matrix
    # -----------------------------
    X = LUSC_merged[top_genes].copy()

    # Remove samples with any missing values in selected genes
    X = X.dropna(axis=0)

    # Keep metadata aligned
    LUSC_merged = LUSC_merged.loc[X.index].copy()

    print('PCA input matrix shape:', X.shape)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Step 5: PCA
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    LUSC_merged['PC1'] = X_pca[:, 0]
    LUSC_merged['PC2'] = X_pca[:, 1]

    print('\nExplained variance ratio:')
    print(f'PC1: {pca.explained_variance_ratio_[0]:.4f}')
    print(f'PC2: {pca.explained_variance_ratio_[1]:.4f}')
    print(f'Total: {pca.explained_variance_ratio_.sum():.4f}')

    # Average expression across top variable genes
    LUSC_merged['avg_expression'] = X.mean(axis=1)

    sns.set(style='whitegrid')

    # -----------------------------
    # Plot 1: Gender
    # -----------------------------
    if 'gender' in LUSC_merged.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='gender',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by Gender')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plot 2: Average expression
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=LUSC_merged,
        x='PC1',
        y='PC2',
        hue='avg_expression',
        s=90
    )
    plt.title('PCA of LUSC Samples Colored by Average Expression')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 3: Tumor stage simplified
    # -----------------------------
    stage_col = 'ajcc_pathologic_tumor_stage'

    if stage_col in LUSC_merged.columns:
        def simplify_stage(x):
            x = str(x).upper()

            if 'IV' in x:
                return 'Stage IV'
            elif 'III' in x:
                return 'Stage III'
            elif 'II' in x:
                return 'Stage II'
            elif 'I' in x:
                return 'Stage I'
            else:
                return 'Unknown'

        LUSC_merged['stage_simple'] = LUSC_merged[stage_col].apply(simplify_stage)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='stage_simple',
            palette='Set2',
            s=90
        )
        plt.title('PCA Colored by Tumor Stage (Simplified)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plot 4: Race cleaned
    # -----------------------------
    race_col = 'race'

    if race_col in LUSC_merged.columns:
        def clean_race(x):
            x = str(x).lower()
            if 'white' in x:
                return 'White'
            elif 'black' in x:
                return 'Black'
            elif 'asian' in x:
                return 'Asian'
            elif 'native' in x:
                return 'Native'
            else:
                return 'Other/Unknown'

        LUSC_merged['race_clean'] = LUSC_merged[race_col].apply(clean_race)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='race_clean',
            palette='Set2',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by Race')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plot 5: Thyroid disease history
    # -----------------------------
    thyroid_col = 'history_thyroid_disease'

    if thyroid_col in LUSC_merged.columns:
        def clean_thyroid(x):
            x = str(x).lower()
            if x in ['yes', 'y']:
                return 'Yes'
            elif x in ['no', 'n']:
                return 'No'
            else:
                return 'Unknown'

        LUSC_merged['thyroid_clean'] = LUSC_merged[thyroid_col].apply(clean_thyroid)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='thyroid_clean',
            palette='Set1',
            s=90
        )
        plt.title('PCA Colored by Thyroid Disease History')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()
    # -----------------------------
    # Plot: DSS
    # -----------------------------
    dss_col = 'DSS'

    if dss_col in LUSC_merged.columns:
        LUSC_merged[dss_col] = LUSC_merged[dss_col].astype(str)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue=dss_col,
            palette='Set1',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by DSS')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping DSS plot because '{dss_col}' not found.")

            # -----------------------------
    # Plot: DFI
    # -----------------------------
    dfi_col = 'DFI'

    if dfi_col in LUSC_merged.columns:
        LUSC_merged[dfi_col] = LUSC_merged[dfi_col].astype(str)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue=dfi_col,
            palette='Set2',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by DFI')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping DFI plot because '{dfi_col}' not found.")

            # -----------------------------
    # Plot: PFI
    # -----------------------------
    pfi_col = 'PFI'

    if pfi_col in LUSC_merged.columns:
        LUSC_merged[pfi_col] = LUSC_merged[pfi_col].astype(str)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue=pfi_col,
            palette='Set3',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by PFI')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping PFI plot because '{pfi_col}' not found.")
        from umap import UMAP

# -----------------------------
# Step 7: UMAP
# -----------------------------
from umap import UMAP
umap_model = UMAP(n_components=2, random_state=0)
X_umap = umap_model.fit_transform(X_scaled)

LUSC_merged['UMAP1'] = X_umap[:, 0]
LUSC_merged['UMAP2'] = X_umap[:, 1]

# Plot UMAP (clusters)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=LUSC_merged,
    x='UMAP1',
    y='UMAP2',
    hue='cluster',
    palette='Set2',
    s=90
)
plt.title('UMAP of LUSC Samples (KMeans Clusters)')
plt.tight_layout()
plt.show()
    # -----------------------------
    # Step 6: KMeans clustering
    # -----------------------------
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

LUSC_merged['cluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=LUSC_merged,
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='Set2',
    s=90
    )
plt.title('KMeans Clustering (k=3)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
plt.tight_layout()
plt.show()


    # -----------------------------
    # Save results
    # -----------------------------
    output_path = script_dir / 'LUSC_PCA_results_top_variable_genes.csv'
    LUSC_merged.to_csv(output_path)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()