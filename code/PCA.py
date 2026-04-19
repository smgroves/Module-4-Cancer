import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'

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

    # -----------------------------
    # Step 2: Define gene set
    # -----------------------------
    desired_gene_list = [
        'ALK', 'BRAF', 'CDK-4', 'BCL-2', 'CSF-1R', 'EGFR', 'FLT3',
        'FGFR', 'JAK', 'KIT', 'MEK', 'mTOR', 'NTRK', 'PI3K',
        'ROS1', 'SMO', 'XPO1', 'VEGFR', 'BTK'
    ]

    alias_map = {
        'CDK-4': 'CDK4',
        'BCL-2': 'BCL2'
    }

    gene_list = []
    for gene in desired_gene_list:
        actual_gene = alias_map.get(gene, gene)
        if actual_gene in LUSC_data.index:
            gene_list.append(actual_gene)
        else:
            print(f"Warning: {gene} not found")

    if len(gene_list) < 2:
        raise ValueError("Need at least 2 genes")

    print('\nGenes used:', gene_list)

    LUSC_gene_data = LUSC_data.loc[gene_list]

    # -----------------------------
    # Step 3: Merge metadata
    # -----------------------------
    LUSC_metadata = metadata_df.loc[shared_samples].copy()

    LUSC_merged = LUSC_gene_data.T.merge(
        LUSC_metadata, left_index=True, right_index=True
    )

    # -----------------------------
    # Step 4: Prepare PCA
    # -----------------------------
    X = LUSC_merged[gene_list].copy()
    X = X.dropna(axis=0)

    LUSC_merged = LUSC_merged.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Step 5: PCA
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    LUSC_merged['PC1'] = X_pca[:, 0]
    LUSC_merged['PC2'] = X_pca[:, 1]

    print('\nExplained variance:')
    print(pca.explained_variance_ratio_)

    # Avg expression
    LUSC_merged['avg_expression'] = X.mean(axis=1)

    sns.set(style='whitegrid')

    # -----------------------------
    # Plot 1: Gender
    # -----------------------------
    if 'gender' in LUSC_merged.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue='gender')
        plt.title('PCA by Gender')
        plt.show()

    # -----------------------------
    # Plot 2: ALK
    # -----------------------------
    key_gene = 'ALK' if 'ALK' in gene_list else gene_list[0]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue=key_gene)
    plt.title(f'PCA by {key_gene}')
    plt.show()

    # -----------------------------
    # Plot 3: Avg expression
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue='avg_expression')
    plt.title('PCA by Avg Gene Expression')
    plt.show()

    # -----------------------------
    # Step 6: KMeans Clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X_scaled)

    LUSC_merged['cluster'] = kmeans.labels_

    # Plot clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=LUSC_merged,
        x='PC1',
        y='PC2',
        hue='cluster',
        palette='Set2'
    )
    plt.title('KMeans Clustering (k=3)')
    plt.show()

    # Save
    output_path = script_dir / 'LUSC_PCA_results.csv'
    LUSC_merged.to_csv(output_path)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()