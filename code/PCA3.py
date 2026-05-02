import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP


def main():
    # -----------------------------
    # Paths (works in Jupyter)
    # -----------------------------
    script_dir = Path().resolve()
    data_dir = script_dir / 'data'

    expr_path = data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv'
    metadata_path = data_dir / 'TRAINING_SET_GSE62944_metadata.csv'

    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(expr_path, index_col=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    print("Expression shape:", data.shape)
    print("Metadata shape:", metadata_df.shape)

    # -----------------------------
    # Filter LUSC samples
    # -----------------------------
    cancer_samples = metadata_df[metadata_df['cancer_type'] == 'LUSC'].index
    shared_samples = cancer_samples.intersection(data.columns)

    LUSC_data = data.loc[:, shared_samples]
    print("LUSC shape:", LUSC_data.shape)

    # -----------------------------
    # Top variable genes
    # -----------------------------
    n_top_genes = 500

    LUSC_data = LUSC_data.dropna(axis=0, how='all')
    gene_variance = LUSC_data.var(axis=1).dropna()

    top_genes = gene_variance.sort_values(ascending=False).head(n_top_genes).index

    print("Using top genes:", len(top_genes))

    X = LUSC_data.loc[top_genes].T

    # Drop NA samples
    X = X.dropna(axis=0)

    # -----------------------------
    # Scale
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # PCA
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'avg_expression': X.mean(axis=1)
    }, index=X.index)

    # Merge with metadata
    LUSC_merged = df_plot.merge(
        metadata_df.loc[X.index],
        left_index=True,
        right_index=True
    )

    print("Explained variance:", pca.explained_variance_ratio_)

    sns.set(style='whitegrid')

    # PCA plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=LUSC_merged,
        x='PC1',
        y='PC2',
        hue='avg_expression',
        s=90
    )
    plt.title('PCA (Top Variable Genes, Colored by Avg Expression)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.tight_layout()
    plt.show()

        # -----------------------------
    # Plot: Age at diagnosis
    # -----------------------------
    age_col = 'age_at_diagnosis'

    if age_col in LUSC_merged.columns:
        # Ensure numeric
        LUSC_merged[age_col] = pd.to_numeric(
            LUSC_merged[age_col], errors='coerce'
        )

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue=age_col,
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by Age at Diagnosis')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping age plot because '{age_col}' not found.")

    # -----------------------------
    # UMAP
    # -----------------------------
    umap_model = UMAP(n_components=2, random_state=0)
    X_umap = umap_model.fit_transform(X_scaled)

    LUSC_merged['UMAP1'] = X_umap[:, 0]
    LUSC_merged['UMAP2'] = X_umap[:, 1]

    # UMAP plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=LUSC_merged,
        x='UMAP1',
        y='UMAP2',
        hue='avg_expression',
        s=90
    )
    plt.title('UMAP (Top Variable Genes, Colored by Avg Expression)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()