import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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
        'BCL-2': 'BCL2',
        'mTOR': 'MTOR'
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

    # -----------------------------
    # Step 5b: PCA loadings
    # -----------------------------
    loadings = pd.DataFrame(
        pca.components_.T,
        index=gene_list,
        columns=['PC1', 'PC2']
    )

    loadings['PC1_abs'] = loadings['PC1'].abs()
    loadings['PC2_abs'] = loadings['PC2'].abs()

    print('\nPCA Loadings:')
    print(loadings[['PC1', 'PC2']])

    print('\nTop positive PC1 genes:')
    print(loadings.sort_values('PC1', ascending=False)[['PC1']].head(5))

    print('\nTop negative PC1 genes:')
    print(loadings.sort_values('PC1', ascending=True)[['PC1']].head(5))

    print('\nTop positive PC2 genes:')
    print(loadings.sort_values('PC2', ascending=False)[['PC2']].head(5))

    print('\nTop negative PC2 genes:')
    print(loadings.sort_values('PC2', ascending=True)[['PC2']].head(5))

    # Save loadings
    loadings_path = script_dir / 'LUSC_PCA_loadings.csv'
    loadings.to_csv(loadings_path)
    print(f'\nSaved loadings to: {loadings_path}')

    sns.set(style='whitegrid')

    # -----------------------------
    # Plot 1: Gender
    # -----------------------------
    if 'gender' in LUSC_merged.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue='gender')
        plt.title('PCA by Gender')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plot 2: ALK
    # -----------------------------
    key_gene = 'ALK' if 'ALK' in gene_list else gene_list[0]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue=key_gene)
    plt.title(f'PCA by {key_gene}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 3: Avg expression
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=LUSC_merged, x='PC1', y='PC2', hue='avg_expression')
    plt.title('PCA by Avg Gene Expression')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 4: Top absolute loadings for PC1
    # -----------------------------
    top_pc1 = loadings.sort_values('PC1_abs', ascending=False).head(10).copy()
    top_pc1 = top_pc1.sort_values('PC1')

    plt.figure(figsize=(8, 6))
    plt.barh(top_pc1.index, top_pc1['PC1'])
    plt.title('Top 10 PCA Loadings for PC1')
    plt.xlabel('Loading')
    plt.ylabel('Gene')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 5: Top absolute loadings for PC2
    # -----------------------------
    top_pc2 = loadings.sort_values('PC2_abs', ascending=False).head(10).copy()
    top_pc2 = top_pc2.sort_values('PC2')

    plt.figure(figsize=(8, 6))
    plt.barh(top_pc2.index, top_pc2['PC2'])
    plt.title('Top 10 PCA Loadings for PC2')
    plt.xlabel('Loading')
    plt.ylabel('Gene')
    plt.tight_layout()
    plt.show()

        # -----------------------------
    # Gradient Descent Regression:
    # Predict age_at_diagnosis from PC1 and PC2
    # -----------------------------
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    age_col = 'age_at_diagnosis'

    if age_col in LUSC_merged.columns:
        LUSC_merged[age_col] = pd.to_numeric(LUSC_merged[age_col], errors='coerce')

        age_df = LUSC_merged[['PC1', 'PC2', age_col]].dropna().copy()

        if len(age_df) > 10:
            X_ml = age_df[['PC1', 'PC2']].values
            y_ml = age_df[age_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X_ml, y_ml, test_size=0.2, random_state=42
            )

            # standardize features only
            X_train_mean = X_train.mean(axis=0)
            X_train_std = X_train.std(axis=0)
            X_train_std[X_train_std == 0] = 1

            X_train_scaled = (X_train - X_train_mean) / X_train_std
            X_test_scaled = (X_test - X_train_mean) / X_train_std

            # initialize parameters explicitly
            w = np.zeros(X_train_scaled.shape[1])   # weights for PC1, PC2
            b = 0.0                                 # intercept

            learning_rate = 0.01
            n_iterations = 2000
            m = len(X_train_scaled)

            # gradient descent
            for _ in range(n_iterations):
                y_pred_train = X_train_scaled @ w + b

                dw = (2 / m) * (X_train_scaled.T @ (y_pred_train - y_train))
                db = (2 / m) * np.sum(y_pred_train - y_train)

                w -= learning_rate * dw
                b -= learning_rate * db

            # predictions
            y_pred = X_test_scaled @ w + b

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print('\nGradient Descent Linear Regression: Predict Age from PC1 + PC2')
            print(f'Intercept: {b:.4f}')
            print(f'PC1 weight: {w[0]:.4f}')
            print(f'PC2 weight: {w[1]:.4f}')
            print(f'MSE: {mse:.4f}')
            print(f'R^2: {r2:.4f}')

            # actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, s=80)
            plt.xlabel('Actual Age at Diagnosis')
            plt.ylabel('Predicted Age at Diagnosis')
            plt.title('Gradient Descent Regression: Actual vs Predicted Age')

            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

            plt.tight_layout()
            plt.show()

        else:
            print("Not enough non-missing age values for regression.")
    else:
        print(f"Skipping age regression because '{age_col}' was not found.")
    # -----------------------------
    # Step 6: KMeans Clustering
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
        palette='Set2'
    )
    plt.title('KMeans Clustering (k=3)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.tight_layout()
    plt.show()

    # Save sample-level results
    output_path = script_dir / 'LUSC_PCA_results.csv'
    LUSC_merged.to_csv(output_path)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()