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
    # Simplified family history
    # -----------------------------
    fam_col = 'family_history_cancer_type'

    if fam_col in LUSC_merged.columns:
        def simplify_family_history(x):
            if pd.isna(x) or x == 'nan':
                return 'None/Unknown'
            elif x.lower() == 'no':
                return 'No'
            else:
                return 'Yes (some cancer)'

        LUSC_merged['family_history_simple'] = LUSC_merged[fam_col].astype(str).apply(simplify_family_history)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='family_history_simple',
            palette='Set1',
            s=90
        )
        plt.title('PCA Colored by Family Cancer History (Simplified)')
        plt.tight_layout()
        plt.show()

            # -----------------------------
    # Clean + plot thyroid disease
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
        plt.tight_layout()
        plt.show()

            # -----------------------------
    # Plot: Ethnicity
    # -----------------------------
    eth_col = 'ethnicity'

    if eth_col in LUSC_merged.columns:

        def clean_ethnicity(x):
            x = str(x).lower()

            if 'not hispanic' in x:
                return 'Not Hispanic'
            elif 'hispanic' in x:
                return 'Hispanic'
            else:
                return 'Unknown'

        LUSC_merged['ethnicity_clean'] = LUSC_merged[eth_col].apply(clean_ethnicity)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=LUSC_merged,
            x='PC1',
            y='PC2',
            hue='ethnicity_clean',
            palette='Set1',
            s=90
        )
        plt.title('PCA of LUSC Samples Colored by Ethnicity')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()

    else:
        print(f"Skipping ethnicity plot because '{eth_col}' not found.")
    
        # -----------------------------
       # -----------------------------
    # Clean + Plot Tumor Stage
    # -----------------------------
    stage_col = 'ajcc_pathologic_tumor_stage'

    if stage_col in LUSC_merged.columns:

        def simplify_stage(x):
            x = str(x).upper()

            if 'I' in x and 'II' not in x:
                return 'Stage I'
            elif 'II' in x and 'III' not in x:
                return 'Stage II'
            elif 'III' in x:
                return 'Stage III'
            elif 'IV' in x:
                return 'Stage IV'
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
        plt.tight_layout()
        plt.show()
    
        # -----------------------------
    # Plot: Race
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
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()

    else:
        print(f"Skipping race plot because '{race_col}' not found.")
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
    
    # Add cluster centroid labels
    for cluster in range(kmeans.n_clusters):
        cluster_points = LUSC_merged[LUSC_merged['cluster'] == cluster]
        centroid_pc1 = cluster_points['PC1'].mean()
        centroid_pc2 = cluster_points['PC2'].mean()
        plt.text(centroid_pc1, centroid_pc2, f'C{cluster}', 
                fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.title('KMeans Clustering (k=3)')
    plt.show()

    # Save
    output_path = script_dir / 'LUSC_PCA_results.csv'
    LUSC_merged.to_csv(output_path)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()