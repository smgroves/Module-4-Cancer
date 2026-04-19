import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'

    expr_path = data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv'
    metadata_path = data_dir / 'TRAINING_SET_GSE62944_metadata.csv'

    print(f'Loading expression data from: {expr_path}')
    print(f'Loading metadata from: {metadata_path}')

    data = pd.read_csv(expr_path, index_col=0, header=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0, header=0)

    print('\nExpression data:')
    print(data.shape)
    print(data.columns[:5].tolist())
    print(data.head())

    print('\nMetadata:')
    print(metadata_df.shape)
    print(metadata_df.columns.tolist())
    print(metadata_df.head())

    cancer_type = 'LUSC'
    cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
    print(f'\nNumber of {cancer_type} samples in metadata: {len(cancer_samples)}')

    shared_samples = cancer_samples.intersection(data.columns)
    missing_samples = cancer_samples.difference(data.columns)

    if len(missing_samples) > 0:
        print(f'Warning: {len(missing_samples)} samples from metadata are missing in expression data.')
        print('Missing sample IDs:')
        print(list(missing_samples)[:10])

    print(f'Number of shared samples used for expression subset: {len(shared_samples)}')
    if len(shared_samples) == 0:
        raise ValueError('No matching sample IDs found between metadata and expression data.')

    LUSC_data = data.loc[:, shared_samples]

    desired_gene_list = ['ALK', 'BRAF', 'CDK-4', 'BCL-2']
    # Alias map to handle gene name variations (e.g., CDK-4 -> CDK4)
    alias_map = {
        'ALK': 'ALK',
        'BRAF': 'BRAF',
        'CDK-4': 'CDK4',
        'BCL-2': 'BCL2'
    }
    gene_list = []
    for gene in desired_gene_list:
        actual_gene = alias_map.get(gene, gene)
        if actual_gene in LUSC_data.index:
            gene_list.append(actual_gene)
        else:
            print(f"Warning: {gene} (mapped to {actual_gene}) not found in expression data.")

    if len(gene_list) == 0:
        raise ValueError('None of the desired genes were found in the expression dataset.')

    LUSC_gene_data = LUSC_data.loc[gene_list]
    print('\nSubset gene expression data:')
    print(LUSC_gene_data.shape)
    print(LUSC_gene_data.head())

    print('\nGene-level statistics:')
    print(LUSC_gene_data.describe())
    print('\nVariance by gene:')
    print(LUSC_gene_data.var(axis=1))
    print('\nMean by gene:')
    print(LUSC_gene_data.mean(axis=1))
    print('\nMedian by gene:')
    print(LUSC_gene_data.median(axis=1))

    print('\nGender counts by cancer type:')
    print(metadata_df.groupby('cancer_type')['gender'].value_counts())

    metadata_df['age_at_diagnosis'] = pd.to_numeric(
        metadata_df['age_at_diagnosis'], errors='coerce')
    print('\nAverage age at diagnosis by cancer type:')
    print(metadata_df.groupby('cancer_type')['age_at_diagnosis'].mean())

    LUSC_metadata = metadata_df.loc[shared_samples]
    LUSC_merged = LUSC_gene_data.T.merge(
        LUSC_metadata, left_index=True, right_index=True)
    print('\nMerged expression + metadata:')
    print(LUSC_merged.head())

    plot_genes = gene_list
    print(f'\nPlotting genes: {plot_genes}')

    sns.set(style='whitegrid')
    if len(plot_genes) == 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=LUSC_merged, x='gender', y=plot_genes[0])
        plt.title(f'{plot_genes[0]} Expression by Gender in LUSC Samples')
        plt.tight_layout()
        plt.show()
    else:
        melted = LUSC_merged.melt(
            id_vars='gender', value_vars=plot_genes,
            var_name='gene', value_name='expression')
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=melted, x='gene', y='expression', hue='gender')
        plt.title('Gene Expression by Gender in LUSC Samples')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        LUSC_merged[plot_genes].plot.box()
        plt.title(f"Expression of {', '.join(plot_genes)} in LUSC Samples")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
