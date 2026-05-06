import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load expression data and clinical metadata
data = pd.read_csv('data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata = pd.read_csv('data/TRAINING_SET_GSE62944_metadata.csv', index_col=0)

# Filter for LUAD and LUSC samples and align indices
lung_metadata = metadata[metadata['cancer_type'].isin(['LUAD', 'LUSC'])].copy()
common_samples = lung_metadata.index.intersection(data.columns)
lung_metadata = lung_metadata.loc[common_samples]
lung_data = data[common_samples]

# List of genes related to immune checkpoints, antigen presentation, and signaling
immune_genes = ['CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3', 'TGFB1']

# Verify gene presence in dataset and transpose for plotting
available_genes = [g for g in immune_genes if g in lung_data.index]
lung_expression = lung_data.loc[available_genes].T 

# Standardize pathologic stage labels into four main categories
def clean_stage(stage):
    if pd.isna(stage) or 'not' in str(stage).lower(): return 'Unknown'
    if 'Stage IV' in stage: return 'Stage IV'
    if 'Stage III' in stage: return 'Stage III'
    if 'Stage II' in stage: return 'Stage II'
    if 'Stage I' in stage: return 'Stage I'
    return 'Unknown'

lung_metadata['clean_stage'] = lung_metadata['ajcc_pathologic_tumor_stage'].apply(clean_stage)

# Merge expression levels with clinical features
combined = lung_expression.merge(lung_metadata, left_index=True, right_index=True)

# Generate boxplots for each gene to compare expression across stages and subtypes
for gene in available_genes:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=combined, 
        x='clean_stage', 
        y=gene, 
        hue='cancer_type', 
        order=['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    )
    
    plt.title(f"Immune Evasion: {gene} Expression by Stage")
    plt.ylabel("Expression log2(TPM+1)")
    plt.xlabel("Pathologic Stage")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Print processing summary
print(f"Analysis complete for {len(available_genes)} genes.")