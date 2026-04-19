# Reduces Data to Target Genes

import pandas as pd

# Load expression data and clinical metadata
data = pd.read_csv('data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata = pd.read_csv('data/TRAINING_SET_GSE62944_metadata.csv', index_col=0)

'''
Immune Evasion Hallmark Genes:
MYC, EGFR, PIK3CA, BRAF, CTNNB1, STAT3, PTEN, BRCA1, BRCA2, TP53, STK11, RB1, SMAD4, APC, ATM, CD274, CTLA4, LAG3, HLA-A, B2M, TGFB1

'''
immune_genes = ['CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3', 'TGFB1']



# Filters rows where gene is in target list
data_filtered = data[data.iloc[:, 0].isin(immune_genes)]

# Filter for LUAD and LUSC samples and align indices
lung_metadata = metadata[metadata['cancer_type'].isin(['LUAD', 'LUSC'])].copy()
common_samples = lung_metadata.index.intersection(data_filtered.columns)
lung_metadata = lung_metadata.loc[common_samples]
lung_data = data_filtered[common_samples]



'''
# Filter rows where gene is in target list
data_filtered = data[data.iloc[:, 0].isin(immune_genes)]
metadata_filtered = metadata[metadata["cancer_type"].isin(["LUAD", "LUSC"])]

combined = pd.concat([data_filtered, metadata_filtered], ignore_index=True)

filter = combined[
    (combined["gene"].isin(targets["gene"])) &
    (combined["cancer_type"].isin(["BRCA", "LUAD"]))
]
'''

# Save new CSV
lung_data.to_csv("filtered_info.csv", index=False)