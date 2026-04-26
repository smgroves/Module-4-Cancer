# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Load the data
####################################################
# Load gene expression data as the main dataset, and load metadata separately
data = pd.read_csv(
    '/Users/vasishtramineni/Downloads/Computational BME/Module-4-Cancer-Horn-Ramineni/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
metadata_df = pd.read_csv(
    '/Users/vasishtramineni/Downloads/Computational BME/Module-4-Cancer-Horn-Ramineni/data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
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
cancer_type = 'LUSC'  # Lung Squamous Cell Carcinoma

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
LUSC_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################
desired_gene_list = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'MYC']
gene_list = [gene for gene in desired_gene_list if gene in LUSC_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
LUSC_gene_data = LUSC_data.loc[gene_list]
print(LUSC_gene_data.head())

# %%
# Basic statistics on the subsetted data
####################################################
print(LUSC_gene_data.describe())
print(LUSC_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(LUSC_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(LUSC_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
# Merge the subsetted expression data with metadata for LUSC samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
LUSC_metadata = metadata_df.loc[cancer_samples]
LUSC_merged = LUSC_gene_data.T.merge(
    LUSC_metadata, left_index=True, right_index=True)
print(LUSC_merged.head())

# %%
# Plotting
####################################################
# Boxplot of EGFR expression in LUSC samples using SEABORN
# Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
sns.boxplot(data=LUSC_merged, x="gender", y='EGFR')
plt.title("EGFR Expression by Gender in LUSC Samples")
plt.show()

# Boxplot of MYC and EGFR expression in LUSC samples using PANDAS directly
LUSC_merged[['MYC', 'EGFR']].plot.box()
plt.title("MYC and EGFR Expression in BRCA Samples")
plt.show()

# %%
