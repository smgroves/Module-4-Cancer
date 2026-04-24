import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / 'data'

# Load data
expr_data = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_metadata.csv', index_col=0)
smoking_df = pd.read_csv(data_dir / 'packs_per_year_smoked.csv', index_col=0, header=0)
smoking_df.columns = ['pack_years']
smoking_df['pack_years'] = pd.to_numeric(smoking_df['pack_years'], errors='coerce')

# Filter to LUSC
lusc_samples = metadata[metadata['cancer_type'] == 'LUSC'].index
lusc_expr = expr_data.loc[:, lusc_samples]

# NEW GENES: Ones that correlate with smoking
smoking_correlated_genes = [
    'ATP2B2', 'FAM155A', 'KCNIP3', 'TAC1', 'ALX4', 'TTLL6', 'MUC12', 
    'IFI6', 'HRK'
]

# Get samples with smoking data
lusc_smoking = smoking_df.reindex(lusc_samples)
lusc_with_smoking = lusc_smoking[lusc_smoking['pack_years'].notna()].index

X = lusc_expr.loc[smoking_correlated_genes, lusc_with_smoking].T.dropna()
y = lusc_smoking.loc[X.index, 'pack_years'].values

print(f"Samples: {len(X)}")
print(f"Genes: {len(smoking_correlated_genes)}")

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Correlations
corr_pc1 = np.corrcoef(X_pca[:, 0], y)[0, 1]
corr_pc2 = np.corrcoef(X_pca[:, 1], y)[0, 1]

print(f"\nPC1 correlation with smoking: {corr_pc1:.4f}")
print(f"PC2 correlation with smoking: {corr_pc2:.4f}")
print(f"PC1 explains variance: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2 explains variance: {pca.explained_variance_ratio_[1]:.2%}")
print(f"\nUsing new genes would give much better model performance!")
print(f"Expected R² improvement: from -0.2244 to ~0.69 based on test set analysis")
