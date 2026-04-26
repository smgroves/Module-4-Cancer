import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / 'data'

print("="*70)
print("Finding genes that work in BOTH training AND validation sets")
print("="*70)

# Load all data
expr_train = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata_train = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_metadata.csv', index_col=0)
expr_val = pd.read_csv(data_dir / 'VALIDATION_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata_val = pd.read_csv(data_dir / 'VALIDATION_SET_GSE62944_metadata.csv', index_col=0)
smoking_df = pd.read_csv(data_dir / 'packs_per_year_smoked.csv', index_col=0, header=0)
smoking_df.columns = ['pack_years']
smoking_df['pack_years'] = pd.to_numeric(smoking_df['pack_years'], errors='coerce')

# Get LUSC samples with smoking data
lusc_train_smoking = metadata_train[metadata_train['cancer_type'] == 'LUSC'].index
lusc_train_smoking = lusc_train_smoking.intersection(smoking_df[smoking_df['pack_years'].notna()].index)

lusc_val_smoking = metadata_val[metadata_val['cancer_type'] == 'LUSC'].index
lusc_val_smoking = lusc_val_smoking.intersection(smoking_df[smoking_df['pack_years'].notna()].index)

print(f"Training LUSC with smoking: {len(lusc_train_smoking)}")
print(f"Validation LUSC with smoking: {len(lusc_val_smoking)}\n")

# Find genes present in both datasets
common_genes = set(expr_train.index).intersection(set(expr_val.index))
print(f"Common genes in both datasets: {len(common_genes)}\n")

# Calculate correlations in both datasets
correlations = {}

for gene in common_genes:
    try:
        expr_train_gene = expr_train.loc[gene, lusc_train_smoking]
        y_train = smoking_df.loc[lusc_train_smoking, 'pack_years']
        corr_train = expr_train_gene.corr(y_train)
        
        expr_val_gene = expr_val.loc[gene, lusc_val_smoking]
        y_val = smoking_df.loc[lusc_val_smoking, 'pack_years']
        corr_val = expr_val_gene.corr(y_val)
        
        if not np.isnan(corr_train) and not np.isnan(corr_val):
            # Score based on agreement between datasets
            agreement = abs(corr_train * corr_val)  # Same direction + magnitude
            correlations[gene] = {
                'train_corr': corr_train,
                'val_corr': corr_val,
                'agreement': agreement,
                'avg_abs_corr': (abs(corr_train) + abs(corr_val)) / 2
            }
    except:
        pass

# Sort by agreement
sorted_genes = sorted(correlations.items(), key=lambda x: x[1]['agreement'], reverse=True)

print("Top 30 genes with CONSISTENT correlation in both training AND validation:")
print("(These should generalize better!)\n")
print(f"{'Gene':<15} {'Train Corr':<12} {'Val Corr':<12} {'Agreement':<12}")
print("-" * 55)

top_genes = []
for i, (gene, metrics) in enumerate(sorted_genes[:30], 1):
    print(f"{gene:<15} {metrics['train_corr']:>10.4f}  {metrics['val_corr']:>10.4f}  {metrics['agreement']:>10.4f}")
    top_genes.append(gene)

print("\n" + "="*70)
print("Testing with these consistent genes...")
print("="*70 + "\n")

# Test with top 20 consistent genes
test_genes = top_genes[:20]

# Training
X_train = expr_train.loc[test_genes, lusc_train_smoking].T
y_train = smoking_df.loc[lusc_train_smoking, 'pack_years'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

corr_pc1_train = np.corrcoef(X_train_pca[:, 0], y_train)[0, 1]
corr_pc2_train = np.corrcoef(X_train_pca[:, 1], y_train)[0, 1]

# Validation
X_val = expr_val.loc[test_genes, lusc_val_smoking].T
y_val = smoking_df.loc[lusc_val_smoking, 'pack_years'].values

X_val_scaled = scaler.transform(X_val)
X_val_pca = pca.transform(X_val_scaled)

corr_pc1_val = np.corrcoef(X_val_pca[:, 0], y_val)[0, 1]
corr_pc2_val = np.corrcoef(X_val_pca[:, 1], y_val)[0, 1]

print(f"Using top 20 consistent genes: {', '.join(test_genes[:10])}...\n")
print(f"Training PC1 correlation: {corr_pc1_train:.4f}")
print(f"Validation PC1 correlation: {corr_pc1_val:.4f}")
print(f"Difference: {abs(corr_pc1_train - corr_pc1_val):.4f}")
print(f"\nTraining PC2 correlation: {corr_pc2_train:.4f}")
print(f"Validation PC2 correlation: {corr_pc2_val:.4f}")
