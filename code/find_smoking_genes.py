import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / 'data'

# Load data
print("Loading data...")
expr_data = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)
metadata = pd.read_csv(data_dir / 'TRAINING_SET_GSE62944_metadata.csv', index_col=0)
smoking_df = pd.read_csv(data_dir / 'packs_per_year_smoked.csv', index_col=0, header=0)
smoking_df.columns = ['pack_years']
smoking_df['pack_years'] = pd.to_numeric(smoking_df['pack_years'], errors='coerce')

# Filter to LUSC
lusc_samples = metadata[metadata['cancer_type'] == 'LUSC'].index
lusc_expr = expr_data.loc[:, lusc_samples]
lusc_smoking = smoking_df.reindex(lusc_samples)

# Get samples with smoking data
lusc_with_smoking = lusc_smoking[lusc_smoking['pack_years'].notna()].index
expr_subset = lusc_expr.loc[:, lusc_with_smoking]
smoking_subset = lusc_smoking.loc[lusc_with_smoking]

print(f"LUSC samples with smoking data: {len(lusc_with_smoking)}\n")

# ====================================================================
# Strategy 1: Find genes that directly correlate with smoking
# ====================================================================
print("="*70)
print("STRATEGY 1: Finding genes that directly correlate with smoking")
print("="*70)

correlations = {}
for gene in expr_subset.index:
    expr_values = expr_subset.loc[gene]
    if expr_values.notna().sum() > 5:  # At least 5 non-null values
        corr = expr_values.corr(smoking_subset['pack_years'])
        if not np.isnan(corr):
            correlations[gene] = corr

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 20 genes positively correlated with smoking:")
top_positive = [g for g, c in sorted_corr if c > 0][:20]
for i, (gene, corr) in enumerate(sorted_corr[:10], 1):
    if corr > 0:
        print(f"{i}. {gene}: {corr:.4f}")

print("\nTop 10 genes negatively correlated with smoking:")
top_negative = [g for g, c in sorted_corr if c < 0][:10]
for i, (gene, corr) in enumerate(sorted_corr[:10]):
    if corr < 0:
        print(f"{i}. {gene}: {corr:.4f}")

# ====================================================================
# Strategy 2: Use genes with highest variance (may relate to smoking exposure)
# ====================================================================
print("\n" + "="*70)
print("STRATEGY 2: Top 50 most variable genes (might mark smoking-exposed tumors)")
print("="*70)

top_variable_genes = expr_subset.var(axis=1).nlargest(50).index.tolist()
print(f"Selected {len(top_variable_genes)} top variable genes")

# ====================================================================
# Strategy 3: Use inflammation and stress response genes (known smoking markers)
# ====================================================================
print("\n" + "="*70)
print("STRATEGY 3: Known smoking/inflammation marker genes")
print("="*70)

# These are genes known to be affected by smoking exposure
smoking_markers = [
    'IL6', 'TNF', 'IL1B', 'IL8', 'CRP', 'CD14', 'TLR4',  # Inflammation
    'TP53', 'KRAS', 'CDKN2A', 'PTEN',  # Tumor suppressors
    'CYP1A1', 'CYP1B1', 'GSTM1', 'GSTP1', 'NAT2',  # Carcinogen metabolism
    'SOD1', 'CAT', 'GPX1', 'NFKB1',  # Oxidative stress
    'NOTCH1', 'FAT1', 'CDKN2A', 'KEAP1',  # LUSC-specific
    'NRF2', 'HMOX1', 'SOD2', 'PARK7'  # Stress response
]

# Filter to genes that exist
smoking_markers_found = [g for g in smoking_markers if g in expr_subset.index]
print(f"Found {len(smoking_markers_found)}/{len(smoking_markers)} smoking marker genes")
if smoking_markers_found:
    print(f"Genes: {', '.join(smoking_markers_found)}")

# ====================================================================
# Test different gene sets with PCA + Linear Regression
# ====================================================================
print("\n" + "="*70)
print("TESTING DIFFERENT GENE SETS")
print("="*70)

test_sets = {
    'Original (12 oncogenes)': ['ALK', 'BRAF', 'CDK4', 'BCL2', 'EGFR', 'FLT3', 'KIT', 'MTOR', 'ROS1', 'SMO', 'XPO1', 'BTK'],
    'Top 20 correlated genes': top_positive[:20],
    'Top 50 variable genes': top_variable_genes,
    'Smoking markers (found)': smoking_markers_found,
}

results = {}

for set_name, gene_list in test_sets.items():
    # Filter to available genes
    available_genes = [g for g in gene_list if g in expr_subset.index]
    
    if len(available_genes) < 2:
        print(f"\n{set_name}: SKIP (only {len(available_genes)} genes found)")
        continue
    
    try:
        # Prepare data
        X = expr_subset.loc[available_genes].T.dropna()
        
        if len(X) < 10:
            print(f"\n{set_name}: SKIP (only {len(X)} samples after removing NaN)")
            continue
        
        y = smoking_subset.loc[X.index, 'pack_years'].values
        
        # Standardize and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=min(2, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Train/test split - keep as 1D arrays before splitting
        indices = np.arange(len(X_pca))
        indices_train, indices_test = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        
        X_train = X_pca[indices_train]
        X_test = X_pca[indices_test]
        y_train = y[indices_train].reshape(-1, 1)
        y_test = y[indices_test].reshape(-1, 1)
        
        # Gradient descent
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        theta = np.zeros((X_train_b.shape[1], 1))
        learning_rate = 0.01
        n_iterations = 2000
        m = len(X_train_b)
        
        for i in range(n_iterations):
            gradients = (2 / m) * X_train_b.T @ (X_train_b @ theta - y_train)
            theta = theta - learning_rate * gradients
        
        # Predictions and metrics
        y_pred = X_test_b @ theta
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Direct correlations with PCs
        corr_pc1_smoke = np.corrcoef(X_pca[:, 0], y)[0, 1]
        if X_pca.shape[1] > 1:
            corr_pc2_smoke = np.corrcoef(X_pca[:, 1], y)[0, 1]
        else:
            corr_pc2_smoke = np.nan
        
        results[set_name] = {
            'n_genes': len(available_genes),
            'n_samples': len(X),
            'pc1_corr': corr_pc1_smoke,
            'pc2_corr': corr_pc2_smoke,
            'mse': mse,
            'r2': r2
        }
        
        print(f"\n{set_name}:")
        print(f"  Genes used: {len(available_genes)}, Samples: {len(X)}")
        print(f"  PC1 correlation: {corr_pc1_smoke:.4f}, PC2 correlation: {corr_pc2_smoke:.4f}")
        print(f"  Gradient Descent R²: {r2:.4f}, MSE: {mse:.2f}")
        
    except Exception as e:
        import traceback
        print(f"\n{set_name}: ERROR - {str(e)}")
        traceback.print_exc()

# ====================================================================
# Summary
# ====================================================================
print("\n" + "="*70)
print("SUMMARY - Best performing gene sets:")
print("="*70)

if results:
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (set_name, metrics) in enumerate(sorted_results[:5], 1):
        print(f"{i}. {set_name}")
        print(f"   R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.2f}")
        print(f"   PC correlations: PC1={metrics['pc1_corr']:.4f}, PC2={metrics['pc2_corr']:.4f}")
