import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# =====================================================
# Setup paths
# =====================================================
script_dir = Path().resolve()
possible_dirs = [
    script_dir / "data",
    script_dir.parent / "data",
    script_dir.parent.parent / "data"
]

for d in possible_dirs:
    if (d / "TRAINING_SET_GSE62944_subsample_log2TPM.csv").exists():
        data_dir = d
        break
else:
    raise FileNotFoundError("Could not find data folder.")

print("Using data directory:", data_dir)

# =====================================================
# Load training set for model training
# =====================================================
print("\n" + "="*60)
print("LOADING TRAINING SET")
print("="*60)

expr_path_train = data_dir / "TRAINING_SET_GSE62944_subsample_log2TPM.csv"
metadata_path_train = data_dir / "TRAINING_SET_GSE62944_metadata.csv"

data_train = pd.read_csv(expr_path_train, index_col=0, header=0)
metadata_train = pd.read_csv(metadata_path_train, index_col=0, header=0)

cancer_type = "LUSC"
cancer_samples_train = metadata_train[metadata_train['cancer_type'] == cancer_type].index
shared_samples_train = cancer_samples_train.intersection(data_train.columns)
LUSC_data_train = data_train.loc[:, shared_samples_train]
LUSC_metadata_train = metadata_train.loc[shared_samples_train].copy()

print(f"Training LUSC samples: {len(shared_samples_train)}")

# =====================================================
# Load test set
# =====================================================
print("\n" + "="*60)
print("LOADING TEST SET")
print("="*60)

expr_path_test = data_dir / "TEST_SET_GSE62944_subsample_log2TPM.csv"
metadata_path_test = data_dir / "TEST_SET_GSE62944_metadata.csv"

data_test = pd.read_csv(expr_path_test, index_col=0, header=0)
metadata_test = pd.read_csv(metadata_path_test, index_col=0, header=0)

cancer_samples_test = metadata_test[metadata_test['cancer_type'] == cancer_type].index
shared_samples_test = cancer_samples_test.intersection(data_test.columns)
LUSC_data_test = data_test.loc[:, shared_samples_test]
LUSC_metadata_test = metadata_test.loc[shared_samples_test].copy()

print(f"Test LUSC samples: {len(shared_samples_test)}")

# =====================================================
# Gene list
# =====================================================
desired_gene_list = [
    'ATP2B2', 'FAM155A', 'KCNIP3', 'TAC1', 'ALX4', 'TTLL6', 'MUC12', 
    'IFI6', 'HRK', 'C15orf41', 'PRSS53', 'CPE', 'GRK7', 'KIAA1671', 
    'NEUROG3', 'OR4F17', 'SLC7A3', 'UBXN10', 'UNC93B1', 'USP2'
]

gene_list = []
for gene in desired_gene_list:
    if gene in LUSC_data_train.index and gene in LUSC_data_test.index:
        gene_list.append(gene)
    else:
        print(f"Warning: {gene} not found in both training and test data")

print(f"\nGenes used: {gene_list}")

# =====================================================
# Prepare training data
# =====================================================
X_train = LUSC_data_train.loc[gene_list].T
LUSC_merged_train = X_train.merge(LUSC_metadata_train, left_index=True, right_index=True)
X_train = LUSC_merged_train[gene_list].copy()
X_train = X_train.dropna(axis=0)
LUSC_merged_train = LUSC_merged_train.loc[X_train.index].copy()

# =====================================================
# Train scaler and PCA on training data
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

LUSC_merged_train["PC1"] = X_train_pca[:, 0]
LUSC_merged_train["PC2"] = X_train_pca[:, 1]

print(f"\nTraining set size: {len(LUSC_merged_train)}")
print(f"Explained variance: {pca.explained_variance_ratio_}")

# =====================================================
# Prepare test data using training scaler/PCA
# =====================================================
X_test = LUSC_data_test.loc[gene_list].T
LUSC_merged_test = X_test.merge(LUSC_metadata_test, left_index=True, right_index=True)
X_test = LUSC_merged_test[gene_list].copy()
X_test = X_test.dropna(axis=0)
LUSC_merged_test = LUSC_merged_test.loc[X_test.index].copy()

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

LUSC_merged_test["PC1"] = X_test_pca[:, 0]
LUSC_merged_test["PC2"] = X_test_pca[:, 1]

print(f"Test set size: {len(LUSC_merged_test)}")

# =====================================================
# Load smoking data
# =====================================================
smoking_path = data_dir / "packs_per_year_smoked.csv"
smoking_df = pd.read_csv(smoking_path, index_col=0, header=0)
smoking_col = "tobacco_smoking_pack_years_smoked"

# Merge smoking data with training set
LUSC_train_smoking = LUSC_merged_train.copy()
LUSC_train_smoking = LUSC_train_smoking.merge(
    smoking_df, left_index=True, right_index=True, how="left"
)
LUSC_train_smoking[smoking_col] = pd.to_numeric(
    LUSC_train_smoking[smoking_col], errors="coerce"
)
train_smoking_data = LUSC_train_smoking.dropna(subset=[smoking_col]).copy()

print(f"\nTraining samples with smoking data: {len(train_smoking_data)}")

# Merge smoking data with test set
LUSC_test_smoking = LUSC_merged_test.copy()
LUSC_test_smoking = LUSC_test_smoking.merge(
    smoking_df, left_index=True, right_index=True, how="left"
)
LUSC_test_smoking[smoking_col] = pd.to_numeric(
    LUSC_test_smoking[smoking_col], errors="coerce"
)
test_smoking_data = LUSC_test_smoking.dropna(subset=[smoking_col]).copy()

print(f"Test samples with smoking data: {len(test_smoking_data)}")

# =====================================================
# Train Elastic Net on training set
# =====================================================
print("\n" + "="*60)
print("TRAINING ELASTIC NET ON TRAINING SET")
print("="*60)

X_train_smoking = train_smoking_data[["PC1", "PC2"]].values
y_train_smoking = train_smoking_data[[smoking_col]].values

# Standardize features
X_train_mean = X_train_smoking.mean(axis=0)
X_train_std = X_train_smoking.std(axis=0)
X_train_std[X_train_std == 0] = 1

X_train_scaled_smk = (X_train_smoking - X_train_mean) / X_train_std

# Train Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0, max_iter=10000)
elastic_net.fit(X_train_scaled_smk, y_train_smoking.ravel())

print(f"Elastic Net Coefficients")
print(f"Intercept: {elastic_net.intercept_:.4f}")
print(f"PC1 weight: {elastic_net.coef_[0]:.4f}")
print(f"PC2 weight: {elastic_net.coef_[1]:.4f}")

# Training predictions
y_train_pred = elastic_net.predict(X_train_scaled_smk)
train_mse = mean_squared_error(y_train_smoking, y_train_pred)
train_r2 = r2_score(y_train_smoking, y_train_pred)

print(f"\nTraining Results")
print(f"MSE: {train_mse:.4f}, R²: {train_r2:.4f}")

# =====================================================
# Apply model to test set
# =====================================================
print("\n" + "="*60)
print("APPLYING MODEL TO TEST SET")
print("="*60)

X_test_smoking = test_smoking_data[["PC1", "PC2"]].values
y_test_smoking = test_smoking_data[[smoking_col]].values

# Apply training standardization
X_test_scaled_smk = (X_test_smoking - X_train_mean) / X_train_std

# Get predictions
y_test_pred = elastic_net.predict(X_test_scaled_smk)

# Metrics
test_mse = mean_squared_error(y_test_smoking, y_test_pred)
test_r2 = r2_score(y_test_smoking, y_test_pred)

print(f"Test Results")
print(f"MSE: {test_mse:.4f}")
print(f"R²: {test_r2:.4f}")

# Residual statistics
residuals = y_test_smoking.flatten() - y_test_pred
print(f"\nResidual Statistics (Test Set)")
print(f"Mean residual: {np.mean(residuals):.4f}")
print(f"Std residual: {np.std(residuals):.4f}")
print(f"Max abs residual: {np.max(np.abs(residuals)):.4f}")

# =====================================================
# Plots
# =====================================================
sns.set(style="whitegrid")

# Test set PCA
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    test_smoking_data["PC1"],
    test_smoking_data["PC2"],
    c=test_smoking_data[smoking_col],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="black"
)
plt.colorbar(scatter, label="Pack Years Smoked")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Test Set PCA Colored by Pack Years Smoked")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_smoking, y_test_pred, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)

min_val = min(y_test_smoking.min(), y_test_pred.min())
max_val = max(y_test_smoking.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 
         linestyle="--", color="red", linewidth=2, label="Perfect prediction")
plt.axhline(y_train_smoking.mean(), linestyle=":", color="gray", 
            linewidth=1.5, label="Mean (training set)")

plt.xlabel("Actual Pack Years Smoked")
plt.ylabel("Predicted Pack Years Smoked")
plt.title(f"Test Set: Actual vs Predicted Smoking\n(R² = {test_r2:.4f}, MSE = {test_mse:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
plt.axhline(0, linestyle="--", color="red", linewidth=2)
plt.xlabel("Predicted Pack Years Smoked")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Test Set: Residual Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# Comparison: Training vs Validation vs Test
# =====================================================
print("\n" + "="*60)
print("SUMMARY: TRAIN vs TEST")
print("="*60)

comparison_df = pd.DataFrame({
    "Dataset": ["Training", "Test"],
    "N_Samples": [len(train_smoking_data), len(test_smoking_data)],
    "MSE": [train_mse, test_mse],
    "R2": [train_r2, test_r2],
    "Smoking_Range": [
        f"{train_smoking_data[smoking_col].min():.1f}-{train_smoking_data[smoking_col].max():.1f}",
        f"{test_smoking_data[smoking_col].min():.1f}-{test_smoking_data[smoking_col].max():.1f}"
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Save results
results_path = script_dir / 'test_set_predictions_results.csv'
test_results_df = pd.DataFrame({
    'Sample': test_smoking_data.index,
    'Actual_Smoking': y_test_smoking.flatten(),
    'Predicted_Smoking': y_test_pred,
    'Residual': residuals
})
test_results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved test predictions to: {results_path}")

print("\nAnalysis complete!")
