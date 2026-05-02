import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# =====================================================
# Helper function: regression metrics
# =====================================================
def compute_regression_metrics(y_train_true, y_train_pred, y_val_true, y_val_pred):
    y_train_true = y_train_true.flatten()
    y_train_pred = y_train_pred.flatten()
    y_val_true = y_val_true.flatten()
    y_val_pred = y_val_pred.flatten()

    train_residuals = y_train_true - y_train_pred
    val_residuals = y_val_true - y_val_pred

    train_mae = np.mean(np.abs(train_residuals))
    val_mae = np.mean(np.abs(val_residuals))

    train_mse = np.mean(train_residuals ** 2)
    val_mse = np.mean(val_residuals ** 2)

    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)

    train_r2 = r2_score(y_train_true, y_train_pred)
    val_r2 = r2_score(y_val_true, y_val_pred)

    baseline_train_pred = np.full_like(y_train_true, y_train_true.mean())
    baseline_val_pred = np.full_like(y_val_true, y_train_true.mean())

    baseline_train_mae = np.mean(np.abs(y_train_true - baseline_train_pred))
    baseline_val_mae = np.mean(np.abs(y_val_true - baseline_val_pred))

    baseline_train_rmse = np.sqrt(np.mean((y_train_true - baseline_train_pred) ** 2))
    baseline_val_rmse = np.sqrt(np.mean((y_val_true - baseline_val_pred) ** 2))

    train_mae_improvement = 100 * (baseline_train_mae - train_mae) / baseline_train_mae
    val_mae_improvement = 100 * (baseline_val_mae - val_mae) / baseline_val_mae

    train_rmse_improvement = 100 * (baseline_train_rmse - train_rmse) / baseline_train_rmse
    val_rmse_improvement = 100 * (baseline_val_rmse - val_rmse) / baseline_val_rmse

    train_nrmse = train_rmse / np.mean(y_train_true)
    val_nrmse = val_rmse / np.mean(y_val_true)

    train_mae_percent = 100 * train_mae / np.mean(y_train_true)
    val_mae_percent = 100 * val_mae / np.mean(y_val_true)

    metrics_df = pd.DataFrame({
        "Dataset": ["Training", "Validation"],
        "MAE": [train_mae, val_mae],
        "MSE": [train_mse, val_mse],
        "RMSE": [train_rmse, val_rmse],
        "R2": [train_r2, val_r2],
        "Normalized_RMSE": [train_nrmse, val_nrmse],
        "MAE_percent_of_mean": [train_mae_percent, val_mae_percent],
        "MAE_improvement_vs_baseline_percent": [train_mae_improvement, val_mae_improvement],
        "RMSE_improvement_vs_baseline_percent": [train_rmse_improvement, val_rmse_improvement]
    })

    print("\n" + "="*60)
    print("ABSOLUTE AND RELATIVE REGRESSION METRICS")
    print("="*60)
    print(metrics_df)

    # Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_pred, train_residuals, alpha=0.7, label="Training")
    plt.scatter(y_val_pred, val_residuals, alpha=0.7, label="Validation")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Pack Years Smoked")
    plt.ylabel("Residual: Actual - Predicted")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Absolute metric comparison
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"],
        [train_mae, val_mae, train_rmse, val_rmse]
    )
    plt.ylabel("Error in Pack Years")
    plt.title("Absolute Error Metrics")
    plt.tight_layout()
    plt.show()

    # Actual vs predicted validation
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val_true, y_val_pred, s=80, alpha=0.7, label="Model predictions")

    min_val = min(y_val_true.min(), y_val_pred.min())
    max_val = max(y_val_true.max(), y_val_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle="--", label="Perfect prediction")

    plt.axhline(y_train_true.mean(),
                linestyle=":",
                label="Mean baseline prediction")

    plt.xlabel("Actual Pack Years Smoked")
    plt.ylabel("Predicted Pack Years Smoked")
    plt.title("Validation Set: Actual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return metrics_df


# =====================================================
# Jupyter-safe path setup
# =====================================================
current_dir = Path.cwd()

possible_dirs = [
    current_dir / "data",
    current_dir.parent / "data",
    current_dir.parent.parent / "data"
]

for d in possible_dirs:
    if (d / "TRAINING_SET_GSE62944_subsample_log2TPM.csv").exists():
        data_dir = d
        break
else:
    raise FileNotFoundError("Could not find data folder. Check where your notebook is running.")

print("Using data directory:", data_dir)


# =====================================================
# Load training set
# =====================================================
expr_path_train = data_dir / "TRAINING_SET_GSE62944_subsample_log2TPM.csv"
metadata_path_train = data_dir / "TRAINING_SET_GSE62944_metadata.csv"

data_train = pd.read_csv(expr_path_train, index_col=0, header=0)
metadata_train = pd.read_csv(metadata_path_train, index_col=0, header=0)

print("Training expression data shape:", data_train.shape)
print("Training metadata shape:", metadata_train.shape)


# =====================================================
# Load validation set
# =====================================================
expr_path_val = data_dir / "VALIDATION_SET_GSE62944_subsample_log2TPM.csv"
metadata_path_val = data_dir / "VALIDATION_SET_GSE62944_metadata.csv"

data_val = pd.read_csv(expr_path_val, index_col=0, header=0)
metadata_val = pd.read_csv(metadata_path_val, index_col=0, header=0)

print("Validation expression data shape:", data_val.shape)
print("Validation metadata shape:", metadata_val.shape)


# =====================================================
# Filter to LUSC
# =====================================================
cancer_type = "LUSC"

cancer_samples_train = metadata_train[metadata_train["cancer_type"] == cancer_type].index
shared_samples_train = cancer_samples_train.intersection(data_train.columns)

LUSC_data_train = data_train.loc[:, shared_samples_train]
LUSC_metadata_train = metadata_train.loc[shared_samples_train].copy()

cancer_samples_val = metadata_val[metadata_val["cancer_type"] == cancer_type].index
shared_samples_val = cancer_samples_val.intersection(data_val.columns)

LUSC_data_val = data_val.loc[:, shared_samples_val]
LUSC_metadata_val = metadata_val.loc[shared_samples_val].copy()


# =====================================================
# Gene set
# =====================================================
desired_gene_list = [
    "ATP2B2", "FAM155A", "KCNIP3", "TAC1", "ALX4", "TTLL6", "MUC12",
    "IFI6", "HRK", "C15orf41", "PRSS53", "CPE", "GRK7", "KIAA1671",
    "NEUROG3", "OR4F17", "SLC7A3", "UBXN10", "UNC93B1", "USP2"
]

gene_list = []

for gene in desired_gene_list:
    if gene in LUSC_data_train.index and gene in LUSC_data_val.index:
        gene_list.append(gene)
    else:
        print(f"Warning: {gene} not found in both training and validation data")

if len(gene_list) < 2:
    raise ValueError("Need at least 2 genes.")

print("\nGenes used:", gene_list)


# =====================================================
# Prepare training data
# =====================================================
X_train = LUSC_data_train.loc[gene_list].T
LUSC_merged_train = X_train.merge(
    LUSC_metadata_train,
    left_index=True,
    right_index=True
)

X_train = LUSC_merged_train[gene_list].copy()
X_train = X_train.dropna(axis=0)
LUSC_merged_train = LUSC_merged_train.loc[X_train.index].copy()


# =====================================================
# Scale and PCA on training data
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

LUSC_merged_train["PC1"] = X_train_pca[:, 0]
LUSC_merged_train["PC2"] = X_train_pca[:, 1]

print("\nTraining set size:", len(LUSC_merged_train))
print("Explained variance:", pca.explained_variance_ratio_)


# =====================================================
# Prepare validation data using training scaler/PCA
# =====================================================
X_val = LUSC_data_val.loc[gene_list].T
LUSC_merged_val = X_val.merge(
    LUSC_metadata_val,
    left_index=True,
    right_index=True
)

X_val = LUSC_merged_val[gene_list].copy()
X_val = X_val.dropna(axis=0)
LUSC_merged_val = LUSC_merged_val.loc[X_val.index].copy()

X_val_scaled = scaler.transform(X_val)
X_val_pca = pca.transform(X_val_scaled)

LUSC_merged_val["PC1"] = X_val_pca[:, 0]
LUSC_merged_val["PC2"] = X_val_pca[:, 1]

print("Validation set size:", len(LUSC_merged_val))


# =====================================================
# PCA plots
# =====================================================
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.scatter(LUSC_merged_train["PC1"], LUSC_merged_train["PC2"], alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Training Set PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(LUSC_merged_val["PC1"], LUSC_merged_val["PC2"], alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Validation Set PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# Load smoking data
# =====================================================
smoking_path = data_dir / "packs_per_year_smoked.csv"

if not smoking_path.exists():
    raise FileNotFoundError(f"Smoking data file not found at: {smoking_path}")

smoking_df = pd.read_csv(smoking_path, index_col=0, header=0)
smoking_col = "tobacco_smoking_pack_years_smoked"


# =====================================================
# Merge smoking data with training set
# =====================================================
LUSC_train_smoking = LUSC_merged_train.copy()
LUSC_train_smoking = LUSC_train_smoking.merge(
    smoking_df,
    left_index=True,
    right_index=True,
    how="left"
)

LUSC_train_smoking[smoking_col] = pd.to_numeric(
    LUSC_train_smoking[smoking_col],
    errors="coerce"
)

train_smoking_data = LUSC_train_smoking.dropna(subset=[smoking_col]).copy()

print("\nTraining samples with smoking data:", len(train_smoking_data))


# =====================================================
# Merge smoking data with validation set
# =====================================================
LUSC_val_smoking = LUSC_merged_val.copy()
LUSC_val_smoking = LUSC_val_smoking.merge(
    smoking_df,
    left_index=True,
    right_index=True,
    how="left"
)

LUSC_val_smoking[smoking_col] = pd.to_numeric(
    LUSC_val_smoking[smoking_col],
    errors="coerce"
)

val_smoking_data = LUSC_val_smoking.dropna(subset=[smoking_col]).copy()

print("Validation samples with smoking data:", len(val_smoking_data))


# =====================================================
# PCA colored by smoking
# =====================================================
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    train_smoking_data["PC1"],
    train_smoking_data["PC2"],
    c=train_smoking_data[smoking_col],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="black"
)
plt.colorbar(scatter, label="Pack Years Smoked")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Training Set PCA Colored by Pack Years Smoked")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    val_smoking_data["PC1"],
    val_smoking_data["PC2"],
    c=val_smoking_data[smoking_col],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="black"
)
plt.colorbar(scatter, label="Pack Years Smoked")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Validation Set PCA Colored by Pack Years Smoked")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# Gradient descent regression
# Predict pack years smoked from PC1 and PC2
# =====================================================
X_train_smoking = train_smoking_data[["PC1", "PC2"]].values
y_train_smoking = train_smoking_data[[smoking_col]].values

X_val_smoking = val_smoking_data[["PC1", "PC2"]].values
y_val_smoking = val_smoking_data[[smoking_col]].values

# Standardize PCA features using training data only
X_train_mean = X_train_smoking.mean(axis=0)
X_train_std = X_train_smoking.std(axis=0)
X_train_std[X_train_std == 0] = 1

X_train_scaled_smk = (X_train_smoking - X_train_mean) / X_train_std
X_val_scaled_smk = (X_val_smoking - X_train_mean) / X_train_std

# Add intercept
X_train_b = np.c_[np.ones((X_train_scaled_smk.shape[0], 1)), X_train_scaled_smk]
X_val_b = np.c_[np.ones((X_val_scaled_smk.shape[0], 1)), X_val_scaled_smk]

# Initialize theta
theta = np.zeros((X_train_b.shape[1], 1))

learning_rate = 0.01
n_iterations = 2000
m = len(X_train_b)

for i in range(n_iterations):
    gradients = (2 / m) * X_train_b.T @ (X_train_b @ theta - y_train_smoking)
    theta = theta - learning_rate * gradients

print("\nGradient Descent Weights")
print("Intercept, PC1, PC2:", theta.ravel())

# Predictions
y_train_pred = X_train_b @ theta
y_val_pred = X_val_b @ theta

# Basic validation metrics
mse_val = mean_squared_error(y_val_smoking, y_val_pred)
r2_val = r2_score(y_val_smoking, y_val_pred)

print("\nValidation Results (Gradient Descent)")
print(f"Validation MSE: {mse_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")


# =====================================================
# Elastic Net Regression
# =====================================================
from sklearn.linear_model import ElasticNet

print("\n" + "="*60)
print("ELASTIC NET REGRESSION")
print("="*60)

# ElasticNet with alpha=0.1 and l1_ratio=0.5 (balanced L1/L2)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0, max_iter=10000)

# Fit on training data
elastic_net.fit(X_train_scaled_smk, y_train_smoking.ravel())

print(f"\nElastic Net Coefficients")
print(f"Intercept: {elastic_net.intercept_:.4f}")
print(f"PC1 weight: {elastic_net.coef_[0]:.4f}")
print(f"PC2 weight: {elastic_net.coef_[1]:.4f}")

# Predictions
y_train_pred_en = elastic_net.predict(X_train_scaled_smk)
y_val_pred_en = elastic_net.predict(X_val_scaled_smk)

# Metrics
mse_train_en = mean_squared_error(y_train_smoking, y_train_pred_en)
r2_train_en = r2_score(y_train_smoking, y_train_pred_en)

mse_val_en = mean_squared_error(y_val_smoking, y_val_pred_en)
r2_val_en = r2_score(y_val_smoking, y_val_pred_en)

print(f"\nElastic Net Results")
print(f"Training MSE: {mse_train_en:.4f}, R²: {r2_train_en:.4f}")
print(f"Validation MSE: {mse_val_en:.4f}, R²: {r2_val_en:.4f}")

# Actual vs predicted plot for Elastic Net
plt.figure(figsize=(8, 6))
plt.scatter(y_val_smoking, y_val_pred_en, s=80, alpha=0.7, label="Elastic Net predictions")

min_val = min(y_val_smoking.min(), y_val_pred_en.min())
max_val = max(y_val_smoking.max(), y_val_pred_en.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.axhline(y_train_smoking.mean(), linestyle=":", label="Mean baseline")

plt.xlabel("Actual Pack Years Smoked")
plt.ylabel("Predicted Pack Years Smoked")
plt.title("Elastic Net: Validation Set Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# Model Comparison
# =====================================================
print("\n" + "="*60)
print("MODEL COMPARISON: GRADIENT DESCENT vs ELASTIC NET")
print("="*60)

comparison_df = pd.DataFrame({
    "Model": ["Gradient Descent", "Elastic Net"],
    "Train_MSE": [
        mean_squared_error(y_train_smoking, y_train_pred),
        mse_train_en
    ],
    "Train_R2": [
        r2_score(y_train_smoking, y_train_pred),
        r2_train_en
    ],
    "Val_MSE": [mse_val, mse_val_en],
    "Val_R2": [r2_val, r2_val_en]
})

print("\n" + comparison_df.to_string(index=False))

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSE comparison
axes[0].bar(["GD Train", "EN Train", "GD Val", "EN Val"],
            [mean_squared_error(y_train_smoking, y_train_pred), 
             mse_train_en,
             mse_val,
             mse_val_en])
axes[0].set_ylabel("MSE")
axes[0].set_title("MSE: Gradient Descent vs Elastic Net")
axes[0].grid(True, alpha=0.3)

# R2 comparison
axes[1].bar(["GD Train", "EN Train", "GD Val", "EN Val"],
            [r2_score(y_train_smoking, y_train_pred),
             r2_train_en,
             r2_val,
             r2_val_en])
axes[1].set_ylabel("R²")
axes[1].set_title("R²: Gradient Descent vs Elastic Net")
axes[1].axhline(0, linestyle="--", color="red", alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================================
# Absolute + relative metrics (using Elastic Net)
# =====================================================
print("\n" + "="*60)
print("ELASTIC NET: DETAILED METRICS")
print("="*60)
metrics_df = compute_regression_metrics(
    y_train_smoking,
    y_train_pred_en,
    y_val_smoking,
    y_val_pred_en
)


# =====================================================
# KMeans clustering on validation PCA
# =====================================================
kmeans_val = KMeans(n_clusters=3, random_state=0)
val_smoking_data["cluster"] = kmeans_val.fit_predict(X_val_smoking)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    val_smoking_data["PC1"],
    val_smoking_data["PC2"],
    c=val_smoking_data["cluster"],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="black"
)
plt.colorbar(scatter, label="Cluster")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("KMeans Clustering on Validation Set PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# Optional UMAP
# =====================================================
try:
    from umap import UMAP

    val_indices = [i for i, idx in enumerate(X_val.index) if idx in val_smoking_data.index]
    X_val_scaled_subset = X_val_scaled[val_indices]

    umap_model = UMAP(n_components=2, random_state=0)
    X_val_umap = umap_model.fit_transform(X_val_scaled_subset)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_val_umap[:, 0],
        X_val_umap[:, 1],
        c=val_smoking_data[smoking_col],
        cmap="coolwarm",
        s=100,
        alpha=0.7,
        edgecolors="black"
    )
    plt.colorbar(scatter, label="Pack Years Smoked")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of Validation Set Colored by Pack Years Smoked")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

except ImportError:
    print("UMAP not installed. Skipping UMAP visualization.")


# =====================================================
# Save outputs
# =====================================================
metrics_path = current_dir / "regression_metrics_summary.csv"
metrics_df.to_csv(metrics_path, index=False)

results_path = current_dir / "validation_smoking_predictions.csv"
val_results = val_smoking_data.copy()
val_results["predicted_pack_years"] = y_val_pred.flatten()
val_results.to_csv(results_path)

print("\nSaved metrics to:", metrics_path)
print("Saved validation predictions to:", results_path)
print("\nAnalysis complete.")