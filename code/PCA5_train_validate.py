import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'

    # =====================================================
    # LOAD TRAINING SET
    # =====================================================
    print('='*60)
    print('LOADING TRAINING SET')
    print('='*60)
    expr_path_train = data_dir / 'TRAINING_SET_GSE62944_subsample_log2TPM.csv'
    metadata_path_train = data_dir / 'TRAINING_SET_GSE62944_metadata.csv'

    print(f'Loading training expression data from: {expr_path_train}')
    print(f'Loading training metadata from: {metadata_path_train}')

    data_train = pd.read_csv(expr_path_train, index_col=0, header=0)
    metadata_train = pd.read_csv(metadata_path_train, index_col=0, header=0)

    print(f'Training expression data shape: {data_train.shape}')
    print(f'Training metadata shape: {metadata_train.shape}')

    # Filter to LUSC samples
    cancer_type = 'LUSC'
    cancer_samples_train = metadata_train[metadata_train['cancer_type'] == cancer_type].index
    shared_samples_train = cancer_samples_train.intersection(data_train.columns)
    LUSC_data_train = data_train.loc[:, shared_samples_train]
    LUSC_metadata_train = metadata_train.loc[shared_samples_train].copy()

    # =====================================================
    # LOAD VALIDATION SET
    # =====================================================
    print('\n' + '='*60)
    print('LOADING VALIDATION SET')
    print('='*60)
    expr_path_val = data_dir / 'VALIDATION_SET_GSE62944_subsample_log2TPM.csv'
    metadata_path_val = data_dir / 'VALIDATION_SET_GSE62944_metadata.csv'

    print(f'Loading validation expression data from: {expr_path_val}')
    print(f'Loading validation metadata from: {metadata_path_val}')

    data_val = pd.read_csv(expr_path_val, index_col=0, header=0)
    metadata_val = pd.read_csv(metadata_path_val, index_col=0, header=0)

    print(f'Validation expression data shape: {data_val.shape}')
    print(f'Validation metadata shape: {metadata_val.shape}')

    # Filter to LUSC samples
    cancer_samples_val = metadata_val[metadata_val['cancer_type'] == cancer_type].index
    shared_samples_val = cancer_samples_val.intersection(data_val.columns)
    LUSC_data_val = data_val.loc[:, shared_samples_val]
    LUSC_metadata_val = metadata_val.loc[shared_samples_val].copy()

    # =====================================================
    # DEFINE GENE SET
    # =====================================================
    desired_gene_list = [
        'ATP2B2', 'FAM155A', 'KCNIP3', 'TAC1', 'ALX4', 'TTLL6', 'MUC12', 
        'IFI6', 'HRK', 'C15orf41', 'PRSS53', 'CPE', 'GRK7', 'KIAA1671', 
        'NEUROG3', 'OR4F17', 'SLC7A3', 'UBXN10', 'UNC93B1', 'USP2'
    ]

    gene_list = []
    for gene in desired_gene_list:
        if gene in LUSC_data_train.index:
            gene_list.append(gene)
        else:
            print(f"Warning: {gene} not found in training data")

    print(f'\nGenes used: {gene_list}')

    # =====================================================
    # PREPARE TRAINING DATA
    # =====================================================
    print('\n' + '='*60)
    print('PREPARING TRAINING DATA')
    print('='*60)

    X_train = LUSC_data_train.loc[gene_list].T
    LUSC_merged_train = X_train.merge(LUSC_metadata_train, left_index=True, right_index=True)
    
    X_train = LUSC_merged_train[gene_list].copy()
    X_train = X_train.dropna(axis=0)
    LUSC_merged_train = LUSC_merged_train.loc[X_train.index].copy()

    # Fit scaler and PCA on TRAINING DATA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    LUSC_merged_train['PC1'] = X_train_pca[:, 0]
    LUSC_merged_train['PC2'] = X_train_pca[:, 1]

    print(f'\nTraining set size: {len(LUSC_merged_train)} LUSC samples')
    print(f'Explained variance: {pca.explained_variance_ratio_}')

    # =====================================================
    # PREPARE VALIDATION DATA
    # =====================================================
    print('\n' + '='*60)
    print('PREPARING VALIDATION DATA')
    print('='*60)

    X_val = LUSC_data_val.loc[gene_list].T
    LUSC_merged_val = X_val.merge(LUSC_metadata_val, left_index=True, right_index=True)
    
    X_val = LUSC_merged_val[gene_list].copy()
    X_val = X_val.dropna(axis=0)
    LUSC_merged_val = LUSC_merged_val.loc[X_val.index].copy()

    # Transform validation data using TRAINING scaler and PCA
    X_val_scaled = scaler.transform(X_val)
    X_val_pca = pca.transform(X_val_scaled)

    LUSC_merged_val['PC1'] = X_val_pca[:, 0]
    LUSC_merged_val['PC2'] = X_val_pca[:, 1]

    print(f'Validation set size: {len(LUSC_merged_val)} LUSC samples')

    # =====================================================
    # LOAD SMOKING DATA AND TRAIN MODEL
    # =====================================================
    print('\n' + '='*60)
    print('GRADIENT DESCENT: TRAIN ON TRAINING SET')
    print('='*60)

    smoking_path = data_dir / 'packs_per_year_smoked.csv'
    
    if smoking_path.exists():
        smoking_df = pd.read_csv(smoking_path, index_col=0, header=0)
        smoking_col = 'tobacco_smoking_pack_years_smoked'
        
        # Get training set smoking data
        LUSC_train_smoking = LUSC_merged_train.copy()
        LUSC_train_smoking = LUSC_train_smoking.merge(
            smoking_df, left_index=True, right_index=True, how='left'
        )
        
        if smoking_col in LUSC_train_smoking.columns:
            LUSC_train_smoking[smoking_col] = pd.to_numeric(
                LUSC_train_smoking[smoking_col], errors='coerce'
            )
            
            # Filter to samples with smoking data
            train_smoking_data = LUSC_train_smoking.dropna(subset=[smoking_col]).copy()
            
            if len(train_smoking_data) > 0:
                print(f'Found smoking data for {len(train_smoking_data)} training LUSC samples')
                print(f'Smoking pack years range: {train_smoking_data[smoking_col].min():.1f} - {train_smoking_data[smoking_col].max():.1f}')
                
                # Plot training set PCA colored by smoking
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(
                    train_smoking_data['PC1'], 
                    train_smoking_data['PC2'], 
                    c=train_smoking_data[smoking_col],
                    cmap='viridis',
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5
                )
                plt.colorbar(scatter, label='Pack Years Smoked')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
                plt.title('Training Set: PCA Colored by Tobacco Smoking Pack Years')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                # ===== TRAIN MODEL ON ALL TRAINING DATA =====
                X_train_smoking = train_smoking_data[['PC1', 'PC2']].values
                y_train_smoking = train_smoking_data[[smoking_col]].values
                
                print(f'\nTraining on ALL {len(X_train_smoking)} training samples with smoking data')
                
                # Standardize features
                X_train_mean = X_train_smoking.mean(axis=0)
                X_train_std = X_train_smoking.std(axis=0)
                X_train_std[X_train_std == 0] = 1
                
                X_train_scaled_smk = (X_train_smoking - X_train_mean) / X_train_std
                
                # Add intercept term
                X_train_b = np.c_[np.ones((X_train_scaled_smk.shape[0], 1)), X_train_scaled_smk]
                
                # Initialize weights
                theta = np.zeros((X_train_b.shape[1], 1))
                
                # Gradient descent settings
                learning_rate = 0.01
                n_iterations = 2000
                m = len(X_train_b)
                
                # Gradient descent on FULL TRAINING DATA
                for i in range(n_iterations):
                    gradients = (2 / m) * X_train_b.T @ (X_train_b @ theta - y_train_smoking)
                    theta = theta - learning_rate * gradients
                
                print(f'Weights (intercept, PC1, PC2): {theta.ravel()}')
                
                # ===== EVALUATE ON VALIDATION SET =====
                print('\n' + '='*60)
                print('EVALUATING ON VALIDATION SET')
                print('='*60)
                
                
                # Get validation set smoking data
                LUSC_val_smoking = LUSC_merged_val.copy()
                LUSC_val_smoking = LUSC_val_smoking.merge(
                    smoking_df, left_index=True, right_index=True, how='left'
                )
                
                if smoking_col in LUSC_val_smoking.columns:
                    LUSC_val_smoking[smoking_col] = pd.to_numeric(
                        LUSC_val_smoking[smoking_col], errors='coerce'
                    )
                    
                    val_smoking_data = LUSC_val_smoking.dropna(subset=[smoking_col]).copy()
                    
                    if len(val_smoking_data) > 0:
                        print(f'Found smoking data for {len(val_smoking_data)} validation LUSC samples')
                        print(f'Smoking pack years range: {val_smoking_data[smoking_col].min():.1f} - {val_smoking_data[smoking_col].max():.1f}')
                        
                        # Plot validation set PCA colored by smoking
                        plt.figure(figsize=(10, 6))
                        scatter = plt.scatter(
                            val_smoking_data['PC1'], 
                            val_smoking_data['PC2'], 
                            c=val_smoking_data[smoking_col],
                            cmap='viridis',
                            s=100,
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=0.5
                        )
                        plt.colorbar(scatter, label='Pack Years Smoked')
                        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
                        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
                        plt.title('Validation Set: PCA Colored by Tobacco Smoking Pack Years')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        
                        # Get validation predictions
                        X_val_smoking = val_smoking_data[['PC1', 'PC2']].values
                        y_val_smoking = val_smoking_data[[smoking_col]].values
                        
                        # Apply training standardization
                        X_val_scaled_smk = (X_val_smoking - X_train_mean) / X_train_std
                        
                        # Add intercept term
                        X_val_b = np.c_[np.ones((X_val_scaled_smk.shape[0], 1)), X_val_scaled_smk]
                        
                        # Predictions on validation data
                        y_val_pred = X_val_b @ theta
                        
                        # Metrics on validation data
                        mse_val = mean_squared_error(y_val_smoking, y_val_pred)
                        r2_val = r2_score(y_val_smoking, y_val_pred)
                        
                        print(f'\nValidation Results (Model trained on training set):')
                        print(f'Validation MSE: {mse_val:.4f}')
                        print(f'Validation R^2: {r2_val:.4f}')
                        
                        # Actual vs predicted plot
                        plt.figure(figsize=(8, 6))
                        plt.scatter(y_val_smoking, y_val_pred, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
                        plt.xlabel('Actual Pack Years Smoked')
                        plt.ylabel('Predicted Pack Years Smoked')
                        plt.title('Gradient Descent Regression: Training Set Model on Validation Data')
                        
                        # Line of perfect prediction
                        min_val = min(y_val_smoking.min(), y_val_pred.min())
                        max_val = max(y_val_smoking.max(), y_val_pred.max())
                        plt.plot([min_val, max_val], [min_val, max_val], 
                                linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        
                        # =====================================================
                        # KMeans and UMAP on Validation Set
                        # =====================================================
                        print('\n' + '='*60)
                        print('KMeans Clustering on Validation Set')
                        print('='*60)
                        
                        # KMeans with k=3
                        kmeans_val = KMeans(n_clusters=3, random_state=0)
                        val_smoking_data['cluster'] = kmeans_val.fit_predict(X_val_smoking)
                        
                        plt.figure(figsize=(10, 6))
                        scatter = plt.scatter(
                            val_smoking_data['PC1'],
                            val_smoking_data['PC2'],
                            c=val_smoking_data['cluster'],
                            cmap='viridis',
                            s=100,
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=0.5
                        )
                        plt.colorbar(scatter, label='Cluster')
                        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
                        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
                        plt.title('KMeans Clustering (k=3) on Validation Set PCA')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        
                        # UMAP
                        print('\nGenerating UMAP visualization for validation set...')
                        try:
                            from umap import UMAP
                            
                            # Get the subset of X_val_scaled corresponding to validation samples
                            val_indices = [i for i, idx in enumerate(X_val.index) if idx in val_smoking_data.index]
                            X_val_scaled_subset = X_val_scaled[val_indices]
                            
                            # Apply UMAP
                            umap_model = UMAP(n_components=2, random_state=0)
                            X_val_umap = umap_model.fit_transform(X_val_scaled_subset)
                            
                            y_smoking_vals = val_smoking_data[smoking_col].values.astype(float)
                            cluster_vals = val_smoking_data['cluster'].values.astype(int)
                            
                            # UMAP colored by smoking
                            plt.figure(figsize=(10, 6))
                            scatter = plt.scatter(
                                X_val_umap[:, 0],
                                X_val_umap[:, 1],
                                c=y_smoking_vals,
                                cmap='coolwarm',
                                s=100,
                                alpha=0.6,
                                edgecolors='black',
                                linewidth=0.5
                            )
                            plt.colorbar(scatter, label='Pack Years Smoked')
                            plt.xlabel('UMAP 1')
                            plt.ylabel('UMAP 2')
                            plt.title('UMAP of Validation Set\nColored by Pack Years Smoked')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.show()
                            
                            # UMAP colored by cluster
                            plt.figure(figsize=(10, 6))
                            scatter = plt.scatter(
                                X_val_umap[:, 0],
                                X_val_umap[:, 1],
                                c=cluster_vals,
                                cmap='viridis',
                                s=100,
                                alpha=0.6,
                                edgecolors='black',
                                linewidth=0.5
                            )
                            plt.colorbar(scatter, label='Cluster')
                            plt.xlabel('UMAP 1')
                            plt.ylabel('UMAP 2')
                            plt.title('UMAP of Validation Set\nColored by KMeans Cluster')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.show()
                            
                        except ImportError:
                            print('UMAP not installed. Skipping UMAP visualization.')
                    
                    else:
                        print('No validation samples with smoking data')
            
            else:
                print('No training samples with smoking data')
    
    else:
        print(f'Smoking data file not found at: {smoking_path}')
compute_regression_metrics(
    y_train_smoking,
    y_train_pred,
    y_val_smoking,
    y_val_pred
)

if __name__ == '__main__':
    main()

def compute_regression_metrics(y_train_true, y_train_pred, y_val_true, y_val_pred):
    import numpy as np
    from sklearn.metrics import r2_score

    # Flatten
    y_train_true = y_train_true.flatten()
    y_train_pred = y_train_pred.flatten()
    y_val_true = y_val_true.flatten()
    y_val_pred = y_val_pred.flatten()

    # Residuals
    train_residuals = y_train_true - y_train_pred
    val_residuals = y_val_true - y_val_pred

    # Absolute metrics
    train_mae = np.mean(np.abs(train_residuals))
    val_mae = np.mean(np.abs(val_residuals))

    train_rmse = np.sqrt(np.mean(train_residuals**2))
    val_rmse = np.sqrt(np.mean(val_residuals**2))

    # R2
    train_r2 = r2_score(y_train_true, y_train_pred)
    val_r2 = r2_score(y_val_true, y_val_pred)

    # Baseline (mean)
    baseline_train = np.full_like(y_train_true, y_train_true.mean())
    baseline_val = np.full_like(y_val_true, y_train_true.mean())

    baseline_train_mae = np.mean(np.abs(y_train_true - baseline_train))
    baseline_val_mae = np.mean(np.abs(y_val_true - baseline_val))

    # Improvement
    train_improve = 100 * (baseline_train_mae - train_mae) / baseline_train_mae
    val_improve = 100 * (baseline_val_mae - val_mae) / baseline_val_mae

    print('\n==== REGRESSION METRICS ====')
    print('\nTRAINING (in-sample)')
    print(f'MAE: {train_mae:.3f}')
    print(f'RMSE: {train_rmse:.3f}')
    print(f'R²: {train_r2:.3f}')
    print(f'Improvement vs baseline: {train_improve:.2f}%')

    print('\nVALIDATION (out-of-sample)')
    print(f'MAE: {val_mae:.3f}')
    print(f'RMSE: {val_rmse:.3f}')
    print(f'R²: {val_r2:.3f}')
    print(f'Improvement vs baseline: {val_improve:.2f}%')