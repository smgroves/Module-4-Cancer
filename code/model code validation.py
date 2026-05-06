from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import joblib


# Load validation data
meta_val = pd.read_csv(r'data\VALIDATION_SET_GSE62944_metadata.csv')
expr_val = pd.read_csv(r'data\VALIDATION_SET_GSE62944_subsample_log2TPM.csv', index_col=0)

# Filter LUAD + LUSC
meta_val = meta_val[meta_val['cancer_type'].isin(['LUAD', 'LUSC'])]
expr_val = expr_val.T

# Genes
my_genes = [
    'CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3',
    'TGFB1', 'MYC', 'EGFR', 'PIK3CA', 'BRAF', 'CTNNB1',
    'PTEN', 'TP53', 'STK11', 'RB1', 'SMAD4', 'APC', 'ATM'
]

# Merge
X_data = expr_val[my_genes]
df = X_data.join(meta_val.set_index('sample'), how='inner')

# Filters by stages
def simplify_stage(stage):
    if pd.isna(stage):
        return None
    stage = str(stage).upper()

    if "I" in stage and "II" not in stage and "III" not in stage and "IV" not in stage:
        return "I"
    elif "II" in stage and "III" not in stage and "IV" not in stage:
        return "II"
    elif "III" in stage and "IV" not in stage:
        return "III"
    elif "IV" in stage:
        return "IV"
    else:
        return None

df["stage"] = df["ajcc_pathologic_tumor_stage"].apply(simplify_stage)
df = df.dropna(subset=["stage"])

# Labels
X = df[my_genes]
y = df["stage"]

# Model
model = joblib.load("lung_stage_model.pkl")

# Predictions
y_pred = model.predict(X)

# Accuracy
val_acc = accuracy_score(y, y_pred)
print(f"Out-of-sample Accuracy (Stage Prediction): {val_acc:.2f}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred,
    cmap=plt.cm.Blues,
    ax=ax,
    normalize=None
)

ax.set_title("Confusion Matrix (Stage Prediction - Validation)")
plt.tight_layout()
plt.show()