from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Load data
meta = pd.read_csv(r'data\TRAINING_SET_GSE62944_metadata.csv')
expr = pd.read_csv(r'data\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0)

# Filter LUAD + LUSC
meta = meta[meta['cancer_type'].isin(['LUSC'])]

# transpose expression matrix (samples as rows)
expr = expr.T

# Genes
my_genes = [
    'CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3',
    'TGFB1', 'MYC', 'EGFR', 'PIK3CA', 'BRAF', 'CTNNB1',
    'PTEN', 'TP53', 'STK11', 'RB1', 'SMAD4', 'APC', 'ATM'
]

# Build dataset
X_data = expr[my_genes]
df = X_data.join(meta.set_index('sample'), how='inner')

# Clean stage labels
df = df.dropna(subset=['ajcc_pathologic_tumor_stage'])

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

# Features and target
X = df[my_genes]
y = df["stage"]

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Training accuracy
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print(f"In-sample Accuracy (Stage Prediction): {acc:.2f}")

# Save model
joblib.dump(model, "lung_stage_model.pkl")