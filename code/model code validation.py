# Filters data
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, roc_curve



meta_val = pd.read_csv(r'data\VALIDATION_SET_GSE62944_metadata.csv')
expr_val = pd.read_csv(r'data\VALIDATION_SET_GSE62944_subsample_log2TPM.csv', index_col=0)


# Filter for lung cancer types LUAD and LUSC
meta_val = meta_val[meta_val['cancer_type'].isin(['LUAD', 'LUSC'])]
expr_val = expr_val.T


my_genes = [
    'CD274', 'CTLA4', 'LAG3', 'HLA-A', 'B2M', 'STAT3', 
    'TGFB1', 'MYC', 'EGFR', 'PIK3CA', 'BRAF', 'CTNNB1',
    'PTEN', 'TP53', 'STK11', 'RB1', 'SMAD4', 'APC', 'ATM'
]

X_data = expr_val[my_genes]
df = X_data.join(meta_val.set_index('sample'), how='inner')

labels = ['cancer_type', 'ajcc_pathologic_tumor_stage']
df_clean = df[my_genes + labels]



# model
# defines x and y
X = df_clean[my_genes]
y = df_clean['cancer_type']


# logistic regression model from training code
import joblib
model = joblib.load('lung_cancer_model.pkl')


# out of sample error
y_val_pred = model.predict(X)
val_acc = accuracy_score(y, y_val_pred)

print(f"Out-of-sample Accuracy (Validation): {val_acc:.2f}")



fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# predictions
y_val_pred = model.predict(X)
val_acc = accuracy_score(y, y_val_pred)

print(f"Out-of-sample Accuracy (Validation): {val_acc:.2f}")

# confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y,
    y_val_pred,
    cmap=plt.cm.Blues,
    ax=ax[0]
)
ax[0].set_title("Confusion Matrix (Validation)")

# ROC curve
y_val_binary = y.map({'LUAD': 0, 'LUSC': 1})
y_score = model.predict_proba(X)[:, 1]

fpr, tpr, _ = roc_curve(y_val_binary, y_score)
roc_auc = auc(fpr, tpr)

ax[1].plot(fpr, tpr, color='darkorange',
           label=f'ROC curve (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('ROC Curve')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()