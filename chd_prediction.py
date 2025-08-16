import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Paths

DATA_PATH = os.path.join('data', 'framingham.csv')
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)


# Load Dataset

df = pd.read_csv(DATA_PATH)

# Exploratory Data Analysis

print('Dataset Shape:', df.shape)
print('\nMissing Values:\n', df.isnull().sum())
print('\nTarget Distribution:\n', df['TenYearCHD'].value_counts(normalize=True))

# Plot Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index, y=missing.values)
    plt.title('Missing Values per Feature')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'missing_values.png'), dpi=150)
    plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()

# Boxplots of selected features
for col in ['age', 'BMI', 'sysBP', 'glucose']:
    if col in df.columns:
        plt.figure()
        sns.boxplot(x='TenYearCHD', y=col, data=df)
        plt.title(f'{col} vs TenYearCHD')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'box_{col}.png'), dpi=150)
        plt.close()


# Data Wrangling

df = df.dropna(subset=['TenYearCHD'])
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = df_imputed.drop('TenYearCHD', axis=1)
y = df_imputed['TenYearCHD']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# SMOTE: Oversample Minority Class

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': acc,
        'recall_class1': recall_1,
        'report': report,
        'confusion_matrix': cm
    }

best_model_name = max(results, key=lambda name: results[name]['recall_class1'])
best_model = results[best_model_name]['model']

print(f'Best Model (based on recall for CHD): {best_model_name}')
print(f'Accuracy: {results[best_model_name]["accuracy"]:.4f}')
print(f'Recall (CHD=1): {results[best_model_name]["recall_class1"]:.4f}\n')
print('Classification Report:')
print(results[best_model_name]['report'])

# Confusion Matrix plot
plt.figure()
sns.heatmap(results[best_model_name]['confusion_matrix'], annot=True, fmt='d')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# ROC Curve
if hasattr(best_model, 'predict_proba'):
    y_score = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    # For models without predict_proba (e.g., some SVM settings), use decision_function
    y_score = best_model.decision_function(X_test_scaled)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# Save a JSON with headline metrics
summary = {
    'best_model': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'recall_chd_1': results[best_model_name]['recall_class1'],
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('\nArtifacts saved to the outputs/ directory.')
