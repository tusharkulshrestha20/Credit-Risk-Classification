"""
Credit Risk Classification Model
=================================
Predicts probability of loan default using supervised machine learning.

Dataset : LendingClub-style structured loan data (5,000 records)
          Swap the synthetic data block for a real CSV via pd.read_csv()

Techniques:
  - Feature Engineering (DTI ratio, credit utilisation, composite risk score)
  - Dimensionality Reduction (PCA — 95% variance threshold)
  - Ridge Regularisation (LogisticRegression, penalty='l2')
  - Lasso Regularisation (LogisticRegression, penalty='l1')
  - Logistic Regression, Random Forest, Gradient Boosting
  - Stratified K-Fold Cross-Validation (k=5)
  - Precision / Recall / F1 on imbalanced default class
  - Feature Importance (Random Forest)

Author  : Tushar Kulshrestha
GitHub  : github.com/tusharkulshrestha20
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)


# ── 1. DATA ───────────────────────────────────────────────────────────────────
# Replace this block with:
#   df = pd.read_csv("loan_data.csv")
#   y  = df.pop("default").values
#   X  = df[FEATURES].values

def make_loan_data(n=5000):
    X_raw, y = make_classification(
        n_samples=n, n_features=15, n_informative=10,
        n_redundant=3, weights=[0.82, 0.18],   # ~18% default rate
        flip_y=0.01, random_state=SEED
    )
    cols = [
        'loan_amnt', 'int_rate', 'installment', 'annual_inc',
        'dti',               # debt-to-income ratio
        'delinq_2yrs',       # delinquencies in past 2 years
        'inq_last_6mths',    # hard credit enquiries
        'open_acc',          # open credit lines
        'revol_bal',         # revolving balance
        'revol_util',        # credit utilisation %
        'total_acc',
        'out_prncp',         # outstanding principal
        'total_pymnt',
        'last_pymnt_amnt',
        'credit_age_months'
    ]
    df = pd.DataFrame(X_raw, columns=cols)
    df['default'] = y
    # Scale to realistic ranges
    df['loan_amnt']         = (df['loan_amnt']         * 5000  + 15000).clip(1000,   40000)
    df['int_rate']          = (df['int_rate']           * 5    + 12   ).clip(5,       30)
    df['annual_inc']        = (df['annual_inc']         * 20000 + 60000).clip(20000, 200000)
    df['dti']               = (df['dti']                * 8    + 18   ).clip(0,       50)
    df['revol_util']        = (df['revol_util']         * 25   + 50   ).clip(0,      100)
    df['delinq_2yrs']       = (df['delinq_2yrs'].abs()  * 1.5        ).clip(0,       10).round()
    df['credit_age_months'] = (df['credit_age_months']  * 40   + 80  ).clip(12,     300)
    return df


print("=" * 62)
print("  CREDIT RISK CLASSIFICATION MODEL")
print("  Author: Tushar Kulshrestha")
print("=" * 62)

df = make_loan_data(5000)
print(f"\n[DATA]  Rows: {len(df):,}  |  Default rate: {df['default'].mean():.1%}")


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
df['pymnt_to_loan']       = df['total_pymnt']  / (df['loan_amnt']     + 1)
df['inc_to_installment']  = df['annual_inc']   / (df['installment'] * 12 + 1)
df['risk_score']          = df['int_rate'] * df['dti'] / (df['annual_inc'] / 10000 + 1)

FEATURES = [
    'loan_amnt', 'int_rate', 'dti', 'annual_inc', 'delinq_2yrs',
    'inq_last_6mths', 'revol_util', 'credit_age_months',
    'pymnt_to_loan', 'inc_to_installment', 'risk_score'
]

X = df[FEATURES].values
y = df['default'].values
print(f"[FEAT]  {len(FEATURES)} features (incl. DTI ratio, credit utilisation, composite risk score)")


# ── 3. PREPROCESSING ──────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=SEED)   # retain 95% variance
X_pca = pca.fit_transform(X_scaled)
n_components = X_pca.shape[1]
print(f"[PCA]   {len(FEATURES)} features → {n_components} components (95% variance retained)")


# ── 4. MODELS ─────────────────────────────────────────────────────────────────
models = {
    "Logistic (L2 / Ridge)": LogisticRegression(
        penalty='l2', C=0.5, max_iter=1000, random_state=SEED),
    "Logistic (L1 / Lasso)": LogisticRegression(
        penalty='l1', C=0.5, solver='saga', max_iter=1000, random_state=SEED),
    "Random Forest":          RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=SEED),
    "Gradient Boosting":      GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=SEED),
}


# ── 5. STRATIFIED K-FOLD CROSS-VALIDATION ─────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ['precision', 'recall', 'f1']

print("\n[CV]    5-Fold Stratified Cross-Validation Results")
print(f"  {'Model':<26}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
print("  " + "-" * 58)

results = {}
for name, model in models.items():
    X_in  = X_pca if "Logistic" in name else X_scaled
    cv_res = cross_validate(model, X_in, y, cv=cv, scoring=scoring)
    p  = cv_res['test_precision'].mean()
    r  = cv_res['test_recall'].mean()
    f1 = cv_res['test_f1'].mean()
    results[name] = dict(precision=p, recall=r, f1=f1)
    print(f"  {name:<26}  {p:>10.3f}  {r:>8.3f}  {f1:>8.3f}")


# ── 6. FINAL FIT — RANDOM FOREST ─────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED)
rf.fit(X_scaled, y)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
top3        = importances.head(3)

y_pred = rf.predict(X_scaled)
cm     = confusion_matrix(y, y_pred)
cr     = classification_report(y, y_pred, target_names=['Performing', 'Default'], output_dict=True)
rf_f1   = cr['Default']['f1-score']
rf_prec = cr['Default']['precision']
rf_rec  = cr['Default']['recall']
rf_acc  = cr['accuracy']

print(f"\n[RF]    Final model performance (default class)")
print(f"  Accuracy:  {rf_acc:.1%}")
print(f"  Precision: {rf_prec:.1%}  — of flagged defaults, how many were real")
print(f"  Recall:    {rf_rec:.1%}  — of actual defaults, how many were caught")
print(f"  F1-Score:  {rf_f1:.3f}")
print(f"\n[FEAT IMPORTANCE] Top predictors: {', '.join(top3.index.tolist())}")


# ── 7. RESULTS FIGURE ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0e1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

GOLD  = '#c9a84c'
WHITE = '#e8e8e8'
RED   = '#e05c5c'
GREEN = '#4caf50'
BLUE  = '#4a90d9'
BG    = '#161b22'


def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')


# (A) Class distribution
ax0 = fig.add_subplot(gs[0, 0])
style_ax(ax0, "Class Distribution")
counts = pd.Series(y).value_counts()
bars = ax0.bar(['Performing', 'Default'], [counts[0], counts[1]],
               color=[GREEN, RED], width=0.5, edgecolor='none')
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
             f'{val:,}\n({val / len(y):.0%})',
             ha='center', va='bottom', color=WHITE, fontsize=8)
ax0.set_ylabel('Count', color=WHITE, fontsize=8)
ax0.yaxis.label.set_color(WHITE)

# (B) Feature importance
ax1 = fig.add_subplot(gs[0, 1:])
style_ax(ax1, "Random Forest — Feature Importance")
imp_sorted  = importances[::-1]
bar_colors  = [GOLD if i >= len(imp_sorted) - 3 else BLUE for i in range(len(imp_sorted))]
ax1.barh(imp_sorted.index, imp_sorted.values, color=bar_colors, edgecolor='none')
ax1.set_xlabel('Importance', color=WHITE, fontsize=8)
ax1.xaxis.label.set_color(WHITE)
for val, nm in zip(imp_sorted.values, imp_sorted.index):
    ax1.text(val + 0.001, list(imp_sorted.index).index(nm),
             f'{val:.3f}', va='center', color=WHITE, fontsize=7)

# (C) PCA explained variance
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "PCA — Cumulative Explained Variance")
pca_full = PCA(random_state=SEED).fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
ax2.plot(range(1, len(cumvar) + 1), cumvar, color=GOLD, lw=2, marker='o', markersize=4)
ax2.axhline(0.95, color=RED,   lw=1, linestyle='--', label='95% threshold')
ax2.axvline(n_components, color=GREEN, lw=1, linestyle='--', label=f'{n_components} components')
ax2.set_xlabel('Components', color=WHITE, fontsize=8)
ax2.set_ylabel('Cumulative Variance', color=WHITE, fontsize=8)
ax2.xaxis.label.set_color(WHITE)
ax2.yaxis.label.set_color(WHITE)
ax2.legend(fontsize=7, facecolor=BG, labelcolor=WHITE, edgecolor='#333')

# (D) Model F1 comparison
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "Model Comparison — F1 Score (Default Class)")
names_short = ['LR-Ridge', 'LR-Lasso', 'Rand\nForest', 'Grad\nBoost']
f1s         = [results[k]['f1'] for k in models]
bar_cols    = [BLUE, BLUE, GOLD, GREEN]
b3 = ax3.bar(names_short, f1s, color=bar_cols, edgecolor='none', width=0.5)
for bar, val in zip(b3, f1s):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', va='bottom', color=WHITE, fontsize=8)
ax3.set_ylim(0, max(f1s) * 1.18)
ax3.set_ylabel('F1 Score', color=WHITE, fontsize=8)
ax3.yaxis.label.set_color(WHITE)

# (E) Confusion matrix
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "Confusion Matrix — Random Forest")
ax4.imshow(cm, interpolation='nearest', cmap='Blues')
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black',
                 fontsize=11, fontweight='bold')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Pred\nPerforming', 'Pred\nDefault'], color=WHITE, fontsize=7)
ax4.set_yticklabels(['Actual\nPerforming', 'Actual\nDefault'], color=WHITE, fontsize=7)

# Title bar
fig.text(0.5, 0.97, 'Credit Risk Classification Model  |  Tushar Kulshrestha',
         ha='center', va='top', color=WHITE, fontsize=13, fontweight='bold')
fig.text(0.5, 0.935,
         f'5,000 loans  ·  18% default rate  ·  11 engineered features  ·  5-Fold CV  ·  RF F1: {rf_f1:.3f}',
         ha='center', va='top', color='#aaa', fontsize=9)

plt.savefig('credit_risk_results.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\n[PLOT]  Saved → credit_risk_results.png")
print("[DONE]  Run complete.\n")
