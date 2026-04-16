"""
=============================================================================
  AI FOR MATERIALS SCIENCE — BANDGAP PREDICTION PROJECT
  Based on: "Machine Learning Prediction for Bandgaps of Inorganic Materials"
            Wu et al., ES Mater. Manuf., 2020, 9, 34-39

  Syllabus Coverage:
    Unit 1: Introduction to AI/ML concepts
    Unit 2: Data Handling, Preprocessing, Feature Engineering
    Unit 3: Supervised Learning — Regression & Classification
    Unit 4: Unsupervised Learning — Clustering, PCA, t-SNE
    Unit 5: Deep Learning — Neural Networks
  =============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 1 — DATA GENERATION
# We simulate a realistic dataset inspired by the paper's 3896 inorganic
# compounds, using physically-motivated features from the periodic table.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  AI FOR MATERIALS SCIENCE — BANDGAP PREDICTION PROJECT")
print("=" * 70)
print("\n[UNIT 1] Generating synthetic materials dataset (3896 compounds)...")

np.random.seed(42)
N = 3896  # match paper's dataset size

# Periodic table-inspired elemental features (composition-based descriptors)
# Each row = one compound; features mimic real matminer/magpie descriptors

electronegativity_mean   = np.random.uniform(1.0, 4.0, N)
electronegativity_range  = np.random.uniform(0.0, 2.5, N)
atomic_radius_mean       = np.random.uniform(0.5, 2.5, N)
atomic_radius_std        = np.random.uniform(0.0, 0.8, N)
ionization_energy_mean   = np.random.uniform(4.0, 14.0, N)
valence_electrons_mean   = np.random.uniform(1.0, 8.0, N)
valence_electrons_std    = np.random.uniform(0.0, 3.0, N)
melting_point_mean       = np.random.uniform(200, 3500, N)
melting_point_std        = np.random.uniform(0, 500, N)
period_mean              = np.random.uniform(1.5, 5.5, N)
group_mean               = np.random.uniform(1.0, 16.0, N)
n_elements               = np.random.randint(2, 6, N).astype(float)

# Physically-motivated bandgap formula (mimics the nonlinear relationships
# found in the paper: ionic/covalent character, electronegativity differences)
bandgap = (
    2.5 * electronegativity_range
    + 1.2 * (ionization_energy_mean / 10)
    - 0.8 * (valence_electrons_mean / 4)
    + 0.5 * (period_mean / 3)
    - 0.3 * (group_mean / 8)
    + 0.4 * atomic_radius_std
    + np.random.normal(0, 0.4, N)  # measurement noise
)
bandgap = np.clip(bandgap, 0.05, 12.0)  # physical range [0, 12 eV]

# Assign material family labels for classification tasks
def assign_family(bg):
    if bg < 0.5:   return "Metal/Near-Metal"
    elif bg < 1.5: return "Narrow Semiconductor"
    elif bg < 3.0: return "Semiconductor"
    elif bg < 6.0: return "Wide Bandgap"
    else:          return "Insulator"

family_labels = [assign_family(bg) for bg in bandgap]

# Assemble DataFrame
feature_names = [
    "electronegativity_mean", "electronegativity_range",
    "atomic_radius_mean", "atomic_radius_std",
    "ionization_energy_mean", "valence_electrons_mean",
    "valence_electrons_std", "melting_point_mean",
    "melting_point_std", "period_mean", "group_mean", "n_elements"
]

X_raw = np.column_stack([
    electronegativity_mean, electronegativity_range,
    atomic_radius_mean, atomic_radius_std,
    ionization_energy_mean, valence_electrons_mean,
    valence_electrons_std, melting_point_mean,
    melting_point_std, period_mean, group_mean, n_elements
])

df = pd.DataFrame(X_raw, columns=feature_names)
df['bandgap_eV'] = bandgap
df['material_family'] = family_labels

print(f"  ✓ Dataset: {N} compounds, {len(feature_names)} features")
print(f"  ✓ Bandgap range: {bandgap.min():.2f} – {bandgap.max():.2f} eV")
print(f"  ✓ Material families: {df['material_family'].nunique()}")
print(df['material_family'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 2 — DATA PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[UNIT 2] Data Preprocessing & Feature Engineering")
print("─" * 70)

# 2a. Inject & handle missing values (simulate real-world dirty data)
df_dirty = df.copy()
missing_mask = np.random.random(df_dirty[feature_names].shape) < 0.02  # 2% missing
df_dirty[feature_names] = df_dirty[feature_names].where(~missing_mask, other=np.nan)

missing_count = df_dirty[feature_names].isna().sum().sum()
print(f"\n  Injected {missing_count} missing values ({100*missing_count/(N*len(feature_names)):.1f}%)")

# Impute with column medians
df_clean = df_dirty.copy()
for col in feature_names:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

print(f"  ✓ Missing values after imputation: {df_clean[feature_names].isna().sum().sum()}")

# 2b. Outlier detection (IQR method)
outlier_counts = {}
for col in feature_names:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = outliers

total_outliers = sum(outlier_counts.values())
print(f"  Detected {total_outliers} outliers across all features (IQR method)")

# 2c. Feature engineering — create physically meaningful derived features
df_clean['EN_x_IE']    = df_clean['electronegativity_mean'] * df_clean['ionization_energy_mean']
df_clean['radius_IE']  = df_clean['atomic_radius_mean'] / (df_clean['ionization_energy_mean'] + 1e-6)
df_clean['valence_EN'] = df_clean['valence_electrons_mean'] * df_clean['electronegativity_range']
df_clean['period_group_ratio'] = df_clean['period_mean'] / (df_clean['group_mean'] + 1e-6)

engineered_features = feature_names + ['EN_x_IE', 'radius_IE', 'valence_EN', 'period_group_ratio']
print(f"  ✓ Feature engineering: {len(feature_names)} → {len(engineered_features)} features")

# 2d. Normalization / Standardization
X = df_clean[engineered_features].values
y = df_clean['bandgap_eV'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (90/10 to match the paper)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)
print(f"  ✓ Train: {len(X_train)} | Test: {len(X_test)} (90/10 split, as in paper)")

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 3 — SUPERVISED LEARNING: REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[UNIT 3A] Supervised Learning — Regression Models")
print("─" * 70)

def evaluate_regression(model, X_tr, X_te, y_tr, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"  {name:<28s} | MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")
    return {"name": name, "model": model, "y_pred": y_pred,
            "MAE": mae, "RMSE": rmse, "R2": r2}

regression_results = []

# 3.1 Ridge Regression
ridge = Ridge(alpha=1.0)
regression_results.append(evaluate_regression(ridge, X_train, X_test, y_train, y_test, "Ridge Regression"))

# 3.2 Lasso Regression
lasso = Lasso(alpha=0.01)
regression_results.append(evaluate_regression(lasso, X_train, X_test, y_train, y_test, "Lasso Regression"))

# 3.3 Support Vector Regression
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
regression_results.append(evaluate_regression(svr, X_train, X_test, y_train, y_test, "SVR (RBF kernel)"))

# 3.4 Random Forest (best single model in the paper)
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
regression_results.append(evaluate_regression(rf, X_train, X_test, y_train, y_test, "Random Forest"))

# 3.5 Gradient Boosting (XGBoost-style)
gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
regression_results.append(evaluate_regression(gbm, X_train, X_test, y_train, y_test, "Gradient Boosting"))

# 3.6 Gaussian Process Regression (on subset due to O(n^3) complexity)
gp_idx = np.random.choice(len(X_train), 800, replace=False)
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)
gpr.fit(X_train[gp_idx], y_train[gp_idx])
y_pred_gp = gpr.predict(X_test)
gp_result = {
    "name": "Gaussian Process",
    "model": gpr,
    "y_pred": y_pred_gp,
    "MAE":  mean_absolute_error(y_test, y_pred_gp),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_gp)),
    "R2":   r2_score(y_test, y_pred_gp)
}
regression_results.append(gp_result)
print(f"  {'Gaussian Process':<28s} | MAE={gp_result['MAE']:.3f} | RMSE={gp_result['RMSE']:.3f} | R²={gp_result['R2']:.3f}")

# 3.7 Neural Network (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                   max_iter=500, random_state=42, learning_rate_init=0.001)
regression_results.append(evaluate_regression(mlp, X_train, X_test, y_train, y_test, "MLP Neural Network"))

# 3.8 Feature importance from Random Forest
feat_importance = pd.Series(rf.feature_importances_, index=engineered_features).sort_values(ascending=False)
print(f"\n  Top 5 Features (Random Forest Importance):")
for feat, imp in feat_importance.head(5).items():
    print(f"    {feat:<30s} {imp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 3B — SUPERVISED LEARNING: CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[UNIT 3B] Supervised Learning — Classification Models")
print("─" * 70)

le = LabelEncoder()
y_class = le.fit_transform(df_clean['material_family'].values)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_class, test_size=0.1, random_state=42, stratify=y_class
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "SVM (RBF)":           __import__('sklearn.svm', fromlist=['SVC']).SVC(kernel='rbf', C=10, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42)
}

from sklearn.svm import SVC

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "SVM (RBF)":           SVC(kernel='rbf', C=10, random_state=42, probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

clf_results = {}
for name, clf in clf_models.items():
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test_c, y_pred_c)
    clf_results[name] = acc
    print(f"  {name:<28s} | Accuracy = {acc:.3f}")

best_clf_name = max(clf_results, key=clf_results.get)
best_clf = clf_models[best_clf_name]
print(f"\n  Best Classifier: {best_clf_name} (Acc={clf_results[best_clf_name]:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 4 — UNSUPERVISED LEARNING: CLUSTERING & DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[UNIT 4] Unsupervised Learning — Clustering & Dimensionality Reduction")
print("─" * 70)

# 4.1 PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum() * 100
print(f"\n  PCA: 2 components explain {explained_var:.1f}% of variance")

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumsum, 0.95) + 1
print(f"  PCA: {n_95} components needed for 95% explained variance")

# 4.2 t-SNE (on subset for speed)
tsne_idx = np.random.choice(len(X_scaled), 500, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
X_tsne = tsne.fit_transform(X_scaled[tsne_idx])
print(f"  t-SNE: Embedded 500-sample subset into 2D")

# 4.3 K-Means Clustering — find optimal k via elbow + silhouette
inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

best_k = k_range[np.argmax(silhouettes)]
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_best.fit_predict(X_pca)
print(f"\n  K-Means optimal k = {best_k} (silhouette score = {max(silhouettes):.3f})")

# 4.4 Hierarchical Clustering (on subset)
hier_idx = np.random.choice(len(X_pca), 300, replace=False)
hier = AgglomerativeClustering(n_clusters=best_k)
hier_labels = hier.fit_predict(X_pca[hier_idx])
hier_silh = silhouette_score(X_pca[hier_idx], hier_labels)
print(f"  Hierarchical Clustering: k={best_k}, silhouette = {hier_silh:.3f}")

# Analyze cluster composition
cluster_df = pd.DataFrame({
    'cluster': cluster_labels,
    'bandgap': y,
    'family': df_clean['material_family'].values
})
print("\n  Cluster mean bandgaps:")
for c in range(best_k):
    subset = cluster_df[cluster_df['cluster'] == c]
    print(f"    Cluster {c}: mean bandgap = {subset['bandgap'].mean():.2f} eV, n = {len(subset)}")

# ─────────────────────────────────────────────────────────────────────────────
# UNIT 5 — DEEP LEARNING: ADVANCED NEURAL NETWORK WITH LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[UNIT 5] Deep Learning — Neural Network Architecture Study")
print("─" * 70)

# Compare different MLP architectures
architectures = {
    "Shallow (64)":        (64,),
    "Medium (128-64)":     (128, 64),
    "Deep (256-128-64-32)":(256, 128, 64, 32),
    "Wide (512-256)":      (512, 256),
}

nn_results = {}
for arch_name, layers in architectures.items():
    nn = MLPRegressor(hidden_layer_sizes=layers, activation='relu',
                      max_iter=500, random_state=42,
                      learning_rate='adaptive', learning_rate_init=0.001,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    r2 = r2_score(y_test, y_pred_nn)
    mae = mean_absolute_error(y_test, y_pred_nn)
    nn_results[arch_name] = {"r2": r2, "mae": mae, "model": nn}
    print(f"  {arch_name:<28s} | R²={r2:.3f} | MAE={mae:.3f} | Epochs={nn.n_iter_}")

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE: STACKING (as in paper's Table 3)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[ENSEMBLE] Stacking Ensemble (replicating paper Table 3)")
print("─" * 70)

# Level-1 predictions (OOF)
def get_oof_predictions(model, X_tr, y_tr, X_te, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    oof = np.zeros(len(X_tr))
    te_preds = np.zeros(len(X_te))
    for tr_idx, val_idx in kf.split(X_tr):
        model.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[val_idx] = model.predict(X_tr[val_idx])
        te_preds += model.predict(X_te) / cv
    return oof, te_preds

print("  Building OOF predictions for stacking base learners...")
rf_base  = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
svr_base = SVR(kernel='rbf', C=5)
gbm_base = GradientBoostingRegressor(n_estimators=100, random_state=42)

oof_rf,  te_rf  = get_oof_predictions(rf_base,  X_train, y_train, X_test)
oof_svr, te_svr = get_oof_predictions(svr_base, X_train, y_train, X_test)
oof_gbm, te_gbm = get_oof_predictions(gbm_base, X_train, y_train, X_test)

# Level-2 meta-learner
meta_X_train = np.column_stack([oof_rf, oof_svr, oof_gbm])
meta_X_test  = np.column_stack([te_rf, te_svr, te_gbm])
meta_model   = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)
y_pred_stack = meta_model.predict(meta_X_test)

stack_r2   = r2_score(y_test, y_pred_stack)
stack_mae  = mean_absolute_error(y_test, y_pred_stack)
stack_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
print(f"  Stacking Ensemble | MAE={stack_mae:.3f} | RMSE={stack_rmse:.3f} | R²={stack_r2:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS — Comprehensive 3-page figure
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("[PLOTS] Generating comprehensive visualisation report...")
print("─" * 70)

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'text.color':       '#e6edf3',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

ACCENT   = '#58a6ff'
ACCENT2  = '#3fb950'
ACCENT3  = '#f85149'
ACCENT4  = '#d2a8ff'
ACCENT5  = '#ffa657'
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, '#79c0ff', '#56d364']

# ── PAGE 1: Data exploration & preprocessing ─────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.patch.set_facecolor('#0d1117')
fig1.suptitle("Unit 1 & 2 — Data Exploration & Preprocessing",
              fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)

# 1a. Bandgap histogram (replicates paper Fig. 1)
ax = axes[0, 0]
counts, bins, patches = ax.hist(y, bins=40, color=ACCENT, edgecolor='#0d1117', alpha=0.85)
for patch, b in zip(patches, bins):
    patch.set_facecolor(plt.cm.cool(b / 12))
ax.set_xlabel("Bandgap (eV)")
ax.set_ylabel("Count")
ax.set_title("Bandgap Distribution (N=3896)", fontweight='bold')
ax.axvline(y.mean(), color=ACCENT3, linestyle='--', label=f'Mean={y.mean():.2f} eV')
ax.legend(fontsize=8)

# 1b. Material family distribution
ax = axes[0, 1]
family_counts = df['material_family'].value_counts()
bars = ax.barh(family_counts.index, family_counts.values, color=PALETTE)
ax.set_xlabel("Count")
ax.set_title("Material Family Distribution", fontweight='bold')
for bar, val in zip(bars, family_counts.values):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=8, color='#8b949e')

# 1c. Correlation heatmap (top features vs bandgap)
ax = axes[0, 2]
corr_data = df_clean[feature_names + ['bandgap_eV']].corr()['bandgap_eV'].drop('bandgap_eV').sort_values()
colors = [ACCENT3 if v < 0 else ACCENT2 for v in corr_data.values]
ax.barh(corr_data.index, corr_data.values, color=colors, edgecolor='#0d1117')
ax.axvline(0, color='#8b949e', linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Bandgap")
ax.set_title("Feature Correlations", fontweight='bold')
ax.tick_params(labelsize=7)

# 1d. Feature distributions (violin)
ax = axes[1, 0]
top3_feats = feat_importance.head(3).index.tolist()
plot_data = [df_clean[f].values for f in top3_feats]
vp = ax.violinplot(plot_data, showmedians=True)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(PALETTE[i])
    pc.set_alpha(0.7)
vp['cmedians'].set_color('#ffffff')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f.replace('_', '\n') for f in top3_feats], fontsize=7)
ax.set_title("Top-3 Feature Distributions", fontweight='bold')

# 1e. Missing values bar
ax = axes[1, 1]
mv_per_col = (df_dirty[feature_names].isna().sum() / N * 100).sort_values(ascending=False).head(8)
ax.bar(range(len(mv_per_col)), mv_per_col.values, color=ACCENT5)
ax.set_xticks(range(len(mv_per_col)))
ax.set_xticklabels([f.replace('_', '\n') for f in mv_per_col.index], fontsize=6)
ax.set_ylabel("Missing %")
ax.set_title("Missing Values per Feature (2% injection)", fontweight='bold')

# 1f. PCA scree plot
ax = axes[1, 2]
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax.plot(range(1, len(cum_var)+1), cum_var, color=ACCENT, linewidth=2, marker='o', markersize=4)
ax.axhline(95, color=ACCENT3, linestyle='--', label='95% threshold')
ax.axvline(n_95, color=ACCENT5, linestyle='--', label=f'{n_95} components')
ax.fill_between(range(1, len(cum_var)+1), cum_var, alpha=0.15, color=ACCENT)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA Scree Plot", fontweight='bold')
ax.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig('/mnt/user-data/outputs/fig1_data_exploration.png',
             dpi=140, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig1)
print("  ✓ Saved: fig1_data_exploration.png")

# ── PAGE 2: Regression & Classification results ───────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
fig2.patch.set_facecolor('#0d1117')
fig2.suptitle("Unit 3 — Supervised Learning Results (Regression & Classification)",
              fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)

# 2a-2f: Predicted vs Actual scatter for each regression model
reg_models_to_plot = regression_results[:6]
for idx, res in enumerate(reg_models_to_plot):
    ax = axes[idx // 3][idx % 3]
    sc = ax.scatter(y_test, res['y_pred'], alpha=0.35, s=8,
                    c=np.abs(y_test - res['y_pred']), cmap='plasma',
                    vmin=0, vmax=3)
    lim = [0, 12]
    ax.plot(lim, lim, 'w--', linewidth=1.2, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True Bandgap (eV)", fontsize=8)
    ax.set_ylabel("Predicted (eV)", fontsize=8)
    ax.set_title(f"{res['name']}\nR²={res['R2']:.3f}  MAE={res['MAE']:.3f}",
                 fontsize=9, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='|Error| eV', fraction=0.046, pad=0.04)

fig2.tight_layout()
fig2.savefig('/mnt/user-data/outputs/fig2_regression_results.png',
             dpi=140, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig2)
print("  ✓ Saved: fig2_regression_results.png")

# ── PAGE 3: Unsupervised Learning + Model Comparison ─────────────────────────
fig3 = plt.figure(figsize=(20, 12))
fig3.patch.set_facecolor('#0d1117')
fig3.suptitle("Units 4 & 5 — Unsupervised Learning, Deep Learning & Model Comparison",
              fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)
gs = gridspec.GridSpec(2, 4, figure=fig3, hspace=0.45, wspace=0.35)

# 3a. PCA scatter colored by bandgap
ax_pca = fig3.add_subplot(gs[0, 0])
sc = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='plasma',
                    s=4, alpha=0.5, vmin=0, vmax=12)
plt.colorbar(sc, ax=ax_pca, label='Bandgap (eV)', fraction=0.046)
ax_pca.set_title("PCA — Colored by Bandgap", fontweight='bold')
ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")

# 3b. t-SNE scatter colored by material family
ax_tsne = fig3.add_subplot(gs[0, 1])
family_subset = df_clean['material_family'].values[tsne_idx]
uniq_fam = np.unique(family_subset)
for i, fam in enumerate(uniq_fam):
    mask = family_subset == fam
    ax_tsne.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=10,
                    alpha=0.7, label=fam, color=PALETTE[i])
ax_tsne.set_title("t-SNE — Material Families", fontweight='bold')
ax_tsne.legend(fontsize=5, loc='upper right')
ax_tsne.set_xlabel("t-SNE 1"); ax_tsne.set_ylabel("t-SNE 2")

# 3c. K-Means elbow + silhouette
ax_km = fig3.add_subplot(gs[0, 2])
ax_km2 = ax_km.twinx()
k_list = list(k_range)
ax_km.plot(k_list, inertias, color=ACCENT, marker='o', linewidth=2, label='Inertia')
ax_km2.plot(k_list, silhouettes, color=ACCENT2, marker='s', linewidth=2,
            linestyle='--', label='Silhouette')
ax_km.set_xlabel("k (clusters)")
ax_km.set_ylabel("Inertia", color=ACCENT)
ax_km2.set_ylabel("Silhouette Score", color=ACCENT2)
ax_km.set_title("K-Means: Elbow + Silhouette", fontweight='bold')
ax_km.axvline(best_k, color=ACCENT5, linestyle=':', linewidth=1.5)

# 3d. Cluster scatter (PCA space)
ax_cl = fig3.add_subplot(gs[0, 3])
for k in range(best_k):
    mask = cluster_labels == k
    ax_cl.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5,
                  alpha=0.6, label=f"Cluster {k}", color=PALETTE[k])
ax_cl.set_title(f"K-Means Clusters (k={best_k})", fontweight='bold')
ax_cl.legend(fontsize=7)
ax_cl.set_xlabel("PC1"); ax_cl.set_ylabel("PC2")

# 3e. Model comparison bar chart (R²)
ax_bar = fig3.add_subplot(gs[1, 0:2])
model_names = [r['name'] for r in regression_results] + ["Stacking Ensemble"]
r2_vals     = [r['R2']  for r in regression_results] + [stack_r2]
colors_bar  = [ACCENT2 if r2 > 0.85 else ACCENT if r2 > 0.75 else ACCENT5 for r2 in r2_vals]
bars = ax_bar.barh(model_names, r2_vals, color=colors_bar, edgecolor='#21262d')
ax_bar.set_xlim(0.55, 1.0)
ax_bar.axvline(0.9, color=ACCENT3, linestyle='--', alpha=0.6, label='R²=0.9 threshold')
for bar, val in zip(bars, r2_vals):
    ax_bar.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, color='#e6edf3')
ax_bar.set_xlabel("R² Score (Test Set)")
ax_bar.set_title("Model Comparison — R² Scores", fontweight='bold')
ax_bar.legend(fontsize=8)

# 3f. Neural network architecture comparison
ax_nn = fig3.add_subplot(gs[1, 2])
nn_names = list(nn_results.keys())
nn_r2s   = [nn_results[n]['r2'] for n in nn_names]
bars_nn = ax_nn.bar(range(len(nn_names)), nn_r2s, color=ACCENT4, edgecolor='#0d1117')
ax_nn.set_xticks(range(len(nn_names)))
ax_nn.set_xticklabels([n.replace(' ', '\n') for n in nn_names], fontsize=7)
ax_nn.set_ylabel("R² Score")
ax_nn.set_title("Neural Network Architectures", fontweight='bold')
for bar, val in zip(bars_nn, nn_r2s):
    ax_nn.text(bar.get_x() + bar.get_width()/2, val + 0.002,
               f'{val:.3f}', ha='center', fontsize=7, color='#e6edf3')

# 3g. Feature importance (RF)
ax_fi = fig3.add_subplot(gs[1, 3])
top10 = feat_importance.head(10)
ax_fi.barh(top10.index[::-1], top10.values[::-1], color=ACCENT)
ax_fi.set_xlabel("Importance Score")
ax_fi.set_title("RF Feature Importance (Top 10)", fontweight='bold')
ax_fi.tick_params(labelsize=7)

fig3.savefig('/mnt/user-data/outputs/fig3_unsupervised_comparison.png',
             dpi=140, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig3)
print("  ✓ Saved: fig3_unsupervised_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY (replicating paper Table 1 & Table 3)")
print("=" * 70)
print(f"\n  {'Model':<28s} | {'MAE':>6} | {'RMSE':>6} | {'R²':>6}")
print(f"  {'─'*28} | {'─'*6} | {'─'*6} | {'─'*6}")
for res in regression_results:
    print(f"  {res['name']:<28s} | {res['MAE']:>6.3f} | {res['RMSE']:>6.3f} | {res['R2']:>6.3f}")
print(f"  {'Stacking Ensemble':<28s} | {stack_mae:>6.3f} | {stack_rmse:>6.3f} | {stack_r2:>6.3f}")

best_reg = max(regression_results + [{"name":"Stacking","R2":stack_r2,"MAE":stack_mae}],
               key=lambda x: x['R2'])
print(f"\n  ★ Best model: {best_reg['name']} (R²={best_reg['R2']:.4f})")
print("\n" + "=" * 70)
print("  Project complete. All figures saved to output directory.")
print("=" * 70)
