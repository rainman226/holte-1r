from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from holte_1r import holte_1r_discretize  # Assuming holte_1r_discretize is defined in holte_1r.py
import pandas as pd
import numpy as np

# Load datasets
datasets = {
    'Iris': load_iris(),
    'Breast Cancer': load_breast_cancer(),
    'Wine': load_wine()
}

# Initialize results storage
results = {'Dataset': [], 'Method': [], 'Accuracy': []}

# Evaluate each dataset
for name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    print(f"\nEvaluating {name}...")
    
    # Raw features (no discretization)
    clf = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    results['Dataset'].extend([name] * 10)
    results['Method'].extend(['Raw'] * 10)
    results['Accuracy'].extend(scores)
    
    # Holte's 1R discretization
    X_1r = X.copy()
    for i in range(X.shape[1]):
        bins, _ = holte_1r_discretize(X[:, i], y, min_size=6)
        X_1r[:, i] = bins
    scores_1r = cross_val_score(clf, X_1r, y, cv=10, scoring='accuracy')
    results['Dataset'].extend([name] * 10)
    results['Method'].extend(['Holte_1R'] * 10)
    results['Accuracy'].extend(scores_1r)
    
    # Equal-width discretization (benchmark)
    X_ew = X.copy()
    for i in range(X.shape[1]):
        X_ew[:, i] = pd.cut(X[:, i], bins=5, labels=False, include_lowest=True)
    scores_ew = cross_val_score(clf, X_ew, y, cv=10, scoring='accuracy')
    results['Dataset'].extend([name] * 10)
    results['Method'].extend(['Equal-Width'] * 10)
    results['Accuracy'].extend(scores_ew)
    
    # Equal-frequency discretization (benchmark)
    X_ef = X.copy()
    for i in range(X.shape[1]):
        X_ef[:, i] = pd.qcut(X[:, i], q=5, labels=False, duplicates='drop')
    scores_ef = cross_val_score(clf, X_ef, y, cv=10, scoring='accuracy')
    results['Dataset'].extend([name] * 10)
    results['Method'].extend(['Equal-Frequency'] * 10)
    results['Accuracy'].extend(scores_ef)

# Summarize results
results_df = pd.DataFrame(results)
print("\nAccuracy Results (10-fold CV):")
print(results_df.groupby(['Dataset', 'Method'])['Accuracy'].mean().unstack())

# Optional: Compute bin purity (entropy) for Holte's 1R
def compute_bin_purity(bins, labels):
    entropy = 0
    for bin_id in np.unique(bins):
        bin_labels = labels[bins == bin_id]
        if len(bin_labels) == 0:
            continue
        counts = np.bincount(bin_labels)
        probs = counts / len(bin_labels)
        probs = probs[probs > 0]
        entropy += -np.sum(probs * np.log2(probs)) * (len(bin_labels) / len(labels))
    return entropy

# Example: Check bin purity for Iris dataset
X_iris, y_iris = datasets['Iris'].data, datasets['Iris'].target
for i in range(X_iris.shape[1]):
    bins, edges = holte_1r_discretize(X_iris[:, i], y_iris, min_size=6)
    purity = compute_bin_purity(bins, y_iris)
    print(f"Iris Feature {i+1} Entropy: {purity:.3f}, Number of Bins: {len(np.unique(bins))}")

