# Implementation of Holte's 1R Algorithm for Data Discretization

This project implements Holte's 1R algorithm for discretizing continuous attributes, developed as part of my Data Mining course. The goal was to transform continuous data into discrete bins for classification tasks, testing the algorithm on some classic datasets.

---

## How This Works

The implementation is a supervised discretization algorithm that creates bins with high class purity, using class labels to guide bin boundaries. It’s written in Python, taking advantage of NumPy, Pandas, Scikit-learn, and SciPy.

### Details of Implementation
- **Function**: `holte_1r_discretize(feature, labels, min_size=6)`
  - **Input**:
    - `feature`: 1D NumPy array of continuous values.
    - `labels`: 1D NumPy array of class labels.
    - `min_size`: Minimum instances per bin (default: 6, or 3 for smaller datasets like Iris).
  - **Output**:
    - `bins`: Array of bin indices for each data point.
    - `bin_edges`: List of bin boundaries.
- **Steps**:
  1. Sort feature values with class labels.
  2. Create bins when class changes, ensuring at least `min_size` instances.
  3. Set bin boundaries at midpoints between consecutive values.
  4. Merge bins with fewer than `min_size` instances, choosing the merge (left or right) that minimizes entropy (using `scipy.stats.entropy`).
- **Key Feature**: Entropy-based merging optimizes bin purity, improving over V1 (excessive bins) and V2 (simple merging).

The main script is `holte_1r.py`, with test code in `main.py`.

---

## Tests and Results

The algorithm was tested on multiple datasets from Scikit-learn using a decision tree classifier with 10-fold cross-validation. Results compare Holte’s 1R against raw data, equal-width, and equal-frequency binning. Additional datasets (e.g., Glass, Diabetes) are planned for future tests™.

### Test Results
| Dataset       | Raw       | Holte_1R  | Equal-Width | Equal-Frequency |
|---------------|-----------|-----------|-------------|-----------------|
| Breast Cancer | 0.9280    | 0.9385 | 0.9279      | **0.9507**      |
| Iris          | **0.9533** | 0.9400    | 0.9200      | 0.9267          |
| Wine          | 0.8650    | **0.9333** | 0.9049      | 0.8938          |
| Digits        | 0.8208  | 0.8152   | 0.8046    | **0.8335**         |



### Iris Bin Statistics
| Feature       | Entropy | Number of Bins |
|---------------|---------|----------------|
| Feature 1     | 0.693   | 19             |
| Feature 2     | 0.925   | 22             |
| Feature 3     | 0.134   | 7              |
| Feature 4     | 0.123   | 7              |

---

## Bibliography
- [1] J. Dougherty, R. Kohavi, and M. Sahami, “Supervised and unsupervised discretization of continuous features,” in *Proc. 12th Int. Conf. Mach. Learn. (ICML)*, Tahoe City, CA, USA, 1995, pp. 194–202.
- [2] H. Liu, F. Hussain, C. L. Tan, and M. Dash, “Discretization: An enabling technique,” *Data Mining Knowl. Disc.*, vol. 6, no. 4, pp. 393–423, Oct. 2002, doi: 10.1023/A:1016304305535.