import numpy as np
from scipy.stats import entropy

def holte_1r_discretize(feature, labels, min_size=6):
    """
    Discretize a continuous feature using Holte's 1R algorithm with improved merging.
    
    Parameters:
    - feature: 1D array of continuous values
    - labels: 1D array of class labels
    - min_size: Minimum number of instances per bin (default: 6)
    
    Returns:
    - bins: Array of bin indices for each data point
    - bin_edges: List of bin boundaries
    """
    # Combine and sort feature values and labels
    data = np.column_stack((feature, labels))
    sorted_idx = np.argsort(data[:, 0])
    sorted_data = data[sorted_idx]
    sorted_values, sorted_labels = sorted_data[:, 0], sorted_data[:, 1]
    
    # Initialize variables
    bin_edges = [sorted_values[0] - 1e-10]
    bins = np.zeros(len(feature), dtype=int)
    current_bin = 0
    start_idx = 0
    current_label = sorted_labels[0]
    count = 1
    
    # Create initial bins based on class changes
    for i in range(1, len(sorted_values)):
        if sorted_labels[i] == current_label:
            count += 1
        else:
            if count >= min_size:
                boundary = (sorted_values[i-1] + sorted_values[i]) / 2
                bin_edges.append(boundary)
                bins[sorted_idx[start_idx:i]] = current_bin
                current_bin += 1
                start_idx = i
                current_label = sorted_labels[i]
                count = 1
            else:
                count += 1
    
    # Handle the last bin
    bins[sorted_idx[start_idx:]] = current_bin
    bin_edges.append(sorted_values[-1] + 1e-10)
    
    # Merge bins to ensure min_size and maximize purity
    def compute_bin_entropy(bins, labels):
        ent = 0
        for bin_id in np.unique(bins):
            bin_labels = labels[bins == bin_id]
            if len(bin_labels) == 0:
                continue
            counts = np.bincount(bin_labels)
            ent += entropy(counts, base=2) * (len(bin_labels) / len(labels))
        return ent
    
    bin_counts = np.bincount(bins)
    while np.any(bin_counts < min_size):
        # Find the smallest bin
        small_bin = np.argmin([count if count < min_size else np.inf for count in bin_counts])
        if small_bin >= len(bin_counts) - 1:
            # Merge with previous bin
            merge_with = small_bin - 1
        else:
            # Choose the merge that minimizes entropy
            entropy_left = entropy_right = np.inf
            if small_bin > 0:
                temp_bins = bins.copy()
                temp_bins[bins == small_bin] = small_bin - 1
                entropy_left = compute_bin_entropy(temp_bins, labels[sorted_idx])
            if small_bin < len(bin_counts) - 1:
                temp_bins = bins.copy()
                temp_bins[bins == small_bin] = small_bin + 1
                entropy_right = compute_bin_entropy(temp_bins, labels[sorted_idx])
            merge_with = small_bin - 1 if entropy_left <= entropy_right else small_bin + 1
        
        # Perform merge
        bins[bins == small_bin] = merge_with
        bin_counts[merge_with] += bin_counts[small_bin]
        bin_counts[small_bin] = 0
        if small_bin < len(bin_edges) - 1:
            bin_edges.pop(min(small_bin, merge_with) + 1)
        # Reassign bin indices
        unique_bins = np.unique(bins)
        bin_mapping = {old: new for new, old in enumerate(unique_bins)}
        bins = np.array([bin_mapping[b] for b in bins])
        bin_counts = np.bincount(bins)
    
    return bins, bin_edges

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