import numpy as np

def holte_1r_discretize(feature, labels, min_size=6):
    """
    Discretize a continuous feature using Holte's 1R algorithm.
    
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
    bin_edges = [sorted_values[0] - 1e-10]  # Include the minimum value
    bins = np.zeros(len(feature), dtype=int)
    current_bin = 0
    start_idx = 0
    current_label = sorted_labels[0]
    count = 1
    
    # Iterate through sorted values
    for i in range(1, len(sorted_values)):
        if sorted_labels[i] == current_label:
            count += 1
        else:
            # Check if the current bin has enough instances
            if count >= min_size:
                # Add boundary (midpoint between current and next value)
                boundary = (sorted_values[i-1] + sorted_values[i]) / 2
                bin_edges.append(boundary)
                bins[sorted_idx[start_idx:i]] = current_bin
                current_bin += 1
                start_idx = i
                current_label = sorted_labels[i]
                count = 1
            else:
                # Continue accumulating instances without creating a new bin
                count += 1
                continue
    
    # Handle the last bin
    bins[sorted_idx[start_idx:]] = current_bin
    bin_edges.append(sorted_values[-1] + 1e-10)
    
    # Merge bins with fewer than min_size instances
    bin_counts = np.bincount(bins)
    if np.any(bin_counts < min_size):
        new_bins = bins.copy()
        new_bin_edges = bin_edges.copy()
        bin_id = 0
        i = 0
        while i < len(bin_counts):
            if bin_counts[i] < min_size and i < len(bin_counts) - 1:
                # Merge with the next bin
                new_bins[bins == i] = bin_id
                new_bins[bins == i + 1] = bin_id
                bin_counts[i + 1] += bin_counts[i]
                bin_counts[i] = 0
                if i < len(new_bin_edges) - 1:
                    new_bin_edges.pop(i + 1)
            else:
                new_bins[bins == i] = bin_id
                bin_id += 1
            i += 1
        bins = np.array([bin_id for bin_id, _ in enumerate(np.unique(new_bins)) for b in new_bins if b == _])
        bin_edges = new_bin_edges
    
    return bins, bin_edges