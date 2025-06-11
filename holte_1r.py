import numpy as np
import pandas as pd

def holte_1r_discretize(feature, labels, min_bin_size=6):
    """
    Discretize a continuous feature using Holte's 1R algorithm.
    
    Parameters:
    - feature: 1D array of continuous values
    - labels: 1D array of class labels
    - min_bin_size: Minimum number of instances per bin (default: 6)
    
    Returns:
    - bins: Array of bin indices for each data point
    - bin_edges: List of bin boundaries
    """
    # Combine feature values and labels, sort by feature
    data = np.column_stack((feature, labels))
    sorted_data = data[data[:, 0].argsort()]
    sorted_values, sorted_labels = sorted_data[:, 0], sorted_data[:, 1]
    
    # Initialize variables
    bin_edges = []
    bins = np.zeros(len(feature), dtype=int)
    current_bin = 0
    start_idx = 0
    
    # Iterate through sorted values
    for i in range(1, len(sorted_values)):
        # Create a new bin if class changes or min_bin_size is reached
        if sorted_labels[i] != sorted_labels[i-1] or (i - start_idx) >= min_bin_size:
            # Add boundary (use midpoint between current and previous value)
            if i < len(sorted_values):
                boundary = (sorted_values[i-1] + sorted_values[i]) / 2
                bin_edges.append(boundary)
            # Assign bin indices for the current bin
            bins[start_idx:i] = current_bin
            current_bin += 1
            start_idx = i
    
    # Assign the last bin
    bins[start_idx:] = current_bin
    bin_edges.append(sorted_values[-1] + 1e-10)  # Ensure last value is included
    
    # Merge bins with fewer than min_bin_size instances
    bin_counts = np.bincount(bins)
    small_bins = np.where(bin_counts < min_bin_size)[0]
    
    if len(small_bins) > 0:
        # Simple merging: combine small bins with the next bin
        new_bins = bins.copy()
        new_bin_edges = bin_edges.copy()
        for small_bin in small_bins:
            if small_bin < current_bin:  # Merge with next bin
                new_bins[bins == small_bin] = small_bin + 1
                # Update bin edges (remove the boundary of the small bin)
                if small_bin < len(new_bin_edges) - 1:
                    new_bin_edges[small_bin] = new_bin_edges[small_bin + 1]
        # Update bins and edges
        bins = new_bins
        bin_edges = [e for i, e in enumerate(new_bin_edges) if i not in small_bins or i == len(new_bin_edges)-1]
        # Reassign bin indices to be consecutive
        unique_bins = np.unique(bins)
        bin_mapping = {old: new for new, old in enumerate(unique_bins)}
        bins = np.array([bin_mapping[b] for b in bins])
    
    return bins, bin_edges