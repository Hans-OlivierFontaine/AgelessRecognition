import numpy as np


def print_probability_statistics(own_probs, threshold: float = 0.5):
    min_prob = min(own_probs)
    max_prob = max(own_probs)
    mean_prob = np.mean(own_probs)
    std_prob = np.std(own_probs)

    print("Minimum probability:", min_prob)
    print("Maximum probability:", max_prob)
    print("Mean probability:", mean_prob)
    print("Standard deviation of probabilities:", std_prob)

    # Additional relevant metrics
    num_samples = len(own_probs)
    num_high_prob = sum(prob > threshold for prob in own_probs)
    num_low_prob = sum(prob < threshold for prob in own_probs)
    ratio_high_prob = num_high_prob / num_samples

    print(f"Number of samples with probability < {threshold}:", num_low_prob)
    print(f"Number of samples with probability > {threshold}:", num_high_prob)
    print(f"Ratio of samples with probability > {threshold}:", ratio_high_prob)
