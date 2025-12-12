import random

import numpy as np
import torch


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def bootstrapped_mean_and_ci(data, num_samples=1000, percentiles=(2.5, 97.5)):
    if len(data.shape) == 1:
        no_batch = True
        data = data[None]
    elif len(data.shape) == 2:
        no_batch = False
    else:
        raise ValueError("Data should have 1 or 2 dimensions")

    scores = []
    for _ in range(num_samples):
        idxs = np.random.randint(0, data.shape[1], size=(data.shape[1],))
        sampled_data = data[:, idxs]
        mean = np.mean(sampled_data, axis=1)
        scores.append(mean)
    scores = np.vstack(scores)
    mean = np.mean(scores, axis=0)
    ci_lower, ci_higher = np.percentile(scores, percentiles, axis=0)
    if no_batch:
        return mean[0], ci_lower[0], ci_higher[0]
    return mean, ci_lower, ci_higher


def bootstrapped_mean_diff_and_ci(data1, data2, num_samples=1000, percentiles=(2.5, 97.5)):
    """
    Calculates the bootstrapped difference of means and confidence intervals for two datasets.

    This function handles both single (1D) and batched (2D) arrays. For batched
    arrays, it computes the statistics for each corresponding pair of samples in the batch.

    Args:
        data1 (np.ndarray): The first array of data. Can be 1D or 2D (batch, samples).
        data2 (np.ndarray): The second array of data. Can be 1D or 2D (batch, samples).
        num_samples (int): The number of bootstrap samples to generate.
        percentiles (tuple): The percentiles for the lower and upper bounds of the confidence interval.

    Returns:
        tuple: A tuple containing:
            - mean_diff (float or np.ndarray): The bootstrapped mean of the differences.
            - ci_lower (float or np.ndarray): The lower bound of the confidence interval.
            - ci_higher (float or np.ndarray): The upper bound of the confidence interval.
    """
    # --- Input Validation and Handling ---
    if len(data1.shape) == 1:
        no_batch = True
        data1 = data1[None]  # Add a batch dimension
    elif len(data1.shape) == 2:
        no_batch = False
    else:
        raise ValueError("data1 should have 1 or 2 dimensions")

    # Ensure data2 has a compatible shape
    if len(data2.shape) == 1:
        data2 = data2[None]
        if not no_batch:
            raise ValueError("Batch dimensions are inconsistent: data1 is 2D but data2 is 1D.")
    elif len(data2.shape) == 2:
        if no_batch:
            raise ValueError("Batch dimensions are inconsistent: data1 is 1D but data2 is 2D.")
    else:
        raise ValueError("data2 should have 1 or 2 dimensions")

    if data1.shape[0] != data2.shape[0]:
        raise ValueError(f"Batch dimensions must match: {data1.shape[0]} != {data2.shape[0]}")

    # --- Bootstrapping ---
    diff_scores = []
    for _ in range(num_samples):
        # Resample data1 with replacement
        idxs1 = np.random.randint(0, data1.shape[1], size=(data1.shape[1],))
        sampled_data1 = data1[:, idxs1]
        mean1 = np.mean(sampled_data1, axis=1)

        # Resample data2 with replacement
        idxs2 = np.random.randint(0, data2.shape[1], size=(data2.shape[1],))
        sampled_data2 = data2[:, idxs2]
        mean2 = np.mean(sampled_data2, axis=1)

        # Calculate and store the difference of the means for this sample
        diff_scores.append(mean1 - mean2)

    # --- Calculate Final Statistics ---
    diff_scores = np.vstack(diff_scores)

    mean_diff = np.mean(diff_scores, axis=0)
    ci_lower, ci_higher = np.percentile(diff_scores, percentiles, axis=0)

    # If the original input was 1D, return scalars instead of single-element arrays
    if no_batch:
        return mean_diff[0], ci_lower[0], ci_higher[0]

    return mean_diff, ci_lower, ci_higher


# from https://gist.github.com/ShuhuaGao/9f0338209a8f1523a6c19b5123ac4052
def partial_correlation(x, y, z):
    """
    Partial correlation coefficients between two variables x and y with respect to
    the control variable z.
    @param x (m, ) 1d array
    @param y (m, ) 1d array
    @param z (m, ) or (m, k), 1d or 2d array. If a 2d array, then each column corresponds to a variable.
	@return float, the partial correlation coefficient
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert z.ndim == 1 or z.ndim == 2
    if z.ndim == 1:
        z = np.reshape(z, (-1, 1))
    # solve two linear regression problems Zw = x and Zw = y
    Z = np.hstack([z, np.ones((z.shape[0], 1))])  # bias
    wx = np.linalg.lstsq(Z, x, rcond=None)[0]
    rx = x - Z @ wx # residual
    wy = np.linalg.lstsq(Z, y, rcond=None)[0]
    ry = y - Z @ wy
    # compute the Pearson correlation coefficient between the two residuals
    return np.corrcoef(rx, ry)[0, 1]


def bootstrapped_partial_correlation(x, y, z, num_samples=1000, percentiles=(2.5, 97.5)):
    partial_corr_boot = []

    for _ in range(num_samples):
        idx = np.random.choice(len(x), len(x), replace=True)
        partial_corr_boot.append(partial_correlation(x[idx], y[idx], z[idx]))
    partial_corr_boot = np.array(partial_corr_boot)
    ci_low, ci_high = np.percentile(partial_corr_boot, percentiles)
    mean_r = partial_corr_boot.mean()
    return mean_r, ci_low, ci_high