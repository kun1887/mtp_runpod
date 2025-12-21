import torch


def geodesic_interpolation(p: torch.Tensor, m: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Performs batched geodesic interpolation between two sets of categorical distributions.

    Args:
        p (torch.Tensor): A batch of starting probability distributions.
                          Shape: (batch_size, num_categories).
        m (torch.Tensor): A batch of ending probability distributions.
                          Shape: (batch_size, num_categories).
        alpha (float): The interpolation factor, from 0 (returns p) to 1 (returns m).

    Returns:
        torch.Tensor: The batch of interpolated probability distributions.
                      Shape: (batch_size, num_categories).
    """
    # 1. Map probabilities to the positive orthant of a sphere
    p_sqrt = torch.sqrt(p)
    m_sqrt = torch.sqrt(m)

    # 2. Calculate the angle for each pair in the batch
    # The dot product is the sum of the element-wise product.
    dot_product = torch.sum(p_sqrt * m_sqrt, dim=-1)

    # Clip for numerical stability. Result shape: (batch_size,)
    omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

    # Add a dimension for broadcasting
    omega = omega.unsqueeze(-1)  # Shape: (batch_size, 1)

    # 3. Handle the small-angle case to avoid division by zero
    # Create a boolean mask for pairs of distributions that are very close
    small_angle_mask = torch.abs(omega) < 1e-10

    # For small angles, linear interpolation is a stable and accurate approximation
    linear_interp_sqrt = (1.0 - alpha) * p_sqrt + alpha * m_sqrt

    # For larger angles, use the full Slerp formula
    sin_omega = torch.sin(omega)
    term1 = torch.sin((1.0 - alpha) * omega) / sin_omega
    term2 = torch.sin(alpha * omega) / sin_omega
    slerp_sqrt = term1 * p_sqrt + term2 * m_sqrt

    # 4. Combine the results using the mask
    # torch.where selects elements from either tensor based on the condition
    interpolated_sqrt = torch.where(small_angle_mask, linear_interp_sqrt, slerp_sqrt)

    # 5. Map back to the probability simplex by squaring
    interpolated_p = interpolated_sqrt**2

    # 6. Re-normalize to correct for any minor floating-point inaccuracies
    # keepdim=True is important for correct broadcasting of the sum
    return interpolated_p / torch.sum(interpolated_p, dim=-1, keepdim=True)


def find_dist_with_entropy(
    p_target: torch.Tensor,
    H_target: torch.Tensor,
    tol: float = 1e-5,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Finds distributions q closest to p_target in KL divergence with specific target entropies.
    This is a batched version that processes multiple distributions and entropies in parallel.

    Args:
        p_target (torch.Tensor): A batch of target probability distributions.
                                 Shape: (B, K), where B is batch size and K is num_categories.
        H_target (torch.Tensor): A batch of desired target entropies.
                                 Shape: (B,).
        tol (float): The tolerance for the entropy difference to determine convergence.
        max_iter (int): Maximum number of iterations for the binary search.

    Returns:
        torch.Tensor: The batch of resulting distributions q, each with its target entropy.
                      Shape: (B, K).
    """
    # --- Input Validation ---
    if H_target.ndim != 1 or p_target.shape[0] != H_target.shape[0]:
        raise ValueError("H_target must be a 1D tensor with the same batch size as p_target.")

    # --- Pre-computation and Helpers ---
    # Normalize and add epsilon for stability
    p_target = p_target / p_target.sum(dim=-1, keepdim=True)
    p_target_stable = torch.clamp(p_target, min=1e-20)
    log_p = torch.log(p_target_stable)

    # Batched entropy calculation
    def get_entropy(q: torch.Tensor) -> torch.Tensor:
        # Clamp q as well to avoid log(0) from softmax output
        return -torch.sum(q * torch.log(torch.clamp(q, min=1e-20)), dim=-1)

    # --- Batched Binary Search for Temperature T ---
    # Initialize search bounds for each item in the batch
    T_low = torch.full_like(H_target, 1e-6)
    T_high = torch.full_like(H_target, 1000.0)

    T_mid = torch.zeros_like(H_target)

    for _ in range(max_iter):
        T_mid = (T_low + T_high) / 2.0

        # Unsqueeze T_mid for broadcasting: (B,) -> (B, 1)
        # This allows division of (B, K) log_p by (B, 1) temperature
        q_mid = torch.softmax(log_p / T_mid.unsqueeze(-1), dim=-1)
        H_mid = get_entropy(q_mid)

        # Optional: early exit if all items in the batch have converged
        if torch.all(torch.abs(H_mid - H_target) < tol):
            break

        # Update search range element-wise using a boolean mask
        is_entropy_too_high = H_mid > H_target
        T_low = torch.where(is_entropy_too_high, T_low, T_mid)
        T_high = torch.where(is_entropy_too_high, T_mid, T_high)

    # --- Final Result ---
    # Compute final q using the converged mid-point temperatures
    final_q = torch.softmax(log_p / T_mid.unsqueeze(-1), dim=-1)

    return final_q
