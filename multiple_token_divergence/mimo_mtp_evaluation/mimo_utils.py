import torch
import torch.nn.functional as F
from IPython.display import display, HTML
import numpy as np
import matplotlib
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def logit_filtering_mask(logits, top_p=0.95, top_k=100):
    """
    Filter a distribution of logits using top-k and nucleus (top-p) filtering.
    """

    # 1. Ensure logits is at least 2D (handle single-instance case)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False

    # 2. Sort logits and their original indices in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # print(sorted_indices)

    # 3. Calculate cumulative probabilities of the sorted logits
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 4. Create a mask to remove tokens with cumulative probability above the threshold
    # We also shift the mask to the right to ensure we keep at least the most probable token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_indices_to_remove[..., top_k:] = True  # Ensure we only keep top_k tokens
    # print(f"sorted indices to remove: {sorted_indices_to_remove}")

    # 5. Scatter the boolean mask back to the original positions
    # This creates a mask of the same shape as the input logits
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    # print(f"indices to remove: {indices_to_remove}")

    # 6. Apply the filter by creating a new tensor
    # The `masked_fill` method returns a new tensor where the mask is True.
    # This avoids the in-place modification that might have caused your CUDA error.
    distribution_mask = torch.ones_like(logits, dtype=torch.bool)
    distribution_mask = distribution_mask.masked_fill(indices_to_remove, False)

    # 7. Squeeze the output if the input was 1D
    if was_1d:
        return distribution_mask.squeeze(0)

    return distribution_mask


def solve_slop_optimization(p, m, alpha, tol=1e-9, max_iter=100, verbose=False):
    """
    Solves the optimization problem to find the distribution x that minimizes:
    L(x) = (1 - |alpha|) * H(p,x) + alpha * H(x,m)
    subject to the constraints:
    1. H(x, p) = H(p)  (cross-entropy constraint)
    2. sum(x_i) = 1    (normalization)
    3. x_i >= 0        (non-negativity)

    Args:
        p (np.ndarray): The first categorical probability distribution.
        m (np.ndarray): The second categorical probability distribution.
        alpha (float): Interpolation parameter in the range [-1, 1].
        tol (float): The tolerance for convergence in the bisection search.
        max_iter (int): Maximum number of iterations for the bisection searches.
        verbose (bool): If True, prints intermediate values of the search.

    Returns:
        np.ndarray: The optimal probability distribution x.
    """
    # --- Input Validation ---
    if not np.isclose(np.sum(p), 1.0, atol=0.001) or not np.all(p >= 0):
        raise ValueError("Distribution p must be a valid probability distribution.")
    if not np.isclose(np.sum(m), 1.0, atol=0.001) or not np.all(m >= 0):
        raise ValueError("Distribution m must be a valid probability distribution.")
    if not (-1 <= alpha <= 1):
        raise ValueError("Alpha must be between -1 and 1.")
    if p.shape != m.shape:
        raise ValueError("Distributions p and m must have the same shape.")

    # --- Pre-compute constants to avoid re-calculation ---
    # Add a small epsilon to prevent log(0) for zero-probability events.
    epsilon = 1e-20
    p_log_p = p * np.log(p + epsilon)
    log_p = np.log(p + epsilon)
    log_m = np.log(m + epsilon)

    # The target value for the cross-entropy constraint: H(p)
    h_p_target = -np.sum(p_log_p)

    # Constant C in the parametric form of x_k
    c_k = (1 - abs(alpha)) * p

    def _calculate_x(lmbda, nu):
        """Calculates x based on the dual variables lambda and nu."""
        denominator = lmbda + nu * log_p - alpha * log_m
        # The denominator must be positive for x_k to be positive.
        # If any element is non-positive, it's an invalid parameter set.
        if np.any(denominator <= 0):
            return None
        return c_k / denominator

    # === INNER LOOP: Find lambda for a given nu ===
    def solve_lambda(nu):
        """
        For a fixed nu, find the lambda that satisfies the normalization
        constraint: sum(x_k) = 1.
        This is a 1D root-finding problem solved with bisection.
        """
        # Define the function whose root we want to find: g(lambda) = sum(x_k) - 1
        def g1_lambda(lmbda):
            x = _calculate_x(lmbda, nu)
            if x is None:
                # Return a large value to guide the bisection away from this region
                return np.inf
            return np.sum(x) - 1.0

        # Determine a safe search range for lambda. The denominator must be positive.
        # lambda > max(alpha*log(m) - nu*log(p))
        lambda_min_bound = np.max(alpha * log_m - nu * log_p)

        # Start search slightly above the theoretical minimum.
        # The upper bound can be quite large; the function is monotonic.
        l_min = lambda_min_bound + 1e-9
        l_max = l_min + 1000 # A sufficiently large upper bound

        optimal_lambda, _ = bisection(g1_lambda, l_min, l_max, tol, max_iter)
        return optimal_lambda

    # === OUTER LOOP: Find nu ===
    # Define the function F(nu) = g2(lambda*(nu), nu) whose root we need to find.
    def F_nu(nu):
        """
        The objective function for the outer bisection search on nu.
        """
        # For the current nu, find the corresponding lambda that satisfies sum(x) = 1
        lmbda_star = solve_lambda(nu)

        if lmbda_star is None:
            return np.inf

        # Using these (lambda, nu), calculate the resulting distribution x
        x = _calculate_x(lmbda_star, nu)

        if x is None:
            return np.inf

        # Evaluate the cross-entropy constraint: sum(x_k * log(p_k)) + H(p) = 0
        # We want this value to be zero.
        current_h_xp = -np.sum(x * log_p)
        return current_h_xp - h_p_target

    # --- Execute the two-level search ---
    if verbose:
        print("Starting two-level bisection search...")

    # Search for the optimal nu in a reasonably large range.
    nu_min, nu_max = -100.0, 100.0
    optimal_nu, success = bisection(F_nu, nu_min, nu_max, tol, max_iter, verbose)

    if not success:
        pass
        # print("Warning: Outer bisection for nu may not have converged.")

    # With the optimal nu, find the final optimal lambda
    optimal_lambda = solve_lambda(optimal_nu)

    # Reconstruct the final optimal distribution x
    x_optimal = _calculate_x(optimal_lambda, optimal_nu)

    if x_optimal is None:
        if verbose:
            print("Failed to compute the optimal distribution x.")

    return x_optimal


def bisection(func, a, b, tol=1e-9, max_iter=100, verbose=False):
    """
    Generic 1D root-finding using the bisection method.

    Args:
        func (callable): The function for which to find a root (func(x) = 0).
        a (float): The lower bound of the search interval.
        b (float): The upper bound of the search interval.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root and a boolean indicating success.
    """
    fa = func(a)
    fb = func(b)

    if np.sign(fa) == np.sign(fb):
        # This can happen if the range is too wide or doesn't contain the root.
        if verbose:
            print(f"Bisection Warning: f(a) and f(b) have the same sign. f({a:.4f})={fa:.4f}, f({b:.4f})={fb:.4f}")
        # Try to expand the search range as a fallback
        if abs(fa) < abs(fb): return a, False
        else: return b, False

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = func(c)

        if abs(fc) < tol or (b - a) / 2.0 < tol:
            if verbose:
                print(f"Bisection converged in {i+1} iterations. Root at {c:.6f}")
            return c, True

        if np.sign(fc) == np.sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc

    if verbose:
        print(f"Bisection failed to converge in {max_iter} iterations.")
    return (a + b) / 2.0, False


def fw_with_mtp(model: torch.nn.Module, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """ Do forward pass of main model and MTP layer """
    fw = model.forward(**inputs, output_hidden_states=True)

    embeds = model.model.embed_tokens(inputs["input_ids"][:, 1:])
    full_hidden = fw.hidden_states[-1][:, :-1]
    attn_mask = inputs["attention_mask"][:, 1:] * inputs["attention_mask"][:, :-1]
    position_ids = attn_mask.clone().long().cumsum(dim=-1)
    position_embeddings = model.model.rotary_emb(full_hidden, position_ids)
    mtp_hidden = model.model.mtp_layers[0](
        input_embeds=embeds,
        hidden_states=full_hidden,
        attention_mask=attn_mask.to(full_hidden),
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    if isinstance(mtp_hidden, tuple):
        mtp_hidden = mtp_hidden[0]
    mtp_logits = model.lm_head(mtp_hidden)
    logits = fw.logits.clone()

    del mtp_hidden, position_embeddings, position_ids, fw, attn_mask, embeds
    return logits, mtp_logits


def generate_with_mtp(model,
                      tokenizer,
                      inputs,
                      custom_blending_function=None,
                      max_new_tokens=512,
                      sampling_method='greedy',
                      stop_strings: list = None):
    """
    Generates a sequence of tokens using a custom blending of normal and MTP logits.

    Args:
        model: The Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        inputs (dict): A dictionary containing 'input_ids' and 'attention_mask'.
        custom_blending_function (callable): A function that takes (logits, mtp_logits)
                                              and returns a single combined logits tensor.
        max_new_tokens (int): The maximum number of tokens to generate.
        sampling_method (str): 'greedy' for argmax sampling or 'multinomial' for probabilistic sampling.
    """
    all_agreements = []
    stop_token_sequences = []
    if stop_strings is not None:
        for s in stop_strings:
            stop_token_sequences.append(tokenizer(s).input_ids)
    model.eval()

    with torch.no_grad():
        # Get the initial input_ids and attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Stop generation if we see the EOS token
        eos_token_id = tokenizer.eos_token_id

        generated_ids = input_ids

        for _ in range(max_new_tokens):
            # 1. Get both sets of logits
            current_inputs = {"input_ids": generated_ids, "attention_mask": attention_mask}
            logits, mtp_logits = fw_with_mtp(model, current_inputs)

            # We only care about the logits for the very next token
            next_token_logits = logits[:, -1, :]
            next_token_mtp_logits = mtp_logits[:, -1, :]

            agreement = (torch.argmax(next_token_logits, dim=-1) == torch.argmax(next_token_mtp_logits, dim=-1)).cpu().numpy()
            all_agreements.append(agreement)

            # 2. Apply your custom blending logic
            if custom_blending_function is None:
                final_logits = next_token_logits
            else:
                final_logits = custom_blending_function(next_token_logits.float(), next_token_mtp_logits.float())

            # 3. Sample the next token
            if sampling_method == 'greedy':
                next_token = torch.argmax(final_logits, dim=-1).unsqueeze(-1)
            elif sampling_method == 'multinomial':
                probs = F.softmax(final_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif sampling_method == 'multinomial_direct':
                probs = final_logits
                if probs.min() < -1e-5:
                    is_invalid = True
                else:
                    is_invalid = torch.isnan(probs).any() or torch.isinf(probs).any()

                if is_invalid:
                    print("WARNING: Invalid probability distribution detected. Falling back to argmax.")
                    # You can also log the full tensor here to inspect it later
                    # print(next_token_probs)

                    # Fallback strategy: instead of crashing, do something safe like picking the most likely token
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.clamp(probs, min=0.0)
                    next_token = torch.multinomial(probs, num_samples=1)

                if next_token.numel() != 1:
                    print("WARNING: Invalid probability distribution detected.")
            else:
                raise ValueError(f"Unknown sampling_method: {sampling_method}")

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)

            if len(stop_token_sequences) > 0:
                stop_generating = False
                for stop_seq in stop_token_sequences:
                    l = len(stop_seq)
                    last_segment = generated_ids[0, -l:].tolist()
                    #print(f"Last segment: {last_segment}, Stop seq: {stop_seq}")
                    if last_segment == stop_seq:
                        stop_generating = True
                        #print("Stopping generation due to stop sequence.")
                        break
            if stop_generating:
                break

            if next_token.item() == tokenizer.eos_token_id:
                break

            if next_token.item() == tokenizer.all_special_ids[2]:
                break  # Stop if we generate the special token <|im_end|>

    all_agreements = np.array(all_agreements)
    # print(f"agreement accuracy: {all_agreements.mean()*100:.2f}%")
    return generated_ids


def fw_with_mtp_cached(
        model: torch.nn.Module,
        inputs: dict,
        main_past_key_values: tuple = None,
        past_embeds: torch.Tensor = None,
        past_hidden_for_mtp: torch.Tensor = None,
        past_attention_mask: torch.Tensor = None,
) -> tuple:
    """
    Correctly performs a forward pass using caching for BOTH the main model and the MTP layer.
    """
    # The `cache_position` tells the model where the new tokens are in the sequence.
    cache_position = inputs["cache_position"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")  # Only needed for the first pass

    position_ids = cache_position.unsqueeze(0)
    # print(f"__position: {position_ids[:, -1].item()}")

    # 1. Forward pass through the main model, NOW WITH position_ids.
    tic = time.time()
    fw = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=main_past_key_values,
        use_cache=True,
        output_hidden_states=True,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    #print(f"Main model forward pass time: {time.time() - tic:.4f} seconds")

    tic = time.time()
    if past_embeds is None:
        embeds = model.model.embed_tokens(inputs["input_ids"][:, 1:])
        hidden_for_mtp = fw.hidden_states[-1][:, :-1]
        attn_mask = inputs["attention_mask"][:, 1:] * inputs["attention_mask"][:, :-1]
    else:
        new_embeds = model.model.embed_tokens(inputs["input_ids"][:, -1:])
        embeds = torch.cat([past_embeds, new_embeds], dim=1)
        hidden_for_mtp = past_hidden_for_mtp
        attn_mask = torch.cat([past_attention_mask, torch.ones((1, 1),
                                                               device=past_attention_mask.device)], dim=1)
    position_ids = attn_mask.clone().long().cumsum(dim=-1)
    position_embeddings = model.model.rotary_emb(hidden_for_mtp, position_ids)
    #print(f"Preparing MTP inputs time: {time.time() - tic:.4f} seconds")

    # 3. Forward pass through the fixed MTP layer with all arguments
    tic = time.time()
    mtp_hidden = model.model.mtp_layers[0](
        input_embeds=embeds,
        hidden_states=hidden_for_mtp,
        attention_mask=attn_mask.to(hidden_for_mtp),
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    #print(f"MTP layer forward pass time: {time.time() - tic:.4f} seconds")

    hidden_for_mtp = torch.cat([hidden_for_mtp, fw.hidden_states[-1][:, -1:]], dim=1)

    mtp_logits = model.lm_head(mtp_hidden)
    logits = fw.logits
    new_main_cache = fw.past_key_values

    return logits, mtp_logits, new_main_cache, embeds, hidden_for_mtp, attn_mask


def generate_with_mtp_cached(model, tokenizer, inputs, custom_blending_function, max_new_tokens=512,
                             sampling_method='greedy', stop_strings: list = None):
    """
    Generates a sequence efficiently, managing caches for both main model and MTP.
    """
    model.eval()
    all_agreements = []
    stop_token_sequences = []
    if stop_strings is not None:
        for s in stop_strings:
            stop_token_sequences.append(tokenizer(s).input_ids)

    with torch.no_grad():
        prompt_ids = inputs['input_ids']
        num_prompt_ids = prompt_ids.shape[-1]
        prompt_len = prompt_ids.shape[1]

        # Initialize caches
        main_past_key_values = None
        embeds = None
        hidden_for_mtp = None
        attention_mask_for_mtp = None

        generated_ids = prompt_ids

        attention_mask = inputs['attention_mask']

        for i in range(max_new_tokens):
            current_seq_len = generated_ids.shape[1]

            if i == 0:  # First step (prompt processing)
                input_ids = prompt_ids
                cache_position = torch.arange(prompt_len, device=model.device)
            else:  # Subsequent steps (token-by-token generation)
                input_ids = generated_ids[:, -1:]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=-1)
                cache_position = torch.tensor([current_seq_len - 1], device=model.device)

            current_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position
            }

            logits, mtp_logits, main_past_key_values, embeds, hidden_for_mtp, attention_mask_for_mtp = fw_with_mtp_cached(
                model, current_inputs, main_past_key_values, embeds, hidden_for_mtp, attention_mask_for_mtp
            )

            # --- Sampling
            next_token_logits = logits[:, -1, :]
            next_token_mtp_logits = mtp_logits[:, -1, :]

            agreement = (torch.argmax(next_token_logits, dim=-1) == torch.argmax(next_token_mtp_logits, dim=-1)).cpu().numpy()
            all_agreements.append(agreement)

            final_logits = custom_blending_function(next_token_logits.float(),
                                                    next_token_mtp_logits.float())

            if sampling_method == 'greedy':
                next_token = torch.argmax(final_logits, dim=-1).unsqueeze(-1)
            elif sampling_method == 'multinomial':
                probs = F.softmax(final_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif sampling_method == 'multinomial_direct':
                probs = final_logits
                if probs.min() < -1e-5:
                    is_invalid = True
                else:
                    is_invalid = torch.isnan(probs).any() or torch.isinf(probs).any()

                if is_invalid:
                    print("WARNING: Invalid probability distribution detected. Falling back to argmax.")
                    # You can also log the full tensor here to inspect it later
                    # print(next_token_probs)

                    # Fallback strategy: instead of crashing, do something safe like picking the most likely token
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.clamp(probs, min=0.0)
                    next_token = torch.multinomial(probs, num_samples=1)

                if next_token.numel() != 1:
                    print("WARNING: Invalid probability distribution detected.")
            else:
                raise ValueError(f"Unknown sampling_method: {sampling_method}")
            # --- End Sampling ---

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if len(stop_token_sequences) > 0:
                stop_generating = False
                for stop_seq in stop_token_sequences:
                    l = len(stop_seq)
                    if generated_ids[0, -l:].tolist() == stop_seq:
                        stop_generating = True
                        break
                if stop_generating:
                    break

            if next_token.item() == tokenizer.eos_token_id:
                break

            if next_token.item() == tokenizer.all_special_ids[2]:
                break  # Stop if we generate the special token <|im_end|>

    all_agreements = np.array(all_agreements)
    # print(f"agreement accuracy: {all_agreements.mean()*100:.2f}%")
    return generated_ids


def predict_and_metrics(model: torch.nn.Module, input_list: list) -> tuple[torch.Tensor, torch.Tensor]:
    """ Do forward pass of main model and MTP layer """
    mtp1, mtp2 = fw_with_mtp(model, input_list)
    V = mtp1.shape[-1]
    logsoft1 = F.log_softmax(mtp1.view(-1, V)[1:-1], dim=-1)
    logsoft2 = F.log_softmax(mtp2.view(-1, V)[:-1], dim=-1)

    losses = dict(
        mtp1_loss=F.cross_entropy(mtp1.view(-1, V)[:-1], input_list["input_ids"].view(-1)[1:], reduction="none"),
        mtp2_loss=F.cross_entropy(mtp2.view(-1, V)[:-1], input_list["input_ids"].view(-1)[2:], reduction="none"),
        kl12=F.kl_div(logsoft2, logsoft1, log_target=True, reduction="none").sum(dim=-1)
    )

    metrics = dict(
        mtp1_acc=(mtp1.view(-1, V)[:-1].argmax(dim=-1) == input_list["input_ids"].view(-1)[1:]).float().mean(),
        mtp1_loss=F.cross_entropy(mtp1.view(-1, V)[:-1], input_list["input_ids"].view(-1)[1:]),
        mtp2_acc=(mtp2.view(-1, V)[:-1].argmax(dim=-1) == input_list["input_ids"].view(-1)[2:]).float().mean(),
        mtp2_loss=F.cross_entropy(mtp2.view(-1, V)[:-1], input_list["input_ids"].view(-1)[2:]),
        mtp12_acc=(mtp2.view(-1, V)[:-1].argmax(dim=-1) == mtp1.view(-1, V)[1:-1].argmax(dim=-1)).float().mean(),
        mtp12_kl=losses["kl12"].mean()
    )
    del logsoft1, logsoft2, mtp1, mtp2,
    reduced_metrics = {k: v.mean().item() for k, v in metrics.items()}
    reduced_metrics["token_lenth"] = input_list["input_ids"].shape[-1]
    return metrics, losses, reduced_metrics


def cleanup_tokens(raw_tokens: list[str]) -> list[str]:
    return [tok.replace("Ġ", " ").replace("Ċ", "\n") for tok in raw_tokens]


def index_of_subsequence(query, base):
    l = len(query)
    for i in range(len(base)):
        if base[i:i+l] == query:
            return i
    return None