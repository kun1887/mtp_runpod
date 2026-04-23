from typing import Any, Optional, Tuple, cast

import torch
from torchtune.modules.transformer import TransformerDecoder

from evaluation.FR_interpolation import interpolate_fisher_rao
from evaluation.custom_generation_utils import find_dist_with_entropy
from mimo_mtp_evaluation.mimo_utils import logit_filtering_mask


def generate_next_token_phi_with_fr_interpolation(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    logit_filter: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    equal_entropy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates next tokens using Fisher-Rao interpolation in the latent Gaussian
    space produced by Shortcut-PHi outputs.

    Notes:
    - `alpha` is used as the interpolation factor `t` for FR interpolation.
        - `temperature` is accepted for API compatibility but intentionally ignored.
    - This function requires model outputs with keys:
      p_mean, p_logvar, q_mean, q_logvar, outputs.
    """
    assert top_p >= 1.0

    return_dict = model(x, input_pos=input_pos, mask=mask)

    required_keys = ["p_mean", "p_logvar", "q_mean", "q_logvar", "outputs"]
    missing = [k for k in required_keys if k not in return_dict]
    if missing:
        raise RuntimeError(
            "FR creativity evaluation requires Shortcut-PHi outputs. "
            f"Missing keys in model forward output: {missing}"
        )

    if not hasattr(model, "shortcut_PHi_layer") or model.shortcut_PHi_layer is None:
        raise RuntimeError("FR interpolation requires a model with shortcut_PHi_layer.")
    if not hasattr(model, "shortcut_PHi_layer_position"):
        raise RuntimeError("FR interpolation requires shortcut_PHi_layer_position on the model.")

    # Reference output distribution used for filtering and entropy target.
    logits_ref = model.output(return_dict["outputs"])[:, -1].float()

    if logit_filter is not None:
        logits_ref = logits_ref - logit_filter.to(logits_ref.device)[None].float() * 1e10

    top_k_eff = top_k if top_k is not None else int(logits_ref.shape[-1])
    logit_mask = logit_filtering_mask(logits_ref.view(-1, logits_ref.shape[-1]), top_p=top_p, top_k=top_k_eff)
    logit_mask = logit_mask.view_as(logits_ref)

    # Interpolate in Gaussian space for all token positions.
    p_dist = {
        "p_mean": return_dict["p_mean"].float(),
        "p_logvar": return_dict["p_logvar"].float(),
    }
    q_dist = {
        "q_mean": return_dict["q_mean"].float(),
        "q_logvar": return_dict["q_logvar"].float(),
    }
    interp = interpolate_fisher_rao(p_dist, q_dist, alpha)

    # Sample latent z from FR-interpolated distribution, then continue with the
    # same reconstruction and downstream decoder path used by Shortcut-PHi.
    eps = torch.randn_like(interp["mean"])
    z_fr = interp["mean"] + interp["sigma"] * eps

    shortcut_layer = cast(Any, model.shortcut_PHi_layer)
    model_dtype = next(model.parameters()).dtype
    h = shortcut_layer.decoder_transform(z_fr.to(dtype=model_dtype))
    h = shortcut_layer.h_reconstruction_norm(h)

    if x.ndim != 2:
        raise RuntimeError(f"Expected x to have shape [batch, seq], got {tuple(x.shape)}")
    padding_mask = x != 0

    start_layer = int(cast(Any, model.shortcut_PHi_layer_position)) + 1
    for layer in model.layers[start_layer:]:
        h = layer(
            h,
            padding_mask=padding_mask,
            mask=mask,
            input_pos=input_pos,
        )

    interp_logits = model.output(h)[:, -1].float()

    filtered_ref_logits = logits_ref[logit_mask].reshape(logits_ref.shape[0], -1)
    filtered_interp_logits = interp_logits[logit_mask].reshape(interp_logits.shape[0], -1)

    ref_distribution = torch.softmax(filtered_ref_logits, dim=-1)
    fr_distribution = torch.softmax(filtered_interp_logits, dim=-1)

    if equal_entropy:
        target_entropy = -torch.sum(ref_distribution * torch.log(ref_distribution + 1e-20), dim=-1)
        optimized_distribution = find_dist_with_entropy(p_target=fr_distribution, H_target=target_entropy)
    else:
        optimized_distribution = fr_distribution

    full_distribution = torch.zeros_like(logits_ref)
    full_distribution[logit_mask] = optimized_distribution.flatten()

    sampled_tokens = torch.zeros(logits_ref.shape[0], dtype=torch.long, device=logits_ref.device)
    for i in range(logits_ref.shape[0]):
        sampled_tokens[i] = torch.distributions.Categorical(full_distribution[i]).sample()

    # Return the same logits tensor contract as existing generation helpers.
    return sampled_tokens.unsqueeze(1), model.output(return_dict["outputs"])
