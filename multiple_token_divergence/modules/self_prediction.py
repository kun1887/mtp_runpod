from typing import Callable, Optional, Union, Dict, Any

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.transformer import _get_clones
try:
    from torchtune.modules.loss.loss_types import SFTLoss
    from torchtune.modules.loss import LinearCrossEntropyLoss
except ImportError:
    from torchtune.modules.loss import CEWithChunkedOutputLoss as LinearCrossEntropyLoss
    class SFTLoss:
        """Stub for torchtune versions that don't have SFTLoss (< 0.7)."""
        pass
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel
from torchtune.utils import get_logger

log = get_logger()


def gaussian_kl(pred_mu, pred_logvar, target_mu, target_logvar):
    """
    Calculates the KL divergence between two diagonal multivariate Gaussians,
    D_KL(p || q), where p is the true distribution and q is the approximation.

    The formula is averaged over the batch dimensions and summed over the feature dimensions.

    Args:
        pred_mu (torch.Tensor): Mean of the approximate distribution q.
        pred_logvar (torch.Tensor): Log-variance of the approximate distribution q.
        target_mu (torch.Tensor): Mean of the true distribution p.
        target_logvar (torch.Tensor): Log-variance of the true distribution p.

    Returns:
        torch.Tensor: A scalar tensor with the KL divergence.
    """
    # Calculate the variance ratio and the squared difference of means
    var_ratio = torch.exp(target_logvar - pred_logvar)
    t = (target_mu - pred_mu).pow(2) / torch.exp(pred_logvar)

    # KL divergence formula, summed over features and averaged over the batch
    kl_div = 0.5 * (pred_logvar - target_logvar + var_ratio + t - 1)

    return torch.sum(kl_div, dim=-1)


def gaussian_entropy(log_var):
    """
    Calculates the entropy of a batch of Gaussian distributions with
    diagonal covariance.

    Entropy H(p) for a D-dimensional Gaussian is:
    H(p) = 0.5 * (D * (1 + log(2π)) + sum(log_var))

    Args:
        log_var (torch.Tensor): A tensor of log-variances of shape
                                (batch_size, feature_dim) or (feature_dim,).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) or a scalar
                      containing the entropy of each Gaussian in the batch.
    """
    # Determine the dimensionality D from the last dimension of the input tensor
    if len(log_var.shape) > 1:
        dim = log_var.shape[-1]
    else:
        dim = log_var.shape[0]

    # The formula combines the constant part with the sum of log-variances
    # We sum over the last dimension (the feature dimension)
    return 0.5 * (dim * (1.0 + np.log(2 * np.pi)) + torch.sum(log_var, dim=-1))


class PHiMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) with SwiGLU and residual connections.

    This module is a building block for the PHi layer, suitable for use in the
    prior predictor or decoder. Its architecture is determined
    by the `num_layers` parameter:
    - `num_layers = 1`: The MLP is a simple linear transformation.
    - `num_layers = 2`: The MLP uses a standard SwiGLU (Swish-Gated Linear Unit) block.
    - `num_layers > 2`: The MLP becomes a deep network of SwiGLU blocks with
      residual skip connections between them.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output features.
        num_layers (int): The number of layers, which dictates the architecture.
        activation (nn.Module, optional): The activation function to use within the
            SwiGLU blocks. Defaults to nn.SiLU().
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: nn.Module = nn.SiLU()):
        super().__init__()
        if num_layers == 1:
            hidden_dim = input_dim
        self.gate_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        current_input_dim = input_dim
        for l in range(1, num_layers):
            self.gate_layers.append(nn.Linear(current_input_dim, hidden_dim))
            self.projection_layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        residual = 0
        for gate_layer, projection_layer in zip(self.gate_layers, self.projection_layers):
            gate = self.activation(gate_layer(x))
            proj = projection_layer(x)
            x = residual + gate * proj
            residual = x
        return self.output_layer(x)



class ShortcutPHiLayer(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            posterior_mean_transform: torch.nn.Module,
            posterior_logvar_transform: torch.nn.Module,
            decoder_transform: torch.nn.Module,
            transformer_layer: torch.nn.Module,
            prior_transform: torch.nn.Module,
            detach_inputs: bool = False,
            use_next_embeddings: bool = True,
            full_information_blockage: bool = False,
            critic_network: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.hidden_state_norm = nn.RMSNorm(d_model)
        self.use_next_embeddings = use_next_embeddings
        if use_next_embeddings:
            self.embedding_norm = nn.RMSNorm(d_model)
            self.projection = nn.Linear(2 * d_model, d_model)
        else:
            self.embedding_norm = None
            self.projection = None
        self.h_input_norm = nn.RMSNorm(d_model, elementwise_affine=True)
        self.q_mean_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.h_reconstruction_norm = nn.RMSNorm(d_model, elementwise_affine=True)
        self.posterior_mean_transform = posterior_mean_transform
        self.posterior_logvar_transform = posterior_logvar_transform
        self.decoder_transform = decoder_transform
        self.transformer_layer = transformer_layer
        self.prior_transform = prior_transform
        self.initial_embeddings = torch.nn.Parameter(torch.zeros(1, d_model))
        self.detach_inputs = detach_inputs
        self.full_information_blockage = full_information_blockage
        self.critic_network = critic_network

    def forward(
            self,
            h: torch.Tensor,
            embeddings: torch.Tensor,
            padding_mask: torch.Tensor,
            mask: Optional[_MaskType] = None,
            input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.detach_inputs:
            h = h.detach()
            embeddings = embeddings.detach()
        h = self.h_input_norm(h)

        # --- Information Bottleneck ---
        # 1. Compute posterior distribution q(z|h) and sample latent z
        q_mean = self.posterior_mean_transform(h)
        q_mean = self.q_mean_norm(q_mean)
        # q_mean = F.rms_norm(q_mean, q_mean.shape, eps=1e-6)  # TODO: remove this line if it does not help
        q_logvar = self.posterior_logvar_transform(h)
        q_logvar = torch.clamp(q_logvar, -8, 4)
        if self.full_information_blockage:
            # block all information in the latent space by having zero mean and log variance
            q_mean = q_mean * 0.0
            q_logvar = q_logvar * 0.0

        # 2. Sample from the posterior using the reparameterization trick
        noise = torch.exp(0.5 * q_logvar) * torch.randn_like(q_mean)
        z = q_mean + noise

        # --- Self Prediction ---
        # shift hidden states
        z_shifted = F.pad(h, (0, 0, 1, -1), value=0.)
        # insert initial_embedding wherever the sequence_starts
        new_sequence_starts = input_pos == 0
        z_shifted[new_sequence_starts] = z_shifted[new_sequence_starts] * 0 + self.initial_embeddings
        z_shifted = self.hidden_state_norm(z_shifted)

        if self.use_next_embeddings:
            embeddings = self.embedding_norm(embeddings)
            prediction_input = torch.cat((z_shifted, embeddings), dim=-1)
            prediction_input = self.projection(prediction_input)
        else:
            prediction_input = z_shifted

        shortcut_predictions = self.transformer_layer(
            prediction_input,
            padding_mask=padding_mask,
            mask=mask,
            input_pos=input_pos
        )
        p_mean, p_logvar = self.prior_transform(shortcut_predictions).chunk(2, dim=-1)

        h_reconstructed = self.decoder_transform(z)
        h_reconstructed = self.h_reconstruction_norm(h_reconstructed)

        return_dict =  {
            "h_reconstructed": h_reconstructed,
            "h_original": h,
            "p_mean": p_mean,
            "p_logvar": p_logvar,
            "q_mean": q_mean,
            "q_logvar": q_logvar,
            "z": z
        }

        if self.critic_network is not None:
            critic_vectors = self.critic_network(h)
            return_dict["critic_vectors"] = critic_vectors

        return return_dict


class MTPLayer(torch.nn.Module):
    """
    A Multiple Token Prediction (MTP) layer.
    """

    def __init__(
            self,
            d_model: int,
            transformer_layer: torch.nn.Module,
            additional_layers: Optional[torch.nn.Module] = None,
            detach_inputs: bool = False,
            use_next_embeddings: bool = True,
    ):
        super().__init__()
        self.hidden_state_norm = nn.RMSNorm(d_model)
        self.use_next_embeddings = use_next_embeddings
        if use_next_embeddings:
            self.embedding_norm = nn.RMSNorm(d_model)
            self.projection = nn.Linear(2 * d_model, d_model)
        else:
            self.embedding_norm = None
            self.projection = None
        self.transformer_layer = transformer_layer
        self.additional_layers = additional_layers
        self.initial_embeddings = torch.nn.Parameter(torch.zeros(1, d_model))
        self.detach_inputs = detach_inputs

    def forward(
            self,
            h: torch.Tensor,
            embeddings: torch.Tensor,
            padding_mask: torch.Tensor,
            mask: Optional[_MaskType] = None,
            input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.detach_inputs:
            h = h.detach()
            embeddings = embeddings.detach()

        # shift hidden states to the right
        h_shifted = F.pad(h, (0, 0, 1, -1), value=0.)
        # insert initial_embedding wherever the sequence_starts
        new_sequence_starts = input_pos == 0
        h_shifted[new_sequence_starts] = h_shifted[new_sequence_starts] * 0 + self.initial_embeddings
        h_shifted = self.hidden_state_norm(h_shifted)

        if self.use_next_embeddings:
            embeddings = self.embedding_norm(embeddings)
            prediction_input = torch.cat((h_shifted, embeddings), dim=-1)
            prediction_input = self.projection(prediction_input)
        else:
            prediction_input = h_shifted

        shortcut_predictions = prediction_input
        if self.additional_layers is not None:
            for layer in self.additional_layers:
                shortcut_predictions = layer(shortcut_predictions,
                                             padding_mask=padding_mask,
                                             mask=mask,
                                             input_pos=input_pos)

        shortcut_predictions = self.transformer_layer(
            shortcut_predictions,
            padding_mask=padding_mask,
            mask=mask,
            input_pos=input_pos
        )

        return shortcut_predictions


def initialize_mtp_layer_with_last_layer_weights(checkpoint_weight_dict, full_state_dict):
    last_layer = 0
    for key in checkpoint_weight_dict.keys():
        if 'layers.' in key:
            current_layer = int(key.split('.')[1])
            if current_layer > last_layer:
                last_layer = current_layer

    print(f"Using weights from layer {last_layer} to initialize MTP layer")

    last_layer_parameters = {key: value for key, value in checkpoint_weight_dict.items() if f"layers.{last_layer}" in key}

    all_keys = set(full_state_dict.keys())
    for key, value in last_layer_parameters.items():
        mtp_key = key.replace(f"layers.{last_layer}", "mtp_layer.transformer_layer")
        assert mtp_key in all_keys, f"{mtp_key} not in full_state_dict"
        full_state_dict[mtp_key] = value.clone()

    num_additional_layers = 0
    while True:
        if any(f'additional_layers.{num_additional_layers}' in key for key in full_state_dict.keys()):
            num_additional_layers += 1
        else:
            break
    if num_additional_layers == 0:
        return full_state_dict

    for layer_idx in range(num_additional_layers):
        og_layer = last_layer - num_additional_layers + layer_idx
        layer_parameters = {key: value for key, value in checkpoint_weight_dict.items() if f"layers.{og_layer}" in key}
        for key, value in layer_parameters.items():
            additional_key = key.replace(f"layers.{og_layer}",
                                         f"mtp_layer.additional_layers.{layer_idx}")
            assert additional_key in all_keys, f"{additional_key} not in full_state_dict"
            full_state_dict[additional_key] = value.clone()

    return full_state_dict


class LinearCEPHiLoss(LinearCrossEntropyLoss, nn.Module):
    def __init__(
        self,
        phi_loss_factor: float = 1.0,
        detached_phi_loss_factor: float = 0.0,
        mse_reconstruction_loss_factor: float = 0.0,
        static_kl_loss_factor: float = 0.0,
        self_critic_loss_factor: float = 0.1,
        self_critic_across_sequence_dim: bool = False,
        self_critic_num_neg_examples: int = 7,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        tp_enabled: bool = False,
        mask_ignored_tokens: bool = True,
        divide_elementwise_training_losses_by_d_model: bool = True,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            mask_ignored_tokens (bool): Whether to mask out ignored tokens during loss computation. Default is True.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.mask_ignored_tokens = mask_ignored_tokens
        self.tp_enabled = tp_enabled
        self.phi_loss_factor = phi_loss_factor
        self.detached_phi_loss_factor = detached_phi_loss_factor
        self.mse_reconstruction_loss_factor = mse_reconstruction_loss_factor
        self.static_kl_loss_factor = static_kl_loss_factor
        self.self_critic_loss_factor = self_critic_loss_factor
        self.self_critic_across_sequence_dim = self_critic_across_sequence_dim
        self.self_critic_num_neg_examples = self_critic_num_neg_examples
        self.divide_elementwise_training_losses_by_d_model = divide_elementwise_training_losses_by_d_model

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        if self.tp_enabled and self.mask_ignored_tokens:
            log.warning(
                "Skipping compile loss, as it is not supported with both masking and tensor parallelism enabled."
            )
        else:
            self.compute_cross_entropy = torch.compile(
                self.compute_cross_entropy, *args, **kwargs
            )
        return self

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        loss = F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        return loss

    def mask_inputs(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.where(target != self.ignore_index)[0]

        if isinstance(hidden, DTensor):
            device_mesh = hidden.device_mesh
            hidden = hidden.to_local().index_select(0, indices)
            hidden = DTensor.from_local(
                hidden,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )

            p_mean = p_mean.to_local().index_select(0, indices)
            p_mean = DTensor.from_local(
                p_mean,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )

            p_logvar = p_logvar.to_local().index_select(0, indices)
            p_logvar = DTensor.from_local(
                p_logvar,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )

            q_mean = q_mean.to_local().index_select(0, indices)
            q_mean = DTensor.from_local(
                q_mean,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )

            q_logvar = q_logvar.to_local().index_select(0, indices)
            q_logvar = DTensor.from_local(
                q_logvar,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )
        else:
            hidden = hidden.index_select(0, indices)
            p_mean = p_mean.index_select(0, indices)
            p_logvar = p_logvar.index_select(0, indices)
            q_mean = q_mean.index_select(0, indices)
            q_logvar = q_logvar.index_select(0, indices)

        target = target.index_select(0, indices)
        return hidden, target, p_mean, p_logvar, q_mean, q_logvar


    def compute_self_critic_losses(self, target_mean, target_logvar, z, valid_token_mask):
        if self.self_critic_across_sequence_dim:
            # Get tensor dimensions
            batch_size, seq_len, d_model = z.shape

            # Pre-calculate variance from log-variance
            target_var = target_logvar.exp()

            all_scores = []

            # Iterate over the query sequence dimension (most memory-efficient way)
            for i in range(self.self_critic_num_neg_examples + 1):
                if i != 0:
                    # randomly permute the sequence dimension of z
                    z_permuted = z[:, torch.randperm(seq_len), :]
                else:
                    z_permuted = z
                # shifted_z = F.pad(z.detach(), (0, 0, -i, i), value=0.)
                scores_slice = -F.gaussian_nll_loss(
                    target_mean.float(),
                    #shifted_z.float(),
                    z_permuted.float(),
                    target_var.float(),
                    reduction="none",
                ).sum(dim=-1)
                all_scores.append(scores_slice)
            # Stack all scores and transpose to get the desired shape
            self_critic_scores = torch.stack(all_scores, dim=2)  # (batch_size, seq_len, self.self_critic_num_neg_examples + 1)
            self_critic_targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=z.device)
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]),
                self_critic_targets.flatten(),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * valid_token_mask.flatten()).view_as(valid_token_mask)
            #self_critic_losses[:, -self.self_critic_num_neg_examples:] = 0.0

            return self_critic_losses
        else:
            self_critic_scores = -F.gaussian_nll_loss(
                target_mean.unsqueeze(1),  # (batch, 1, seq_len, d_model)
                z.unsqueeze(0),  # (1, batch, seq_len, d_model)
                target_logvar.exp().unsqueeze(1),  # (batch, 1, seq_len, d_model)
                reduction="none",
            )  # (batch_size_q, batch_size_z, seq_len, d_model)
            self_critic_scores = self_critic_scores.sum(-1).transpose(1, 2)  # (batch_size_q, seq_len, batch_size_z)
            self_critic_targets = torch.arange(self_critic_scores.shape[0])[:, None].repeat(1,
                                                                                            self_critic_scores.shape[1])
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]),
                self_critic_targets.flatten().to(z.device),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * valid_token_mask.flatten()).view_as(valid_token_mask)
            return self_critic_losses

    def compute_neural_critic_loss(self, critic_vectors, z, valid_token_mask):
        if self.self_critic_across_sequence_dim:
            # Get tensor dimensions
            batch_size, seq_len, d_model = z.shape

            all_scores = []

            # Iterate over the query sequence dimension (most memory-efficient way)
            for i in range(self.self_critic_num_neg_examples + 1):
                if i != 0:
                    # randomly permute the sequence dimension of z
                    z_permuted = z[:, torch.randperm(seq_len), :]
                else:
                    z_permuted = z
                # shifted_z = F.pad(z.detach(), (0, 0, -i, i), value=0.)
                scores_slice = torch.einsum('bsd,bsd->bs', critic_vectors, z_permuted)
                all_scores.append(scores_slice)
            # Stack all scores and transpose to get the desired shape
            self_critic_scores = torch.stack(all_scores, dim=2).float()  # (batch_size, seq_len, self.self_critic_num_neg_examples + 1)
            self_critic_targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=z.device)
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]),
                self_critic_targets.flatten(),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * valid_token_mask.flatten()).view_as(valid_token_mask)
            #self_critic_losses[:, -self.self_critic_num_neg_examples:] = 0.0

            return self_critic_losses


    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """
        outputs = input_dict["outputs"]
        prediction_mean = input_dict["p_mean"]
        prediction_logvar = input_dict["p_logvar"]
        target_mean = input_dict["q_mean"]
        target_logvar = input_dict["q_logvar"]
        z = input_dict["z"]
        critic_vectors = input_dict["critic_vectors"] if "critic_vectors" in input_dict else None
        h_original = input_dict["h_original"]
        h_reconstructed = input_dict["h_reconstructed"]
        d_model = z.shape[-1]

        # if no mtp output is provided, use the normal LinearCrossEntropyLoss
        if prediction_mean is None:
            loss = super(LinearCEPHiLoss, self).forward(outputs, targets)
            return loss, {}
        else:
            assert prediction_logvar is not None
            assert target_mean is not None
            assert target_logvar is not None

        valid_token_mask = targets != self.ignore_index
        total_valid_tokens = torch.where(valid_token_mask)[0].numel()
        if total_valid_tokens == 0:
            return torch.tensor(0.0, device=targets.device)

        # this redistribute allows tensor spitting without replication
        if isinstance(outputs, DTensor):
            outputs = outputs.redistribute(
                device_mesh=outputs.device_mesh,
                placements=[Shard(-1)] * outputs.device_mesh.ndim,
            )

        if not self.training:
            return self.compute_validation_loss(outputs,
                                                targets,
                                                prediction_mean=prediction_mean,
                                                prediction_logvar=prediction_logvar,
                                                target_mean=target_mean,
                                                target_logvar=target_logvar,
                                                z=z,
                                                h_original=h_original,
                                                h_reconstructed=h_reconstructed,
                                                critic_vectors=critic_vectors)

        if critic_vectors is not None:
            self_critic_losses = self.compute_neural_critic_loss(critic_vectors=critic_vectors,
                                                                 z=z,
                                                                 valid_token_mask=valid_token_mask)
        else:
            self_critic_losses = self.compute_self_critic_losses(target_mean=target_mean,
                                                                 target_logvar=target_logvar,
                                                                 z=z,
                                                                 valid_token_mask=valid_token_mask)

        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        prediction_mean = prediction_mean.reshape(-1, prediction_mean.shape[-1])
        prediction_logvar = prediction_logvar.reshape(-1, prediction_logvar.shape[-1])
        target_mean = target_mean.reshape(-1, target_mean.shape[-1])
        target_logvar = target_logvar.reshape(-1, target_logvar.shape[-1])
        h_original = h_original.reshape(-1, h_original.shape[-1])
        h_reconstructed = h_reconstructed.reshape(-1, h_reconstructed.shape[-1])

        if self.mask_ignored_tokens:
            outputs, targets, prediction_mean, prediction_logvar, target_mean, target_logvar = self.mask_inputs(outputs,
                                                                                    targets,
                                                                                    p_mean=prediction_mean,
                                                                                    p_logvar=prediction_logvar,
                                                                                    q_mean=target_mean,
                                                                                    q_logvar=target_logvar)

        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        loss_dict = {
            "ce_loss": [],
        }

        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            ce_loss = self.compute_cross_entropy(hidden_chunk, target_chunk)
            ce_loss = ce_loss.sum()

            # without this backprop throws `'Tensor' object has no attribute '_local_tensor'`
            if isinstance(ce_loss, DTensor):
                ce_loss = ce_loss.full_tensor()

            loss_dict["ce_loss"].append(ce_loss.detach())
            total_loss += ce_loss

        for key, value in loss_dict.items():
            loss_dict[key] = torch.stack(value, dim=0).sum() / total_valid_tokens

        # Calculate PHi loss
        detached_phi_losses = gaussian_kl(
            pred_mu=prediction_mean.float(),
            pred_logvar=prediction_logvar.float(),
            target_mu=target_mean.float().detach(),
            target_logvar=target_logvar.float().detach(),
        )
        phi_losses = gaussian_kl(
            pred_mu=prediction_mean.float(),
            pred_logvar=prediction_logvar.float(),
            target_mu=target_mean.float(),
            target_logvar=target_logvar.float(),
        )
        static_kl_losses = gaussian_kl(
            pred_mu=torch.zeros_like(prediction_mean).float(),
            pred_logvar=torch.zeros_like(prediction_logvar).float(),
            target_mu=target_mean.float(),
            target_logvar=target_logvar.float(),
        )

        mse_reconstruction_loss = F.mse_loss(h_reconstructed.float(),
                                             h_original.float(), reduction="none").sum(dim=-1)

        entropy_posterior = gaussian_entropy(target_logvar.float())
        entropy_prior = gaussian_entropy(prediction_logvar.float())

        d_model_factor = 1 / d_model if self.divide_elementwise_training_losses_by_d_model else 1.0

        total_loss += (phi_losses.sum() * self.phi_loss_factor * d_model_factor +
                       static_kl_losses.sum() * self.static_kl_loss_factor * d_model_factor +
                       detached_phi_losses.sum() * self.detached_phi_loss_factor * d_model_factor+
                       self_critic_losses.sum() * self.self_critic_loss_factor +
                       mse_reconstruction_loss.sum() * self.mse_reconstruction_loss_factor * d_model_factor)
        loss_dict["phi_loss"] = phi_losses.sum().detach() / total_valid_tokens
        loss_dict["static_kl_loss"] = static_kl_losses.sum().detach() / total_valid_tokens
        loss_dict["self_critic_loss"] = self_critic_losses.sum().detach() / total_valid_tokens
        loss_dict["entropy_posterior"] = entropy_posterior.sum().detach() / total_valid_tokens
        loss_dict["entropy_prior"] = entropy_prior.sum().detach() / total_valid_tokens
        loss_dict["mse_reconstruction_loss"] = mse_reconstruction_loss.sum().detach() / total_valid_tokens

        return total_loss / total_valid_tokens, loss_dict


    def compute_validation_loss(self,
                                outputs,
                                targets,
                                prediction_mean,
                                prediction_logvar,
                                target_mean,
                                target_logvar,
                                z,
                                h_original,
                                h_reconstructed,
                                critic_vectors=None):
        valid_token_mask = targets != self.ignore_index
        target_shape = targets.shape

        if critic_vectors is not None:
            self_critic_losses = self.compute_neural_critic_loss(critic_vectors=critic_vectors,
                                                                 z=z,
                                                                 valid_token_mask=valid_token_mask)
        else:
            self_critic_losses = self.compute_self_critic_losses(target_mean=target_mean,
                                                                 target_logvar=target_logvar,
                                                                 z=z,
                                                                 valid_token_mask=valid_token_mask)

        # flatten
        valid_token_mask = valid_token_mask.flatten()
        outputs = outputs.view(-1, outputs.shape[-1])
        targets  = targets.flatten()
        prediction_mean = prediction_mean.view(-1, prediction_mean.shape[-1])
        prediction_logvar = prediction_logvar.view(-1, prediction_logvar.shape[-1])
        target_mean = target_mean.view(-1, target_mean.shape[-1])
        target_logvar = target_logvar.view(-1, target_logvar.shape[-1])

        mask_chunks = valid_token_mask.tensor_split(self.num_output_chunks, dim=0)
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        loss_dict = {
            "ce_loss": [],
            "tokenwise_ce_loss": [],
            "tokenwise_phi_loss": [],
        }

        for hidden_chunk, target_chunk, mask_chunk in zip(hidden_chunks,
                                                          target_chunks,
                                                          mask_chunks):
            ce_loss = self.compute_cross_entropy(hidden_chunk, target_chunk)

            # without this backprop throws `'Tensor' object has no attribute '_local_tensor'`
            if isinstance(ce_loss, DTensor):
                ce_loss = ce_loss.full_tensor()

            loss_dict["tokenwise_ce_loss"].append(ce_loss.detach())
            ce_loss = (ce_loss * mask_chunk).sum()
            loss_dict["ce_loss"].append(ce_loss.detach())
            total_loss += ce_loss

        total_valid_tokens = valid_token_mask.sum().item()
        valid_token_mask = valid_token_mask.float()
        loss_dict["ce_loss"] = torch.stack(loss_dict["ce_loss"], dim=0).sum() / total_valid_tokens
        loss_dict["tokenwise_ce_loss"] = torch.cat(loss_dict["tokenwise_ce_loss"], dim=0).view(target_shape)

        # Calculate PHi loss
        phi_losses = gaussian_kl(
            pred_mu=prediction_mean.float(),
            pred_logvar=prediction_logvar.float(),
            target_mu=target_mean.float(),
            target_logvar=target_logvar.float(),
        ) * valid_token_mask

        static_kl_losses = gaussian_kl(
            pred_mu=torch.zeros_like(prediction_mean).float(),
            pred_logvar=torch.zeros_like(prediction_logvar).float(),
            target_mu=target_mean.float(),
            target_logvar=target_logvar.float(),
        ) * valid_token_mask

        mean_mse = F.mse_loss(prediction_mean.float(),
                              target_mean.float(), reduction="none").sum(dim=-1).view(-1) * valid_token_mask

        logvar_mse = F.mse_loss(prediction_logvar.float(),
                                target_logvar.float(), reduction="none").sum(dim=-1).view(-1) * valid_token_mask

        mse_reconstruction_loss = F.mse_loss(h_reconstructed.float(),
                                             h_original.float(), reduction="none").sum(dim=-1).view(-1) * valid_token_mask

        entropies_posterior = gaussian_entropy(target_logvar.float())
        entropies_prior = gaussian_entropy(prediction_logvar.float())

        loss_dict["phi_loss"] = phi_losses.sum().detach() / total_valid_tokens
        loss_dict["static_kl_loss"] = static_kl_losses.sum().detach() / total_valid_tokens
        loss_dict["mean_mse"] = mean_mse.sum().detach() / total_valid_tokens
        loss_dict["logvar_mse"] = logvar_mse.sum().detach() / total_valid_tokens
        loss_dict["mse_reconstruction_loss"] = mse_reconstruction_loss.sum().detach() / total_valid_tokens
        loss_dict["self_critic_loss"] = self_critic_losses.sum().detach() / total_valid_tokens
        loss_dict["entropy_posterior"] = entropies_posterior.sum().detach() / total_valid_tokens
        loss_dict["entropy_prior"] = entropies_prior.sum().detach() / total_valid_tokens
        loss_dict["tokenwise_phi_loss"] = phi_losses.detach().view(target_shape)
        loss_dict["tokenwise_static_kl_loss"] = static_kl_losses.detach().view(target_shape)
        loss_dict["tokenwise_mean_mse"] = mean_mse.detach().view(target_shape)
        loss_dict["tokenwise_logvar_mse"] = logvar_mse.detach().view(target_shape)
        loss_dict["tokenwise_mse_reconstruction_loss"] = mse_reconstruction_loss.view(target_shape)
        loss_dict["tokenwise_self_critic_loss"] = self_critic_losses.detach().view(target_shape)
        loss_dict["tokenwise_entropy_posterior"] = entropies_posterior.detach().view(target_shape)
        loss_dict["tokenwise_entropy_prior"] = entropies_prior.detach().view(target_shape)

        total_loss += phi_losses.sum() * self.phi_loss_factor

        return total_loss / total_valid_tokens, loss_dict


class LinearCEMTPLoss(LinearCrossEntropyLoss, nn.Module):
    def __init__(
        self,
        mtp_ce_loss_factor: float = 0.0,
        mtp_kl_loss_factor: float = 1.0,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        tp_enabled: bool = False,
        mask_ignored_tokens: bool = True,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            mask_ignored_tokens (bool): Whether to mask out ignored tokens during loss computation. Default is True.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.mask_ignored_tokens = mask_ignored_tokens
        self.tp_enabled = tp_enabled
        self.mtp_ce_loss_factor = mtp_ce_loss_factor
        self.mtp_kl_loss_factor = mtp_kl_loss_factor

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        if self.tp_enabled and self.mask_ignored_tokens:
            log.warning(
                "Skipping compile loss, as it is not supported with both masking and tensor parallelism enabled."
            )
        else:
            self.compute_cross_entropy = torch.compile(
                self.compute_cross_entropy, *args, **kwargs
            )
            self.compute_cross_entropy_and_kl = torch.compile(
                self.compute_cross_entropy_and_kl, *args, **kwargs
            )
        return self

    def compute_cross_entropy_and_kl(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        mtp_hidden_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        loss = F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="none",
            #reduction="sum",
            ignore_index=self.ignore_index,
        )

        mtp_logits = self.linear_projection(mtp_hidden_chunk)
        mtp_ce = F.cross_entropy(
            mtp_logits.float(),
            target_chunk,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        mtp_kl = F.kl_div(
            input=F.log_softmax(mtp_logits, dim=-1).float(),
            target=F.softmax(logits, dim=-1).float(),
            reduction="none",
            log_target=False
        ).sum(-1)
        # mtp_kl = (- F.softmax(logits, dim=-1) * F.log_softmax(mtp_logits, dim=-1)).sum(-1)
        agreement_accuracy = (logits.argmax(dim=-1) == mtp_logits.argmax(dim=-1)).float()

        return loss, mtp_ce, mtp_kl, agreement_accuracy

    def mask_inputs(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        mtp_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz*seq_len, emb_dim]``
            target (torch.Tensor): Labels for the model. Shape ``[bsz*seq_len]``

        Returns:
            tuple[torch.Tensor, torch.Tensor]: returns a tuple of
            - The indexed hidden states
            - The indexed targets
        """

        if mtp_hidden is None:
            return super(LinearCEMTPLoss, self).mask_inputs(hidden, target)

        indices = torch.where(target != self.ignore_index)[0]

        if isinstance(hidden, DTensor):
            device_mesh = hidden.device_mesh
            hidden = hidden.to_local().index_select(0, indices)
            hidden = DTensor.from_local(
                hidden,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )

            mtp_hidden = mtp_hidden.to_local().index_select(0, indices)
            mtp_hidden = DTensor.from_local(
                mtp_hidden,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )
        else:
            hidden = hidden.index_select(0, indices)
            mtp_hidden = mtp_hidden.index_select(0, indices)

        target = target.index_select(0, indices)
        return hidden, target, mtp_hidden

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """

        outputs = input_dict["outputs"]
        mtp_outputs = input_dict["mtp_outputs"] if "mtp_outputs" in input_dict else None

        # if no mtp output is provided, use the normal LinearCrossEntropyLoss
        if mtp_outputs is None:
            loss = super(LinearCEMTPLoss, self).forward(outputs, targets)
            return loss, {}

        total_valid_tokens = torch.where(targets != self.ignore_index)[0].numel()
        if total_valid_tokens == 0:
            return torch.tensor(0.0, device=targets.device)

        # this redistribute allows tensor spitting without replication
        if isinstance(outputs, DTensor):
            outputs = outputs.redistribute(
                device_mesh=outputs.device_mesh,
                placements=[Shard(-1)] * outputs.device_mesh.ndim,
            )

        if not self.training:
            return self.compute_validation_loss(outputs, targets, mtp_outputs)

        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        mtp_outputs = mtp_outputs.reshape(-1, mtp_outputs.shape[-1])

        if self.mask_ignored_tokens:
            outputs, targets, mtp_outputs = self.mask_inputs(outputs, targets, mtp_outputs)

        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)
        mtp_hidden_chunks = mtp_outputs.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        loss_dict = {
            "ce_loss": [],
            "mtp_ce_loss": [],
            "mtp_kl_loss": [],
            "agreement_accuracy": [],
        }

        for hidden_chunk, mtp_hidden_chunk, target_chunk in zip(hidden_chunks, mtp_hidden_chunks, target_chunks):
            ce_loss, mtp_ce, mtp_kl, agreement_acc = self.compute_cross_entropy_and_kl(hidden_chunk,
                                                                                       target_chunk,
                                                                                       mtp_hidden_chunk)
            ce_loss = ce_loss.sum()
            mtp_ce = mtp_ce.sum()
            mtp_kl = mtp_kl.sum()
            agreement_acc = agreement_acc.sum()

            # without this backprop throws `'Tensor' object has no attribute '_local_tensor'`
            if isinstance(ce_loss, DTensor):
                ce_loss = ce_loss.full_tensor()
                mtp_ce = mtp_ce.full_tensor()
                mtp_kl = mtp_kl.full_tensor()
                agreement_acc = agreement_acc.full_tensor()

            loss_dict["ce_loss"].append(ce_loss.detach())
            loss_dict["mtp_ce_loss"].append(mtp_ce.detach())
            loss_dict["mtp_kl_loss"].append(mtp_kl.detach())
            loss_dict["agreement_accuracy"].append(agreement_acc.detach())
            total_loss += (ce_loss + mtp_ce * self.mtp_ce_loss_factor + mtp_kl * self.mtp_kl_loss_factor)

        for key, value in loss_dict.items():
            loss_dict[key] = torch.stack(value, dim=0).sum() / total_valid_tokens

        return total_loss / total_valid_tokens, loss_dict


    def compute_validation_loss(self, outputs, targets, mtp_outputs):
        valid_token_mask = targets != self.ignore_index
        target_shape = targets.shape

        # flatten
        valid_token_mask = valid_token_mask.flatten()
        outputs = outputs.view(-1, outputs.shape[-1])
        targets  = targets.flatten()
        mtp_outputs = mtp_outputs.view(-1, mtp_outputs.shape[-1])

        mask_chunks = valid_token_mask.tensor_split(self.num_output_chunks, dim=0)
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)
        mtp_hidden_chunks = mtp_outputs.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        loss_dict = {
            "ce_loss": [],
            "mtp_ce_loss": [],
            "mtp_kl_loss": [],
            "agreement_accuracy": [],
            "tokenwise_ce_loss": [],
            "tokenwise_mtp_ce_loss": [],
            "tokenwise_mtp_kl_loss": [],
            "tokenwise_agreement_accuracy": []
        }

        for hidden_chunk, mtp_hidden_chunk, target_chunk, mask_chunk in zip(hidden_chunks,
                                                                            mtp_hidden_chunks,
                                                                            target_chunks,
                                                                            mask_chunks):
            ce_loss, mtp_ce, mtp_kl, agreement_acc = self.compute_cross_entropy_and_kl(hidden_chunk,
                                                                                       target_chunk,
                                                                                       mtp_hidden_chunk)

            # without this backprop throws `'Tensor' object has no attribute '_local_tensor'`
            if isinstance(ce_loss, DTensor):
                ce_loss = ce_loss.full_tensor()
                mtp_ce = mtp_ce.full_tensor()
                mtp_kl = mtp_kl.full_tensor()
                agreement_acc = agreement_acc.full_tensor()

            loss_dict["tokenwise_ce_loss"].append(ce_loss.detach())
            loss_dict["tokenwise_mtp_ce_loss"].append(mtp_ce.detach())
            loss_dict["tokenwise_mtp_kl_loss"].append(mtp_kl.detach())
            loss_dict["tokenwise_agreement_accuracy"].append(agreement_acc.detach())

            ce_loss = (ce_loss * mask_chunk).sum()
            mtp_ce = (mtp_ce * mask_chunk).sum()
            mtp_kl = (mtp_kl * mask_chunk).sum()
            agreement_acc = (agreement_acc * mask_chunk).sum()

            loss_dict["ce_loss"].append(ce_loss.detach())
            loss_dict["mtp_ce_loss"].append(mtp_ce.detach())
            loss_dict["mtp_kl_loss"].append(mtp_kl.detach())
            loss_dict["agreement_accuracy"].append(agreement_acc.detach())
            total_loss += ce_loss + mtp_ce * self.mtp_ce_loss_factor + mtp_kl * self.mtp_kl_loss_factor

        total_valid_tokens = valid_token_mask.sum().item()
        for key, value in loss_dict.items():
            if "tokenwise" in key:
                loss_dict[key] = torch.cat(value, dim=0).view(target_shape)
            else:
                loss_dict[key] = torch.stack(value, dim=0).sum() / total_valid_tokens

        return total_loss / total_valid_tokens, loss_dict









