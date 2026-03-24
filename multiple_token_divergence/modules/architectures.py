import logging
from typing import Callable, Optional, Union
import math

import torch.nn
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.transformer import _get_clones

logger = logging.getLogger(__name__)


def swiglu_mlp(dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    hidden_dim = dim * 8 // 3
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


class TransformerLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (nn.Module): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: nn.Module,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm
        self.mlp_norm = mlp_norm
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        encoder_max_seq_len = 0 * encoder_max_seq_len
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.attn.kv_cache is not None

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """See torchtune.modules.transformer.TransformerSelfAttentionLayer.forward"""
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        if not isinstance(self.attn, torch.nn.Identity):
            attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

            # Residual connection; shape: [batch_size, seq_length, embed_dim]
            h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp_norm(h)
        if not isinstance(self.mlp, torch.nn.Identity):
            mlp_out = self.mlp_scale(self.mlp(mlp_out))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + mlp_out
        return out


class TransformerDecoder(nn.Module):
    """Transformer decoder module, based on the Llama 3.2 architecture."""
    def __init__(
        self,
        *,
        tok_embeddings: torch.nn.Embedding,
        layers: Union[nn.Module, list[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[list[int]] = None,
        num_output_chunks: int = 0,
        tied_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.tied_embeddings = tied_embeddings
        self.num_output_chunks = num_output_chunks
        self.skip_output_layer = False

        self._embedding_are_tied = False
        self.device_mesh = None

        # attributes for KV caches during inference
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len = None

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """torchtune.modules.transformer.TransformerDecoder.setup_caches"""
        if decoder_max_seq_len is not None:
            self.decoder_max_cache_seq_len = decoder_max_seq_len
        else:
            self.decoder_max_cache_seq_len = self.max_seq_len

        for layer in self.layers:
            layer.setup_cache(
                batch_size,
                dtype,
                encoder_max_seq_len=self.encoder_max_cache_seq_len,
                decoder_max_seq_len=self.decoder_max_cache_seq_len,
            )

    def caches_are_setup(self) -> bool:
        return self.encoder_max_cache_seq_len is not None

    def encoder_caches_are_enabled(self) -> bool:
        """Checks if there are any :class:`~torchtune.modules.TransformerCrossAttentionLayer`,
        or :class:`~torchtune.modules.fusion.FusionLayer` layers which have cache enabled.
        """
        return self.encoder_max_cache_seq_len is not None

    def decoder_caches_are_enabled(self) -> bool:
        """Check if there are any :class:`~torchtune.modules.TransformerCrossAttentionLayer`
        layers which have cache enabled."""
        return self.decoder_max_cache_seq_len is not None

    def caches_are_enabled(self) -> bool:
        return self.decoder_caches_are_enabled()

    def reset_caches(self):
        """Reset the key value caches."""
        if not (self.encoder_caches_are_enabled() or self.decoder_caches_are_enabled()):
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            layer.reset_cache()

    def _validate_inputs(
        self,
        seq_len: int,
        mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        """torchtune.modules.transformer.TransformerDecoder._validate_inputs"""

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if self.decoder_caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

        if self.encoder_caches_are_enabled():
            if encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode, causal masks must be provided!"
                    " Use the `encoder_mask` arg to provide a causal mask."
                )

        if (
            self.encoder_caches_are_enabled() or self.decoder_caches_are_enabled()
        ) and input_pos is None:
            raise ValueError(
                "KV-caches are setup for inference mode, input positions must be provided!"
            )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """See torchtune.modules.transformer.TransformerDecoder.forward

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        self._validate_inputs(
            seq_len, mask=mask, encoder_mask=encoder_mask, input_pos=input_pos
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, seq_len, out_dim]
        output = self.unembed(h)

        # Output list if hidden states are requested, otherwise just the output
        output = output if not hidden else [*hidden, output]
        return output

    def unembed(self, h):
        # shape: [b, s, d]
        h = self.norm(h)
        if self.skip_output_layer:
            output = h
        elif self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        return output


class TransformerDecoderMTP(TransformerDecoder):
    def __init__(
        self,
        *args,
        mtp_layer: Optional[nn.Module] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mtp_layer = mtp_layer
        self.pad_token_id = pad_token_id

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        self._validate_inputs(
            seq_len, mask=mask, encoder_mask=encoder_mask, input_pos=input_pos
        )
        padding_mask = tokens == self.pad_token_id

        #print("tokens type:", type(tokens))
        #print("embedding weights type:", type(self.tok_embeddings.weight))
        # shape: [b, s, d]
        embeddings = self.tok_embeddings(tokens)
        h = embeddings

        #print("embedding type:", type(h))

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)

            h = layer(
                h,
                padding_mask=padding_mask,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        output = self.unembed(h)
        if self.mtp_layer is not None:
            shortcut_prediction = self.mtp_layer(
                h=h,
                embeddings=embeddings,
                padding_mask=padding_mask,
                mask=mask,
                input_pos=input_pos,
            )
            shortcut_output = self.unembed(shortcut_prediction)
        else:
            shortcut_output = None

        output_dict = {
            "outputs": output,
            "mtp_outputs": shortcut_output,
        }

        # Output list if hidden states are requested, otherwise just the output
        return output_dict if not hidden else [*hidden, output_dict]

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """Extends parent method to also set up the PHi layer's cache."""
        super().setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )
        if self.mtp_layer is not None:
            self.mtp_layer.setup_cache(
                batch_size, dtype, max_seq_len=decoder_max_seq_len
            )

    def reset_caches(self):
        """Extends parent method to also reset the PHi layer's cache."""
        super().reset_caches()
        if self.mtp_layer is not None:
            self.mtp_layer.reset_cache()



class TransformerDecoderShortcutPHi(TransformerDecoder):
    def __init__(
        self,
        *args,
        shortcut_PHi_layer: Optional[nn.Module] = None,
        shortcut_PHi_layer_position: Optional[int] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shortcut_PHi_layer = shortcut_PHi_layer
        self.pad_token_id = pad_token_id
        if shortcut_PHi_layer_position is None:
            shortcut_PHi_layer_position = len(self.layers) - 1
        self.shortcut_PHi_layer_position = shortcut_PHi_layer_position

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        self._validate_inputs(
            seq_len, mask=mask, encoder_mask=encoder_mask, input_pos=input_pos
        )
        padding_mask = tokens == self.pad_token_id

        # shape: [b, s, d]
        embeddings = self.tok_embeddings(tokens)
        h = embeddings

        shortcut_PHi_dict = None
        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)

            h = layer(
                h,
                padding_mask=padding_mask,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

            if i == self.shortcut_PHi_layer_position and self.shortcut_PHi_layer is not None:
                shortcut_PHi_dict = self.shortcut_PHi_layer(
                    h=h,
                    embeddings=embeddings,
                    padding_mask=padding_mask,
                    mask=mask,
                    input_pos=input_pos,
                )
                h = shortcut_PHi_dict["h_reconstructed"]

        output = self.unembed(h)
        output_dict = shortcut_PHi_dict
        output_dict["outputs"] = output

        # Output list if hidden states are requested, otherwise just the output
        return output_dict if not hidden else [*hidden, output_dict]

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """Extends parent method to also set up the PHi layer's cache."""
        super().setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )
        if self.shortcut_PHi_layer is not None:
            self.shortcut_PHi_layer.setup_cache(
                batch_size, dtype, max_seq_len=decoder_max_seq_len
            )

    def reset_caches(self):
        """Extends parent method to also reset the PHi layer's cache."""
        super().reset_caches()
        if self.shortcut_PHi_layer is not None:
            self.shortcut_PHi_layer.reset_cache()


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with a linear warmup followed by a constant learning rate.

    This scheduler increases the learning rate linearly from 0 to the optimizer's
    initial LR over the course of `num_warmup_steps`. After the warmup period,
    the learning rate is held constant at the initial LR for the remainder of training.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the linear warmup phase.
        num_training_steps (int): The total number of training steps. Note: this
            argument is not used in this specific scheduler but is often included
            for API consistency with other schedulers.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. Defaults to -1.

    Returns:
        LambdaLR: A PyTorch learning rate scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with a linear warmup followed by a cosine decay.

    This scheduler increases the learning rate linearly from 0 to the optimizer's
    initial LR over the course of `num_warmup_steps`. After the warmup period,
    the learning rate decreases following a cosine curve down to 0 at the end
    of training.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the linear warmup phase.
        num_training_steps (int): The total number of training steps.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. Defaults to -1.

    Returns:
        LambdaLR: A PyTorch learning rate scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

