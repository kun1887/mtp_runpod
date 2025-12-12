from typing import Optional

import torch
from torch import nn
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules.tied_linear import TiedLinear

from torchtune.modules import (
    RMSNorm,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
    FeedForward,
    FrozenNF4Linear,
    RotaryPositionalEmbeddings
)

from modules.architectures import (
    TransformerDecoderShortcutPHi,
    TransformerDecoderMTP,
)
from modules.self_prediction import (
    PHiMLP,
    MTPLayer,
    ShortcutPHiLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook


def llama3_mlp(dim: int,
               hidden_dim: int,
               output_dim: Optional[int] = None,
               quantize_base: bool = False) -> FeedForward:
    """
    Factory function to create a Llama 3-style MLP (SwiGLU).

    This function constructs the feed-forward network used in Llama 3, which is
    a Swish-Gated Linear Unit (SwiGLU). It initializes the gate, down, and up
    projection layers and returns them encapsulated in a `FeedForward` module.

    Args:
        dim (int): The input and default output dimension of the MLP.
        hidden_dim (int): The intermediate hidden dimension.
        output_dim (Optional[int], optional): The final output dimension. If None,
            defaults to `dim`. Defaults to None.
        quantize_base (bool, optional): If True, uses quantized linear layers
            (`FrozenNF4Linear`) instead of standard `nn.Linear`. Defaults to False.

    Returns:
        FeedForward: A `FeedForward` module configured with the Llama 3 MLP layers.
    """
    if output_dim is None:
        output_dim = dim
    gate_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, output_dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, output_dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


def self_prediction_mlp(dim: int,
                        hidden_dim: int,
                        output_dim: Optional[int] = None,
                        num_layers: int = 2) -> PHiMLP:
    """
    Factory function to create a `PHiMLP` for the self-prediction module.

    This function is a simple wrapper that constructs a `PHiMLP` instance. This
    type of MLP is used as a building block within the PHi-Layer, for example
    as the prior predictor or the posterior's encoder network.

    Args:
        dim (int): The input and default output dimension.
        hidden_dim (int): The intermediate hidden dimension.
        output_dim (Optional[int], optional): The final output dimension. If None,
            defaults to `dim`. Defaults to None.
        num_layers (int, optional): The number of layers for the `PHiMLP`, which
            determines its architecture (e.g., linear, SwiGLU, or deep residual).
            Defaults to 2.

    Returns:
        PHiMLP: An instance of the configured `PHiMLP`.
    """
    if output_dim is None:
        output_dim = dim
    return PHiMLP(input_dim=dim,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  num_layers=num_layers)


def llama3_shortcut_phi(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 8,
    tied_embeddings: bool = True,
    use_shortcut_phi: bool = True,
    shortcut_PHi_layer_position: Optional[int] = None,
    detach_inputs: bool = False,
    use_next_embeddings: bool = True,
    learnable_critic: bool = False
) -> TransformerDecoderShortcutPHi:
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)

    # Build the stack of Transformer layers
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    nn.init.uniform_(tok_embeddings.weight, a=-1e-4, b=1e-4)
    if tied_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    # Conditionally build the PHi self-prediction layer
    self_prediction_layer = None
    if use_shortcut_phi:
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        if learnable_critic:
            critic_network = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            critic_network = None

        decoder_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        decoder_transform_weight = torch.empty(embed_dim, embed_dim, dtype=torch.float32)
        nn.init.orthogonal_(decoder_transform_weight)
        decoder_transform_weight = decoder_transform_weight.to(decoder_transform.weight.device,
                                                               dtype=decoder_transform.weight.dtype)
        decoder_transform.weight.data = decoder_transform_weight

        posterior_mean_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        posterior_mean_transform.weight.data = decoder_transform_weight.clone().t().contiguous()

        posterior_logvar_transform = nn.Linear(embed_dim, embed_dim, bias=True)
        posterior_logvar_transform.bias.data.fill_(-7.)

        shortcut_PHi_layer = ShortcutPHiLayer(d_model=embed_dim,
                                              posterior_mean_transform=posterior_mean_transform,
                                              posterior_logvar_transform=posterior_logvar_transform,
                                              decoder_transform=decoder_transform,
                                              transformer_layer=layer,
                                              prior_transform=nn.Linear(embed_dim, 2 * embed_dim, bias=True),
                                              detach_inputs=detach_inputs,
                                              use_next_embeddings=use_next_embeddings,
                                              critic_network=critic_network,)
    else:
        shortcut_PHi_layer = None

    # Assemble the final model
    return TransformerDecoderShortcutPHi(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        tied_embeddings=tied_embeddings,
        output_hidden_states=None,
        shortcut_PHi_layer=shortcut_PHi_layer,
        shortcut_PHi_layer_position=shortcut_PHi_layer_position,
    )


def llama3_mtp(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 8,
    tied_embeddings: bool = True,
    use_mtp: bool = True,
    detach_inputs: bool = False,
    use_next_embeddings: bool = True,
    mtp_layer_hidden_size_factor: float = 1.,
    num_mtp_layers: int = 1,
) -> TransformerDecoderMTP:
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    #rope = RotaryPositionalEmbeddings(
    #    dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    #)
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)

    # Build the stack of Transformer layers
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    nn.init.uniform_(tok_embeddings.weight, a=-1e-4, b=1e-4)
    if tied_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    # Conditionally build the PHi self-prediction layer
    self_prediction_layer = None
    if use_mtp:
        mtp_transformer_layers = []
        for _ in range(num_mtp_layers):
            self_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
            )
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=int(hidden_dim*mtp_layer_hidden_size_factor))
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
                mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            )
            mtp_transformer_layers.append(layer)
        layer = mtp_transformer_layers.pop(0)
        if len(mtp_transformer_layers) > 0:
            mtp_transformer_layers = nn.ModuleList(mtp_transformer_layers)
        else:
            mtp_transformer_layers = None
        mtp_layer = MTPLayer(d_model=embed_dim,
                             transformer_layer=layer,
                             additional_layers=mtp_transformer_layers,
                             detach_inputs=detach_inputs,
                             use_next_embeddings=use_next_embeddings)
    else:
        mtp_layer = None

    # Assemble the final model
    return TransformerDecoderMTP(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        tied_embeddings=tied_embeddings,
        output_hidden_states=None,
        mtp_layer=mtp_layer
    )
