from modules.architectures import TransformerDecoderShortcutPHi, TransformerDecoderMTP
from models.component_builders import llama3_mtp, llama3_shortcut_phi


def llama3_ascii_mtp_0_1b(
        max_seq_len=2048,
        use_mtp=True,
        detach_inputs=False,
        use_next_embeddings=True,
        tied_embeddings=True) -> TransformerDecoderMTP:
    """
    Config approximated from GPT2-small
    """
    return llama3_mtp(
        vocab_size=128,
        num_layers=12,
        num_heads=6,
        num_kv_heads=6,
        embed_dim=768,
        max_seq_len=max_seq_len,
        intermediate_dim=768 * 8 // 3,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        tied_embeddings=tied_embeddings,
        use_mtp=use_mtp,
        detach_inputs=detach_inputs,
        use_next_embeddings=use_next_embeddings
    )


def llama3_creativity_mtp_0_1b(
        max_seq_len=2048,
        use_mtp=True,
        num_layers=12,
        detach_inputs=False,
        use_next_embeddings=True,
        tied_embeddings=True) -> TransformerDecoderMTP:
    """
    Config approximated from GPT2-small
    """
    return llama3_mtp(
        vocab_size=10_000,
        num_layers=num_layers,
        num_heads=6,
        num_kv_heads=6,
        embed_dim=768,
        max_seq_len=max_seq_len,
        intermediate_dim=768 * 8 // 3,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        tied_embeddings=tied_embeddings,
        use_mtp=use_mtp,
        detach_inputs=detach_inputs,
        use_next_embeddings=use_next_embeddings
    )


def llama3_ascii_shortcut_phi_0_1b(
        max_seq_len=2048,
        use_shortcut_phi=True,
        shortcut_PHi_layer_position=None,
        detach_inputs=False,
        use_next_embeddings=True,
        tied_embeddings=True,
        learnable_critic=False) -> TransformerDecoderShortcutPHi:
    """
    Config approximated from GPT2-small
    """
    return llama3_shortcut_phi(
        vocab_size=128,
        num_layers=12,
        num_heads=6,
        num_kv_heads=6,
        embed_dim=768,
        max_seq_len=max_seq_len,
        intermediate_dim=768 * 8 // 3,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        tied_embeddings=tied_embeddings,
        use_shortcut_phi=use_shortcut_phi,
        shortcut_PHi_layer_position=shortcut_PHi_layer_position,
        detach_inputs=detach_inputs,
        use_next_embeddings=use_next_embeddings,
        learnable_critic=learnable_critic
    )


def mistral_mtp_7b(max_seq_len=2048,
                   use_mtp=True,
                   detach_inputs=False,
                   use_next_embeddings=True,
                   tied_embeddings=False,
                   mtp_layer_hidden_size_factor=1.,
                   num_mtp_layers=1) -> TransformerDecoderMTP:
    return llama3_mtp(
        vocab_size=32768,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=max_seq_len,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1_000_000,
        scale_factor=1,
        tied_embeddings=tied_embeddings,
        use_mtp=use_mtp,
        detach_inputs=detach_inputs,
        use_next_embeddings=use_next_embeddings,
        mtp_layer_hidden_size_factor=mtp_layer_hidden_size_factor,
        num_mtp_layers=num_mtp_layers
    )