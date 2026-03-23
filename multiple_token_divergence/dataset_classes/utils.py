from typing import Any, Optional, Union, Callable

import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
from torchtune.utils._logging import get_logger, log_once
from torchtune.modules.attention_utils import create_block_causal_mask, _get_document_ids_from_seq_lens
try:
    from torchtune.modules.attention_utils import create_block_causal_mask_flex
except ImportError:
    from torch.nn.attention.flex_attention import create_block_mask as create_block_causal_mask_flex

_log: logging.Logger = get_logger()

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask
    _MaskType = Union[torch.Tensor, BlockMask]
else:
    _MaskType = torch.Tensor


def dummy_collate(batch: list[Any]):
    return batch


def padded_collate_packed(
    batch: list[PACK_TYPE],
) -> dict[str, torch.Tensor]:
    """Collate packed sequences into a batch. Only convert the seq lens into
    a block mask for use with attention. Tokens, labels, and input_pos are
    already padded to the same length within :class:`~torchtune.datasets.PackedDataset`.

    Args:
        batch (list[PACK_TYPE]): A list of pack dictionaries containing the following keys:
            - tokens: input token ids
            - labels: label token ids
            - input_pos: relative position ids for each sequence in pack
            - seq_lens: lengths of each sample within the pack

    Returns:
        dict[str, torch.Tensor]: Collated input, label, input_pos, mask tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3, 4, 5, 6], "labels": [7, 8, 9, 10, 11, 12],
        >>>     "input_pos": [0, 1, 2, 0, 1, 0], "seq_lens": [3, 2, 1]},
        >>>    {"tokens": [13, 14, 15, 16, 17, 18], "labels": [19, 20, 21, 22, 23, 24],
        >>>     "input_pos": [0, 1, 0, 1, 0, 1], "seq_lens": [2, 2, 2]},
        >>> ]
        >>> collated = padded_collate_packed(
        >>>    batch=token_pairs,
        >>>    device=device,
        >>> )
        >>> collated["mask"]
        >>> tensor([
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [1, 1, 1, 0, 0, 0],
        >>>  [0, 0, 0, 1, 0, 0],
        >>>  [0, 0, 0, 1, 1, 0],
        >>>  [0, 0, 0, 0, 0, 1]],
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [0, 0, 1, 0, 0, 0],
        >>>  [0, 0, 1, 1, 0, 0],
        >>>  [0, 0, 0, 0, 1, 0],
        >>>  [0, 0, 0, 0, 1, 1]])
    """

    tokens = torch.stack([x["tokens"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    input_pos = torch.stack([x["input_pos"] for x in batch])
    seq_lens = [x["seq_lens"] for x in batch]

    block_mask = packed_block_causal_mask(
        seq_lens=seq_lens,
    )

    return {
        "tokens": tokens,
        "labels": labels,
        "input_pos": input_pos,
        "mask": block_mask,
    }


def _flex_block_causal_mask_mod(document_ids, b, h, q_idx, kv_idx):
    """
    Top-level, picklable function defining the logic of a block causal mask.
    Combines a standard causal mask and a block diagonal document mask.

    See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
    for an illustration.
    """
    causal_mask = q_idx >= kv_idx
    document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
    return causal_mask & document_mask


def packed_block_causal_mask(
    seq_lens: list[torch.Tensor],
) -> _MaskType:
    """
    Create a block causal document mask for a batch of packed sequences. If
    flex attention is supported by the current hardware, block causal logic and
    passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. If on an older version, a standard 2D block causal mask is created and returned.

    Args:
        seq_lens (list[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    if _SUPPORTS_FLEX_ATTENTION:
        document_ids = _get_document_ids_from_seq_lens(seq_lens)
        batch_size, max_seq_len = document_ids.shape
        document_ids = document_ids.to("cuda")

        # Instead of passing a tensor mask, flex attention requires a mask_mod function
        # that determines which elements of QK^T should be included in the attention
        # computation prior to the softmax. For sample packing, we need both the
        # logic for both causal mask and document mask. See PyTorch's official
        # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods

        # Create a partial function with document_ids "baked in".
        # This object is picklable.
        mask_mod = functools.partial(_flex_block_causal_mask_mod, document_ids)

        return create_block_causal_mask_flex(
            mask_mod,
            batch_size,
            None,
            max_seq_len,
            max_seq_len,
            device="cuda",
        )
    else:
        return create_block_causal_mask(seq_lens=seq_lens)
