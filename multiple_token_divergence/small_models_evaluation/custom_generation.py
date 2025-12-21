from typing import Optional, Tuple, List, Dict, Callable, Any
import torch

from torchtune.generation._generation import (sample, get_causal_mask_from_padding_mask, generate_next_token,
                                              update_stop_tokens_tracker, get_position_ids_from_padding_mask)
from torchtune.modules.transformer import TransformerDecoder

from mimo_mtp_evaluation.mimo_utils import (logit_filtering_mask,
                                            solve_slop_optimization)
from small_models_evaluation.custom_generation_utils import geodesic_interpolation, find_dist_with_entropy


ASCII_CODES = [ord(c) for c in 'abcdegfhijklmnopqrstuvwxyz']
ASCII_LOGIT_MASK = torch.ones(128).bool()
ASCII_LOGIT_MASK[ASCII_CODES] = False


def generate_next_token_mtp(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    logit_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next token given a prompt and returns the corresponding logits,
    with an option to apply a filter (mask) to the logits before sampling.

    Args:
        model (TransformerDecoder): The Transformer model used for generation.
        input_pos (torch.Tensor): A tensor of positional encodings for the prompt,
            with shape `[batch_size x seq_length]`.
        x (torch.Tensor): A tensor of token IDs for the prompt, with shape
            `[batch_size x seq_length]`.
        q (torch.Tensor): A randomly sampled tensor for the softmax sampling trick.
        mask (Optional[torch.Tensor], optional): An attention mask. Defaults to None.
        temperature (float, optional): The value to scale logits by. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.
        logit_mask (Optional[torch.Tensor], optional): A boolean tensor where `True`
            indices will be set to -inf in the logits, effectively preventing those
            tokens from being sampled. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The generated token IDs, with shape `[batch_size x 1]`.
            - The filtered logits, with shape `[batch_size x seq_length x vocab_size]`.
    """
    # The model produces logits of shape `[batch_size, seq_length, vocab_size]`.
    # We use the logits of the last token in the sequence for the next prediction.
    return_dict = model(x, input_pos=input_pos, mask=mask)
    logits = model.output(return_dict['outputs'])
    mtp_logits = model.output(return_dict['mtp_outputs'])

    if logit_mask is not None:
        # Create a mask to set disabled logits to negative infinity.
        inf_mask = torch.zeros_like(logit_mask, dtype=logits.dtype)
        inf_mask[logit_mask] = -float("inf")
        logits = logits + inf_mask.to(logits.device)

    # Sample the next token from the last time step's logits.
    next_token = sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, q=q)
    return next_token, logits


def generate_next_token_mtp_with_crossentropy_blending(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.,
    logit_filter: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next token given a prompt and returns the corresponding logits,
    with an option to apply a filter (mask) to the logits before sampling.

    Args:
        model (TransformerDecoder): The Transformer model used for generation.
        input_pos (torch.Tensor): A tensor of positional encodings for the prompt,
            with shape `[batch_size x seq_length]`.
        x (torch.Tensor): A tensor of token IDs for the prompt, with shape
            `[batch_size x seq_length]`.
        q (torch.Tensor): A randomly sampled tensor for the softmax sampling trick.
        mask (Optional[torch.Tensor], optional): An attention mask. Defaults to None.
        temperature (float, optional): The value to scale logits by. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.
        logit_filter (Optional[torch.Tensor], optional): A boolean tensor where `True`
            indices will be set to -inf in the logits, effectively preventing those
            tokens from being sampled. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The generated token IDs, with shape `[batch_size x 1]`.
            - The filtered logits, with shape `[batch_size x seq_length x vocab_size]`.
    """
    # The model produces logits of shape `[batch_size, seq_length, vocab_size]`.
    # We use the logits of the last token in the sequence for the next prediction.
    return_dict = model(x, input_pos=input_pos, mask=mask)
    logits = model.output(return_dict['outputs'])[:, -1].float()
    mtp_logits = model.output(return_dict['mtp_outputs'])[:, -1].float()

    logit_mask = logit_filtering_mask(logits.view(-1, logits.shape[-1]), top_p=top_p, top_k=top_k)
    if logit_filter is not None:
        logit_mask = torch.logical_and(logit_mask, torch.logical_not(logit_filter.to(logit_mask.device)[None]))
    logit_mask = logit_mask.view_as(logits)

    optimized_distribution = []
    sampled_tokens = []
    for i in range(logits.shape[0]):
        logits_i = logits[i, :]
        mtp_logits_i = mtp_logits[i, :]
        logit_mask_i = logit_mask[i, :]
        if logit_mask_i.sum() <= 1:
            # return deterministic distribution with argmax
            argmax = torch.argmax(logits_i)
            sampled_tokens.append(argmax)
            continue

        filtered_logits = logits_i[logit_mask_i]
        filtered_p_model = torch.softmax(filtered_logits / temperature, dim=-1)
        p_model = torch.zeros_like(logits_i)
        p_model[logit_mask_i] = filtered_p_model

        filtered_mtp_logits = mtp_logits_i[logit_mask_i]
        filtered_p_mtp = torch.softmax(filtered_mtp_logits, dim=-1)
        p_mtp = torch.zeros_like(mtp_logits_i)
        p_mtp[logit_mask_i] = filtered_p_mtp

        if alpha != 0. and len(filtered_p_model) > 1:
            x = solve_slop_optimization(filtered_p_model.cpu().numpy(),
                                        filtered_p_mtp.cpu().numpy(),
                                        alpha=alpha)
            if x is None:
                x = filtered_p_model.cpu().numpy()
            x = torch.from_numpy(x) #, dtype=filtered_p_model.dtype, device=filtered_p_model.device)
            x = x.to(filtered_p_model.device).type_as(filtered_p_model)
        else:
            x = filtered_p_model

        p_optimized_full = torch.zeros_like(p_model)
        p_optimized_full[logit_mask_i] = x

        # sample token from optimized distribution
        next_token = torch.distributions.Categorical(p_optimized_full).sample()
        sampled_tokens.append(next_token)
    sampled_tokens = torch.stack(sampled_tokens)
    return sampled_tokens.unsqueeze(1), model.output(return_dict['outputs'])


def generate_next_token_mtp_with_blending(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.,
    logit_filter: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    equal_entropy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next token given a prompt and returns the corresponding logits,
    with an option to apply a filter (mask) to the logits before sampling.

    Args:
        model (TransformerDecoder): The Transformer model used for generation.
        input_pos (torch.Tensor): A tensor of positional encodings for the prompt,
            with shape `[batch_size x seq_length]`.
        x (torch.Tensor): A tensor of token IDs for the prompt, with shape
            `[batch_size x seq_length]`.
        q (torch.Tensor): A randomly sampled tensor for the softmax sampling trick.
        mask (Optional[torch.Tensor], optional): An attention mask. Defaults to None.
        temperature (float, optional): The value to scale logits by. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.
        logit_filter (Optional[torch.Tensor], optional): A boolean tensor where `True`
            indices will be set to -inf in the logits, effectively preventing those
            tokens from being sampled. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The generated token IDs, with shape `[batch_size x 1]`.
            - The filtered logits, with shape `[batch_size x seq_length x vocab_size]`.
    """
    # The model produces logits of shape `[batch_size, seq_length, vocab_size]`.
    # We use the logits of the last token in the sequence for the next prediction.

    assert top_p >= 1.

    return_dict = model(x, input_pos=input_pos, mask=mask)
    logits = model.output(return_dict['outputs'])[:, -1].float()
    mtp_logits = model.output(return_dict['mtp_outputs'])[:, -1].float()

    if logit_filter is not None:
        logits = logits - logit_filter.to(logits.device)[None].float() * 1e10
    logit_mask = logit_filtering_mask(logits.view(-1, logits.shape[-1]), top_p=top_p, top_k=top_k)
    logit_mask = logit_mask.view_as(logits)

    filtered_logits = logits[logit_mask].reshape(logits.shape[0], -1)
    filtered_mtp_logits = mtp_logits[logit_mask].reshape(logits.shape[0], -1)

    # add -inf mask
    #if logit_mask is not None:
    #    inf_mask = torch.zeros_like(logit_mask, dtype=logits.dtype)
    #    inf_mask[logit_mask] = -float("inf")
    #    logits = logits + inf_mask.to(logits.device)
    #    mtp_logits = mtp_logits + inf_mask.to(mtp_logits.device)

    p_distributions = torch.softmax(filtered_logits / temperature, dim=-1)
    m_distributions = torch.softmax(filtered_mtp_logits / temperature, dim=-1)
    p_entropies = -torch.sum(p_distributions * torch.log(p_distributions + 1e-20), dim=-1)

    interpolated_distribution = geodesic_interpolation(p_distributions, m_distributions, alpha)
    if equal_entropy:
        optimized_distribution = find_dist_with_entropy(p_target=interpolated_distribution, H_target=p_entropies)
    else:
        optimized_distribution = interpolated_distribution

    full_distribution = torch.zeros_like(logits)
    full_distribution[logit_mask] = optimized_distribution.flatten()

    # sample token from optimized distributions
    sampled_tokens = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    for i in range(logits.shape[0]):
        next_token = torch.distributions.Categorical(full_distribution[i]).sample()
        sampled_tokens[i] = next_token

    return sampled_tokens.unsqueeze(1), model.output(return_dict['outputs'])


def generate_next_token_with_filter(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    logit_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next token given a prompt and returns the corresponding logits,
    with an option to apply a filter (mask) to the logits before sampling.

    Args:
        model (TransformerDecoder): The Transformer model used for generation.
        input_pos (torch.Tensor): A tensor of positional encodings for the prompt,
            with shape `[batch_size x seq_length]`.
        x (torch.Tensor): A tensor of token IDs for the prompt, with shape
            `[batch_size x seq_length]`.
        q (torch.Tensor): A randomly sampled tensor for the softmax sampling trick.
        mask (Optional[torch.Tensor], optional): An attention mask. Defaults to None.
        temperature (float, optional): The value to scale logits by. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.
        logit_mask (Optional[torch.Tensor], optional): A boolean tensor where `True`
            indices will be set to -inf in the logits, effectively preventing those
            tokens from being sampled. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The generated token IDs, with shape `[batch_size x 1]`.
            - The filtered logits, with shape `[batch_size x seq_length x vocab_size]`.
    """
    # The model produces logits of shape `[batch_size, seq_length, vocab_size]`.
    # We use the logits of the last token in the sequence for the next prediction.
    logits = model(x, input_pos=input_pos, mask=mask)

    if logit_mask is not None:
        # Create a mask to set disabled logits to negative infinity.
        inf_mask = torch.zeros_like(logit_mask, dtype=logits.dtype)
        inf_mask[logit_mask] = -float("inf")
        logits = logits + inf_mask

    # Sample the next token from the last time step's logits.
    next_token = sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, q=q)
    return next_token, logits


def generate_next_token_only_lowercase(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A specific wrapper around `generate_next_token_with_filter` that constrains
    generation to only lowercase ASCII characters.
    """
    return generate_next_token_with_filter(
        model=model,
        input_pos=input_pos,
        x=x,
        q=q,
        mask=mask,
        temperature=temperature,
        top_k=top_k,
        logit_mask=ASCII_LOGIT_MASK.to(x.device),
    )


def generate_from_recipe(prompt_tokens,
                         recipe,
                         max_new_tokens=100,
                         return_logits=False,
                         temperature=0.6,
                         top_k=300,
                         stop_tokens=[],
                         custom_generate_next_token=None):
    """
    Generates token sequences from a model provided within a 'recipe' object.

    This function serves as a high-level wrapper for a generation pipeline. It handles
    batching of multiple prompts, padding them to the same length, calling the core
    generation function, and then decoding and post-processing the results into a
    user-friendly format.

    Args:
        prompt_tokens (List[List[int]]): A list of prompts, where each prompt is a
            list of token IDs.
        recipe (Any): A recipe object that must contain `_model` and `_tokenizer`
            attributes.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
            Defaults to 100.
        temperature (float, optional): The temperature for sampling. Defaults to 0.6.
        top_k (int, optional): The top-k value for sampling. Defaults to 300.
        stop_tokens (List[int], optional): A list of token IDs that will halt generation
            if produced. Defaults to [].
        custom_generate_next_token (Optional[callable], optional): A custom function for
            next-token generation, useful for applying constraints. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one for each generated sequence.
            Each dictionary contains:
            - "decoded_tokens" (str): The generated text.
            - "generated_tokens" (torch.Tensor): The raw generated token IDs.
            - "neg_log_probs" (torch.Tensor): The logits of the generated tokens.
    """
    if not type(prompt_tokens[0]) == list:
        prompt_tokens = [prompt_tokens]
    for i, prompt in enumerate(prompt_tokens):
        if prompt[-1] == recipe._tokenizer.eos_id:
            # print("remove eos")
            prompt_tokens[i] = prompt[:-1]
    max_len = max([len(p) for p in prompt_tokens])
    # fill with zeros
    prompt_tokens = [p + [0] * (max_len - len(p)) for p in prompt_tokens]
    prompt = torch.tensor(prompt_tokens, dtype=torch.int, device=recipe._device)

    generated_tokens, generated_logits = generate_cumstomizable(
                model=recipe._model,
                prompt=prompt,
                max_generated_tokens=max_new_tokens,
                pad_id=recipe._tokenizer.pad_id,
                temperature=temperature,
                top_k=top_k,
                stop_tokens=stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
                only_return_log_probs=True
            )
    generated_tokens = generated_tokens[:, 1:]

    return_dicts = []
    for i in range(generated_tokens.shape[0]):
        gen_tok = generated_tokens[i]
        gen_log = generated_logits[i]
        mask = gen_tok == recipe._tokenizer.pad_id
        gen_tok = gen_tok[~mask]
        gen_log = gen_log[~mask.to(gen_log.device)]
        if len(gen_tok) == 0:
            gen_tok = torch.tensor([recipe._tokenizer.eos_id], device=gen_tok.device)
            gen_log = torch.tensor([0.0], device=gen_log.device)
        elif 'MIDI' in recipe._tokenizer.__class__.__name__:
            token_to_str = {v: k for k, v in recipe._tokenizer.vocab.items()}
            token_list = gen_tok.squeeze().tolist()
            decoded_tokens = [token_to_str[t] for t in token_list]
        else:
            decoded_tokens = recipe._tokenizer.decode(gen_tok.tolist())
        #selected_log_probs = gen_log.log_softmax(dim=-1)[range(len(gen_tok)), gen_tok.to(gen_log.device)]
        return_dict = {
            "decoded_tokens": decoded_tokens,
            "generated_tokens": gen_tok,
            "neg_log_probs": gen_log,
        }
        return_dicts.append(return_dict)
    return return_dicts

    # generated_tokens: [bsz x seq_length]
    # generated_logits: [bsz x seq_length x vocab_size]

    selected_neg_log_probs = generated_logits.log_softmax(dim=-1)[
        range(len(generated_tokens[1:])), generated_tokens[1:]
    ]

    decoded_tokens = recipe._tokenizer.decode(generated_tokens.tolist())
    return_dict = {
        "decoded_tokens": decoded_tokens,
        "generated_tokens": generated_tokens,
        "neg_log_probs": selected_neg_log_probs,
    }
    if return_logits:
        return_dict["logits"] = generated_logits
    return return_dict


@torch.no_grad()
def generate_cumstomizable(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[list[int]] = None,
    rng: Optional[torch.Generator] = None,
    custom_generate_next_token: Optional[Callable] = None,
    only_return_log_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length].
        max_generated_tokens (int): number of tokens to be generated
        pad_id (int): token ID to use for padding, default 0.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities,
            default None.
        stop_tokens (Optional[list[int]]): If specified, generation is stopped when any of these tokens are generated,
            default None.
        rng (Optional[torch.Generator]): random number generator, default None.
        custom_generate_next_token (Optional[Callable]): This argument is typically a reference to a compiled version of
            the :func:`generate_next_token` function. During autoregressive decoding, this function is called instead of the default
            :func:`generate_next_token` in order to accelerate generation. :func:`generate_next_token` will still be used for the
            first token generation - or "pre-fill" pass.
            Default is None.

    Note:
        This function has only been tested with decoder-only models.

    Examples:
        >>> import torch
        >>> from torchtune.models.llama3 import llama3_tokenizer
        >>> from torchtune.models.llama3 import llama3_8b
        >>> from torchtune.generation import generate
        >>> from torchtune.training.checkpointing import FullModelHFCheckpointer
        >>> from torchtune.data import Message

        >>> model = llama3_8b().cuda()

        >>> checkpointer = FullModelHFCheckpointer(
        ...     checkpoint_dir="/tmp/Meta-Llama-3-8B-Instruct",
        ...     checkpoint_files=[
        ...         "model-00001-of-00004.safetensors",
        ...         "model-00002-of-00004.safetensors",
        ...         "model-00003-of-00004.safetensors",
        ...         "model-00004-of-00004.safetensors",
        ...     ],
        ...     model_type="LLAMA3",
        ...     output_dir="/tmp/torchtune/llama3_8b",
        ... )
        >>> checkpoint = checkpointer.load_checkpoint()
        >>> model.load_state_dict(checkpoint["model"])

        >>> tokenizer = llama3_tokenizer("/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
        >>> messages = [
        ...     Message(role="assistant", content="Hi my name is"),
        ... ]
        >>> prompt = tokenizer({"messages": messages}, inference=True)
        >>> output, logits = generate_from_recipe(model, torch.tensor(prompt["tokens"], device='cuda'), max_generated_tokens=100, pad_id=0)
        >>> print(tokenizer.decode(output[0].tolist()))

        >>> Hi my name is Marley. Nice to meet you, Marley! How are you doing today?... [truncated]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape ``[bsz x seq_len + num_generated_tokens]`` where ``num_generated_tokens``
                may be less than ``max_generated_tokens`` if ``stop_tokens`` are provided.
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape ``[bsz x num_generated_tokens x vocab_size]``.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + max_generated_tokens

    generated_tokens = prompt.clone()
    incremental_decoding = model.caches_are_enabled()

    # grab the correct max_seq_len to generate full causal masks/position ids
    # this is the model's max cache len if incremental decoding, or the sequence
    # length otherwise
    max_seq_len = (
        total_response_length
        if not incremental_decoding
        else model.decoder_max_cache_seq_len
    )

    padding_masks = generated_tokens != pad_id

    if not padding_masks.all():
        # we have padding in the prompt due to varying-length sequences in a batch
        # extend padding masks out to the correct seq len
        padding_masks = torch.nn.functional.pad(
            padding_masks, (0, max_generated_tokens), value=True
        )

        # generate the full causal mask for the whole padding mask with padding ignored
        masks = get_causal_mask_from_padding_mask(
            padding_masks, target_seq_len=max_seq_len
        )

        # right-shift position IDs to account for padding
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        # just use a regular causal mask if there is no padding
        masks = torch.tril(
            torch.ones(
                total_response_length,
                max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            )
        ).unsqueeze(0).repeat(bsz, 1, 1)
        input_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        ).unsqueeze(0).repeat(bsz, 1)

    if incremental_decoding:
        # if KV-caches are enabled, we need a causal mask of shape [bsz, prompt_length, max_cache_len]
        # to match the key/value cache tensor shapes
        curr_masks = masks[:, :prompt_length]
    else:
        # otherwise the causal mask is shape [bsz, prompt_length, prompt_length] because key/value
        # tensors are of identical shape to the prompt
        curr_masks = masks[:, :prompt_length, :prompt_length]

    q = None
    if rng is not None:
        uniform_val = torch.rand(
            bsz,
            model.tok_embeddings.num_embeddings,
            generator=rng,
            device=prompt.device,
        )
        epsilon = torch.finfo(uniform_val.dtype).eps / 2
        condition = uniform_val >= 1.0 - epsilon
        q = -torch.where(condition, -epsilon, torch.log(uniform_val))

    next_token_fn = (
        custom_generate_next_token
        if custom_generate_next_token is not None
        else generate_next_token
    )

    tokens, generated_logits = next_token_fn(
        model,
        input_pos=input_pos[:, :prompt_length].squeeze(),
        mask=curr_masks,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
        q=q,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    curr_pos = prompt_length

    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype)
        if stop_tokens
        else None
    )

    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(
            tokens, stop_tokens, stop_token_reached
        )
        if stop_token_reached.all().item():
            return generated_tokens, generated_logits

    for _ in range(max_generated_tokens - 1):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        # if incremental decoding is enabled, we can use the current position
        # otherwise, we take the whole sequence up to the current position
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos].contiguous()
            curr_masks = masks[:, curr_pos, None, :].contiguous()
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = None
        if rng is not None:
            uniform_val = torch.rand(
                bsz,
                model.tok_embeddings.num_embeddings,
                generator=rng,
                device=prompt.device,
            )
            epsilon = torch.finfo(uniform_val.dtype).eps / 2
            condition = uniform_val >= 1.0 - epsilon
            q = -torch.where(condition, -epsilon, torch.log(uniform_val))

        tokens, logits = next_token_fn(
            model,
            input_pos=curr_input_pos,
            x=tokens.clone(),
            mask=curr_masks,
            temperature=temperature,
            top_k=top_k,
            q=q,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        generated_logits = torch.cat([generated_logits, logits[:, -1:]], dim=1)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if stop_token_reached.all():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens.masked_fill_(~stop_token_mask.bool(), pad_id)
        generated_logits *= stop_token_mask[:, -generated_logits.shape[1] :, None]

    return generated_tokens, generated_logits