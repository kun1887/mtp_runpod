import torch

from evaluation.custom_generation import generate_from_recipe
from evaluation.custom_generation_fr import generate_next_token_phi_with_fr_interpolation
from evaluation.pfa_evaluation import process_data


def creativity_evaluation_fr(
    recipe,
    num_datapoints=500,
    top_k=50,
    alpha=0.0,
    equal_entropy=True,
):
    """
    FR-only variant of creativity evaluation.

    This mirrors the logic and data flow of creativity_evaluation, but uses
    Fisher-Rao interpolation during generation and intentionally ignores
    temperature (fixed to 1.0).
    """
    recipe._model.eval()
    recipe._loss_fn.eval()

    dataset = recipe._dataloader.dataset
    if hasattr(dataset, "ds"):
        dataset = dataset.ds
    dataset.split = "validation"

    datapoints = process_data(
        recipe,
        num_datapoints=min(num_datapoints, len(dataset)),
        dataset=dataset,
        batch_size=recipe.cfg.batch_size,
    )

    total_tokens = 0
    total_losses = {}
    for datapoint in datapoints:
        for key in datapoint:
            if "tokenwise" in key:
                loss_key = key.replace("tokenwise_", "eval_")
                if loss_key not in total_losses:
                    total_losses[loss_key] = 0.0
                total_losses[loss_key] += datapoint[key].sum()
        total_tokens += len(datapoint["tokens"])
    avg_losses = {k: v / total_tokens for k, v in total_losses.items()}

    # Generate data with the same flow as geodesic evaluation.
    num_seq_to_generate = 1024
    batch_size = 128
    tokenizer = recipe._tokenizer
    logit_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    logit_mask[tokenizer.special_tokens["edge: "]] = True
    generate_next_token = lambda logits, **kwargs: generate_next_token_phi_with_fr_interpolation(
        logits,
        **kwargs,
        logit_filter=logit_mask,
        alpha=alpha,
        equal_entropy=equal_entropy,
    )

    prompt_tokens = [[tokenizer.bos_id]] * batch_size
    token_sequences = []
    for _ in range(num_seq_to_generate // batch_size):
        with torch.no_grad():
            generated_tokens = generate_from_recipe(
                prompt_tokens=prompt_tokens,
                recipe=recipe,
                max_new_tokens=dataset.sequence_length - 1,
                temperature=1.0,  
                top_k=top_k,
                stop_tokens=[tokenizer.eos_id, tokenizer.pad_id],
                custom_generate_next_token=generate_next_token,
            )
        for seq in generated_tokens:
            token_sequences.append(seq["generated_tokens"].tolist())

    split_data = dataset.split_generated_data(token_sequences)
    if alpha == 1:
        breakpoint()
    creativity_scores = dataset.creativity_score(split_data, num_items=len(token_sequences))

    for key, value in creativity_scores.items():
        avg_losses[f"eval_{key}"] = value

    eval_loss = 1 / (creativity_scores["creativity_score"] + 1e-4)
    avg_losses["eval_loss"] = eval_loss

    recipe._model.train()
    recipe._loss_fn.train()
    dataset.split = "train"
    return {}, avg_losses
