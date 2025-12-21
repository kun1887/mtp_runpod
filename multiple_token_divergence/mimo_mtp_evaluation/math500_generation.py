import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import pickle

from small_models_evaluation.custom_generation_utils import geodesic_interpolation, find_dist_with_entropy

from mimo_mtp_evaluation.mimo_utils import (generate_with_mtp, logit_filtering_mask,
                                            solve_slop_optimization)
from config import HF_CACHE_DIR

MATH_GENERAL_PROMPT = """Answer the following math question, given in LaTeX format, clearly and concisely, and present the final answer as \(\\boxed{x}\), where x is the fully simplified solution.

Example:
**Question:** \(\int_0^1 (3x^2 + 2x) \,dx\)
**Solution:** \(\int (3x^2 + 2x) \,dx = x^3 + x^2 + C\) Evaluating from 0 to 1: \((1^3 + 1^2) - (0^3 + 0^2) = 1 + 1 - 0 = 2 \\boxed{2}\)

Now, solve the following question:
"""


def extract_math_ans_from_response(answer: str,
                                   eos=None,
                                   correct_answer=None,
                                   look_for_answer_in_last_n_letters=10):
    if eos:
        answer = answer.split(eos)[0].strip()

    tex_string = answer
    """
    Returns the content of the *last* \boxed{...} found in tex_string,
    correctly handling balanced braces (including nested ones).
    """
    # Find the last occurrence of "\boxed{"
    start_marker = r'\boxed{'
    idx = tex_string.rfind(start_marker)
    if idx == -1:
        return answer  # no \boxed{ found

    # We begin parsing after "\boxed{"
    start = idx + len(start_marker)

    # Use a simple stack/brace counter to find the matching "}"
    depth = 1
    i = start
    while i < len(tex_string) and depth > 0:
        if tex_string[i] == '{':
            depth += 1
        elif tex_string[i] == '}':
            depth -= 1
        i += 1

    # If depth != 0, there was no proper closing brace
    if depth != 0:
        return tex_string

    # The content is everything between start and the brace that closed the depth
    # We incremented i one extra time after matching, so the content ends at i-1
    answer = tex_string[start: i - 1].strip()

    # additional cleanup:
    # replace \dfrac with \frac
    answer = answer.replace(r'\dfrac', r'\frac')
    answer = answer.replace(r'^\circ', '')
    answer = answer.replace(r',\!', '')

    answer = answer.replace(' ', '')
    answer = answer.replace('\n', '')
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace('\\;', '')
    return answer


def cross_entropy_optimized_blending_function(logits, mtp_logits,
                                              top_p=0.95, top_k=100, alpha=-0.9, optimize_cross_entropy=True):
    logit_mask = logit_filtering_mask(logits, top_p=top_p, top_k=top_k)
    if logit_mask.sum() <= 1:
        # return deterministic distribution with argmax
        p_optimized = torch.zeros_like(logits)
        p_optimized[0, torch.argmax(logits[0])] = 1.0
        return p_optimized
    filtered_logits = logits[logit_mask]
    filtered_p_model = torch.softmax(filtered_logits, dim=-1)
    p_model = torch.zeros_like(logits)
    p_model[logit_mask] = filtered_p_model

    filtered_mtp_logits = mtp_logits[logit_mask]
    filtered_p_mtp = torch.softmax(filtered_mtp_logits, dim=-1)
    p_mtp = torch.zeros_like(mtp_logits)
    p_mtp[logit_mask] = filtered_p_mtp

    if optimize_cross_entropy and alpha != 0 and len(filtered_p_model) > 1:
        #x = find_maximum_cross_entropy_distribution(filtered_p_model.cpu().numpy(), filtered_log_p_mtp.cpu().numpy(),
        #                                            alpha=alpha)
        x = solve_slop_optimization(filtered_p_model.cpu().numpy(), filtered_p_mtp.cpu().numpy(),
                                    alpha=alpha)
        if x is None:
            x = filtered_p_model.cpu().numpy()
    else:
        x = filtered_p_model.cpu().numpy()

    p_optimized = torch.zeros_like(p_model)
    p_optimized[logit_mask] = torch.tensor(x, dtype=p_model.dtype, device=p_model.device)
    return p_optimized


def interpolation_blending_function(logits, mtp_logits, alpha=-0.9, top_p=1., top_k=100, temperature=1., equal_entropy=True):
    logit_mask = logit_filtering_mask(logits, top_p=top_p, top_k=top_k)
    if logit_mask.sum() <= 1:
        # return deterministic distribution with argmax
        p_optimized = torch.zeros_like(logits)
        p_optimized[0, torch.argmax(logits[0])] = 1.0
        return p_optimized

    filtered_logits = logits[logit_mask]
    filtered_p_model = torch.softmax(filtered_logits / temperature, dim=-1)
    p_entropy = -torch.sum(filtered_p_model * torch.log(filtered_p_model + 1e-10))

    filtered_mtp_logits = mtp_logits[logit_mask]
    filtered_p_mtp = torch.softmax(filtered_mtp_logits / temperature, dim=-1)

    interpolated_distribution = geodesic_interpolation(filtered_p_model[None],
                                                       filtered_p_mtp[None],
                                                       alpha)
    if equal_entropy:
        optimized_distribution = find_dist_with_entropy(p_target=interpolated_distribution, H_target=p_entropy[None])
    else:
        optimized_distribution = interpolated_distribution

    p_optimized = torch.zeros_like(logits)
    p_optimized[logit_mask] = optimized_distribution.flatten()
    return p_optimized


def process_problem(model, tokenizer, device, problem_data,
                    max_response_length=4096,
                    top_p=1.,
                    top_k=100,
                    temperature=1.,
                    alpha=-0.9,
                    equal_entropy=True):
    """
    Processes a single problem from the dataset.
    This is the core logic that will be executed by each worker.
    """
    i, row = problem_data
    gt_answer = extract_math_ans_from_response(row['solution'])
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": MATH_GENERAL_PROMPT + row["problem"]},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    blending_function = lambda logits, mtp_logits: interpolation_blending_function(
        logits, mtp_logits,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        alpha=alpha,
        equal_entropy=equal_entropy)

    # Generate the response
    generated_ids_cached = generate_with_mtp(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        custom_blending_function=blending_function,
        max_new_tokens=max_response_length,
        sampling_method='multinomial_direct'
    )

    num_generated_tokens = generated_ids_cached.shape[1] - inputs['input_ids'].shape[1]

    solution_text = tokenizer.decode(generated_ids_cached[0], skip_special_tokens=False)
    stripped_solution_text = solution_text.split('<|im_start|>assistant\n')[-1].replace('<|im_end|>', '').strip()

    generated_answer = extract_math_ans_from_response(stripped_solution_text, correct_answer=gt_answer)
    answer_is_correct = generated_answer == gt_answer

    return {
        "index": i,
        "problem": row["problem"],
        "gt_answer": gt_answer,
        "generated_CoT": stripped_solution_text,
        "generated_answer": generated_answer,
        "num_generated_tokens": num_generated_tokens,
        "correct": answer_is_correct,
        "level": row["level"],
        "levelint": row["levelint"],
        "type": row["type"],
        "unique_id": row["unique_id"]
    }


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
            "level": "Level 5",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
            "level": "Level 5",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
            "level": "Level 5",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
            "level": "Level 5",
        },
    ]


def process_problem_few_shot(model, tokenizer, device, problem_data,
                    max_response_length=4096,
                    top_p=1.,
                    top_k=100,
                    temperature=1.,
                    alpha=-0.9,
                    equal_entropy=True):
    """
    Processes a single problem from the dataset.
    This is the core logic that will be executed by each worker.
    """
    i, row = problem_data
    gt_answer = extract_math_ans_from_response(row['solution'])
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": MATH_GENERAL_PROMPT + row["problem"]},
    ]
    few_shot_samples = list_fewshot_samples()
    # assemble prompt with few-shot samples
    few_shot_text = ""
    for sample in few_shot_samples:
        few_shot_text += f"**Question:** {sample['problem']}\n**Solution:** {sample['solution']}\n\n"
    few_shot_text += f"**Question:** {row['problem']}\n**Solution:** "
    inputs = tokenizer(few_shot_text, return_tensors="pt").to(device)

    blending_function = lambda logits, mtp_logits: interpolation_blending_function(
        logits, mtp_logits,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        alpha=alpha,
        equal_entropy=equal_entropy)

    # Generate the response
    generated_ids_cached = generate_with_mtp(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        custom_blending_function=blending_function,
        max_new_tokens=max_response_length,
        sampling_method='multinomial_direct',
        stop_strings=['Final Answer:', '**Question**']
    )

    num_generated_tokens = generated_ids_cached.shape[1] - inputs['input_ids'].shape[1]

    solution_text = tokenizer.decode(generated_ids_cached[0], skip_special_tokens=False)
    stripped_solution_text = solution_text.split('<|im_start|>assistant\n')[-1].replace('<|im_end|>', '').strip()

    generated_answer = extract_math_ans_from_response(stripped_solution_text, correct_answer=gt_answer)
    answer_is_correct = generated_answer == gt_answer

    return {
        "index": i,
        "problem": row["problem"],
        "gt_answer": gt_answer,
        "generated_CoT": stripped_solution_text,
        "generated_answer": generated_answer,
        "num_generated_tokens": num_generated_tokens,
        "correct": answer_is_correct,
        "level": row["level"],
        "levelint": row["levelint"],
        "type": row["type"],
        "unique_id": row["unique_id"]
    }


# The original main function can be kept for single-GPU runs or debugging
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="XiaomiMiMo/MiMo-7B-Base")
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = Path("../../data/MATH500/test")

    # Load model and tokenizer
    kwargs = dict(
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=kwargs["cache_dir"])

    # Load dataset
    filelist = []
    for file in tqdm(dataset_path.rglob("*.json")):
        with open(file, "r") as f:
            ctx = json.loads(f.read())
            ctx['unique_id'] = str(file).split('MATH500/')[-1]
            filelist.append(ctx)
    df = pd.DataFrame.from_records(filelist)
    df["levelint"] = [int(x.split()[-1]) for x in df["level"]]
    print(f"Loaded {len(df)} problems from MATH dataset")

    # Splitting logic
    split_size = len(df) // args.num_splits
    start_idx = args.split * split_size
    end_idx = (args.split + 1) * split_size if args.split < args.num_splits - 1 else len(df)

    datapoints = []
    for i, row in tqdm(df.iloc[start_idx:end_idx].iterrows(), total=end_idx - start_idx):
        result = process_problem_few_shot(model, tokenizer, device, (i, row),
                                          alpha=0., temperature=1., max_response_length=512,)
        print(f"Index {i}: Correct: {result['correct']}")
        datapoints.append(result)

    # Save results
    log_name = f"MATH500_generate_{args.model_id.split('/')[-1]}_split_{args.split}_of_{args.num_splits}.pkl"
    log_path = Path("mimo_mtp_evaluation/results") / log_name
    pickle.dump(datapoints, open(log_path, "wb"))
    print(f"Saved {len(datapoints)} results to {log_path}")


if __name__ == "__main__":
    main()