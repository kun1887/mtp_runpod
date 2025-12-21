import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import pandas as pd
import seaborn as sns
import gc
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import argparse

from mimo_mtp_evaluation.mimo_utils import predict_and_metrics, cleanup_tokens, index_of_subsequence
from mimo_mtp_evaluation.math500_generation import extract_math_ans_from_response
from config import HF_CACHE_DIR

def evaluate_mtp_model_on_MATH(model_id, dataset_path, response_samples_path, log_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = Path(dataset_path)

    # load dataset
    filelist = []
    for file in tqdm(dataset_path.rglob("*.json")):
        # for file in subset.iterdir("*.json"):
        with open(file, "r") as f:
            ctx = json.loads(f.read())
            ctx['unique_id'] = str(file).split("MATH500/")[-1]
            filelist.append(ctx)
    df = pd.DataFrame.from_records(filelist)
    df["levelint"] = [int(x.split()[-1]) for x in df["level"]]
    print(f"Loaded {len(df)} problems from MATH dataset")
    longest_input_so_far = 0

    # load model responses
    with open(response_samples_path, "r") as f:
        responses = json.loads(f.read())

    responses = {r['unique_id']: r for r in responses}

    # load model and tokenizer
    kwargs = dict(
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=kwargs["cache_dir"])

    # process each problem
    datapoints = []
    for i, row in tqdm(df.iterrows()):
        if row['unique_id'] not in responses:
            print(f"Skipping {row['unique_id']} as no response found")
            continue

        gt_solution = extract_math_ans_from_response(row['solution'])
        all_responses = responses[row['unique_id']]

        all_reponse_data = []
        for j, response in enumerate(all_responses['samples_token_ids']):
            response_string = response['text']
            pred_solution = extract_math_ans_from_response(response_string)
            is_correct = pred_solution == gt_solution

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": response_string}
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            toks = cleanup_tokens(tokenizer.tokenize(full_text))
            inputs = tokenizer(full_text, return_tensors="pt").to(device)

            input_length = inputs['input_ids'].shape[1]
            if input_length > 4240:
                print(f"Skipping {i} response {j} as too long: {input_length} tokens")
                continue

            if input_length > longest_input_so_far:
                longest_input_so_far = input_length
                print(f"New longest input: {longest_input_so_far} tokens")

            start_index_of_assistant = index_of_subsequence(['<|im_start|>', 'assistant', '\n'], toks)
            if start_index_of_assistant is None:
                raise Exception("No start index")

            assistant_mask = np.zeros(len(toks), dtype=bool)
            assistant_mask[start_index_of_assistant + 3:] = True

            # print(f"Predicting: {i}: {inputs['input_ids'].shape}")
            with torch.no_grad():
                _, response_datapoint, _ = predict_and_metrics(model, inputs)
                response_datapoint = {k: v.cpu().float().numpy() for k, v in response_datapoint.items()}
            response_datapoint["tokens"] = toks
            response_datapoint["mask"] = assistant_mask
            response_datapoint["correct"] = is_correct
            all_reponse_data.append(response_datapoint)
            del inputs
            gc.collect()
            # torch.cuda.empty_cache()

        question_datapoint = {}
        question_datapoint["level"] = row["levelint"]
        question_datapoint["type"] = row["type"]
        question_datapoint["index"] = i
        question_datapoint["unique_id"] = row["unique_id"]
        question_datapoint["responses"] = all_reponse_data

        datapoints.append(question_datapoint)

        if i % 100 == 0:
            save_results(log_path, datapoints, model_id, dataset_path)

    save_results(log_path, datapoints, model_id, dataset_path)


def save_results(log_path, datapoints, model_id, dataset_path):
    config_dict = {
        "model_id": model_id,
        "dataset": "MATH",
        "dataset_path": 'MATH' + str(dataset_path).split("MATH")[-1],
    }
    pickle.dump([config_dict] + datapoints, open(log_path, "wb"))
    print(f"Saved {len(datapoints)} datapoints to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTP model on GSM8K dataset")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        default="XiaomiMiMo/MiMo-7B-SFT",
        # default = "XiaomiMiMo/MiMo-7B-Base"
        # default = "XiaomiMiMo/MiMo-7B-RL"
        # default = "XiaomiMiMo/MiMo-7B-RL-Zero"
        help="Model ID to evaluate",
    )
    args = parser.parse_args()
    model_id = args.model_id

    dataset_path = "../data/MATH500/test"
    response_samples_path = "mimo_mtp_evaluation/results/math500_samples_topk50_10_responses.json"
    log_name = f"MATH500_pre-processed_10_reponses_eval_{model_id.split('/')[-1]}.pkl"

    log_path = Path("mimo_mtp_evaluation/results") / log_name

    evaluate_mtp_model_on_MATH(model_id, dataset_path, response_samples_path, log_path)


if __name__ == "__main__":
    main()

