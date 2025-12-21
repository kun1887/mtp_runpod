import argparse
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

from mimo_mtp_evaluation.mimo_utils import predict_and_metrics, cleanup_tokens, index_of_subsequence


def evaluate_mtp_model_on_MATH(model_id, dataset_path, log_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_dir = "/home/vincent/storage/huggingface/cache"
    if not Path(cache_dir).exists():
        cache_dir = None
    dataset_path = Path(dataset_path)

    # load model and tokenizer
    kwargs = dict(
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=kwargs["cache_dir"])

    # load dataset
    filelist = []
    for file in tqdm(dataset_path.rglob("*.json")):
        # for file in subset.iterdir("*.json"):
        with open(file, "r") as f:
            ctx = json.loads(f.read())
            ctx["names"] = str(file)
            filelist.append(ctx)
    df = pd.DataFrame.from_records(filelist)
    df["levelint"] = [int(x.split()[-1]) for x in df["level"]]
    print(f"Loaded {len(df)} problems from MATH dataset")

    # process each problem
    losses = []
    datapoints = []
    reduced_metrics = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        toks = cleanup_tokens(tokenizer.tokenize(full_text))
        inputs = tokenizer(full_text, return_tensors="pt").to(device)

        start_index_of_assistant = index_of_subsequence(['<|im_start|>', 'assistant', '\n'], toks)

        if start_index_of_assistant is None:
            print(f"Tokens: {toks}")
            print(f"Full Text: {full_text}")
            raise Exception("No start index")

        assistant_mask = np.zeros(len(toks), dtype=bool)
        assistant_mask[start_index_of_assistant + 3:] = True

        print(
            f"Predicting: {i}: {inputs['input_ids'].shape}; "
            f"Type: {row.type} and Diff: {row.levelint}. Name. {row.names} "
            f"and split {start_index_of_assistant}"
        )
        with torch.no_grad():
            loss, datapoint, reduced_metric = predict_and_metrics(model, inputs)
            datapoint = {k: v.cpu().float().numpy() for k, v in datapoint.items()}
        datapoint["tokens"] = toks
        datapoint["mask"] = assistant_mask
        datapoint["level"] = row["levelint"]
        datapoint["type"] = row["type"]
        datapoint["name"] = row["names"]
        datapoint["idex"] = i

        del inputs
        gc.collect()
        losses.append(loss)
        datapoints.append(datapoint)
        reduced_metrics.append(reduced_metric)

        if i % 1000 == 0:
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
    parser = argparse.ArgumentParser(description="Evaluate MTP model on MATH test dataset")
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="XiaomiMiMo/MiMo-7B-SFT",
        help="Model ID to evaluate",
    )
    args = parser.parse_args()
    model_id = args.model_id

    dataset_path = "../data/MATH/test"
    log_name = f"math500_gt_eval_{model_id.split('/')[-1]}_{dataset_path.split('MATH/')[-1]}.pkl"

    log_path = Path("mimo_mtp_evaluation/results") / log_name

    evaluate_mtp_model_on_MATH(model_id, dataset_path, log_path)


if __name__ == "__main__":
    main()
