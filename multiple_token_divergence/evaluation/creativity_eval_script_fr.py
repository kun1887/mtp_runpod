import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from evaluation.creativity_evaluation_fr import creativity_evaluation_fr
from training import SelfPredictionTrainingRecipeDistributed


def main(model_name, equal_entropy=True, seed=123, overwrite=False):
    checkpoint_path = f"checkpoints/{model_name}/"

    config_file = checkpoint_path + "config.yaml"
    cfg = OmegaConf.load(config_file)

    cfg.checkpointer.checkpoint_dir = checkpoint_path
    cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    cfg.optimizer.lr = 0
    cfg.compile = False
    cfg.train_from_scratch = False
    cfg.dataset.single_example = True
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"

    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe._model.eval()
    recipe._loss_fn.eval()

    task = cfg.dataset._component_.split(".")[-1].split("_dataset")[0]
    print(task)

    file_name = (
        f"evaluation/results/creativity_results_fr_{model_name}_{task}_seed_{seed}"
        f"{'' if equal_entropy else '_change_entropy'}.json"
    )
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    if os.path.exists(file_name) and not overwrite:
        print(f"Results already exist at {file_name}, skipping evaluation.")
        return
    if os.path.exists(file_name) and overwrite:
        print(f"Results already exist at {file_name}, overwriting.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    alphas = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    results_data = []
    for alpha in tqdm(alphas):
        _, eval_dict = creativity_evaluation_fr(
            recipe,
            top_k=1000,
            alpha=alpha,
            equal_entropy=equal_entropy,
        )
        tqdm.write(f"alpha {alpha}, score: {eval_dict['eval_creativity_score']:.2f}")
        results_data.append(
            {
                "alpha": alpha,
                "creativity_score": eval_dict["eval_creativity_score"],
                "uniqueness_score": eval_dict["eval_uniqueness_score"],
                "validity_score": eval_dict["eval_validity_score"],
                "novelty_score": eval_dict["eval_novelty_score"],
            }
        )

    results_dict = {
        "model_name": model_name,
        "task": task,
        "interpolation": "fisher_rao",
        "results": results_data,
    }
    with open(file_name, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved results to {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sibling_discovery_shortcut_PHi_0_1B_h3kwa2lj",
        help="Model name",
    )
    parser.add_argument(
        "--equal_entropy",
        type=int,
        default=0,
        help="Whether to use equal entropy blending",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result json files",
    )
    args = parser.parse_args()

    for seed in [4, 5]:
        main(
            args.model_name,
            equal_entropy=bool(args.equal_entropy),
            seed=seed,
            overwrite=args.overwrite,
        )
