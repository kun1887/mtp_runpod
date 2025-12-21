import os
import torch
from omegaconf import OmegaConf
from training import SelfPredictionTrainingRecipeDistributed
from utils.misc import bootstrapped_mean_and_ci
import argparse
import numpy as np
import json
from tqdm.auto import tqdm

from small_models_evaluation.creativity_evaluation import creativity_evaluation

def main(model_name, equal_entropy=True, seed=123):
    # model_name = "creativity_0_1B_MTP_4flno3uz"  # sibling discovery
    # model_name = "creativity_0_1B_MTP_udfvbjrd"  # triangle discovery
    # model_name = "creativity_0_1B_MTP_7h8a91gt"  # line construction
    # model_name = "creativity_0_1B_MTP_pmepwvx5"  # circle construction

    checkpoint_path = f"checkpoints/{model_name}/"  # MTP with next embedding access, trained on all tasks

    config_file = checkpoint_path + "config.yaml"
    model_id = checkpoint_path.split('_')[-1].split('/')[0]
    cfg = OmegaConf.load(config_file)

    cfg.checkpointer.checkpoint_dir = checkpoint_path
    cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    cfg.optimizer.lr = 0
    # cfg.model.max_seq_len = 4096
    cfg.compile = False
    cfg.train_from_scratch = False
    cfg.dataset.single_example = True
    # cfg.model.use_self_prediction = True
    # cfg.model.self_prediction_layer = 13
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe._model.eval()
    recipe._loss_fn.eval()

    task = cfg.dataset._component_.split('.')[-1].split('_dataset')[0]
    print(task)


    file_name = f"/results/creativity_results_{model_name}_{task}_seed_{seed}{'' if equal_entropy else '_change_entropy'}.json"
    if os.path.exists(file_name):
        print(f"Results already exist at {file_name}, skipping evaluation.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    parameters = []
    temperature_ranges = {
        "sibling_discovery": [(0.6, 1.4)],
        "triangle_discovery": [(0.2, 1.0)],
        "line_construction": [(0.1, 0.9)],
        "circle_construction": [(0.1, 0.9)],
    }

    temperature_range = temperature_ranges[task]
    # go through the temperature range with step size 0.1
    for temperature in np.arange(temperature_range[0][0], temperature_range[0][1] + 1e-3, 0.1):
        for alpha in [-1, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.]:
            parameters.append((temperature, alpha))

    results_data = []
    for i in tqdm(range(len(parameters))):
        temperature, alpha = parameters[i]
        # print(f"Evaluating creativity at temperature {temperature}, alpha {alpha}")
        _, eval_dict = creativity_evaluation(recipe,
                                             temperature=temperature,
                                             top_k=1000,
                                             alpha=alpha,
                                             equal_entropy=equal_entropy)
        tqdm.write(f"Temperature {temperature}, alpha {alpha}, score: {eval_dict['eval_creativity_score']:.2f}")
        results_data.append({
            "temperature": temperature,
            "alpha": alpha,
            "creativity_score": eval_dict["eval_creativity_score"],
            "uniqueness_score": eval_dict["eval_uniqueness_score"],
            "validity_score": eval_dict["eval_validity_score"],
            "novelty_score": eval_dict["eval_novelty_score"],
        })

    results_dict = {
        "model_name": model_name,
        "task": task,
        "results": results_data
    }
    with open(file_name, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved results to {file_name}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="creativity_0_1B_MTP_4flno3uz", help="Model name")
    parser.add_argument("--equal_entropy", type=int, default=1, help="Whether to use equal entropy blending")
    # parser.add_argument("--num_seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    for seed in [ 4, 5 ]: #[1, 2, 3, 4, 5]:
        main(args.model_name,
             equal_entropy=bool(args.equal_entropy),
             seed=seed)
