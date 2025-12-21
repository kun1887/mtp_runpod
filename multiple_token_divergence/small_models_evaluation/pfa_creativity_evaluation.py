import os
from omegaconf import OmegaConf
from training import SelfPredictionTrainingRecipeDistributed
from utils.misc import bootstrapped_mean_and_ci
import argparse
import numpy as np
import json
import wandb

from small_models_evaluation.pfa_evaluation import evaluate_language_generation_creativity


def pfa_creativity_evaluation(model_name, num_samples=1000, batch_size=100,
                              temperature=1., top_k=18, top_p=1., alpha=0.,
                              equal_entropy=True, generation_length=10):

    checkpoint_path = f"checkpoints/{model_name}/"  # MTP with next embedding access, trained on all tasks

    config_file = checkpoint_path + "config.yaml"
    model_id = checkpoint_path.split('_')[-1].split('/')[0]
    cfg = OmegaConf.load(config_file)

    cfg.checkpointer.checkpoint_dir = checkpoint_path
    cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    cfg.optimizer.lr = 0
    #cfg.model.max_seq_len = 4096
    cfg.compile = False
    cfg.train_from_scratch = False
    # cfg.model.use_self_prediction = True
    # cfg.model.self_prediction_layer = 13
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe._model.eval()
    recipe._loss_fn.eval()

    generation_eval_dict = evaluate_language_generation_creativity(recipe,
                                                               num_samples=num_samples,
                                                               batch_size=batch_size,
                                                               top_k=top_k,
                                                               top_p=top_p,
                                                               temperature=temperature,
                                                               alpha=alpha,
                                                               equal_entropy=equal_entropy,
                                                               generation_length=generation_length)

    results_dict = {
        "model_id": model_id,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "alpha": alpha,
        "creativity_scores": generation_eval_dict["creativity_scores"],
        "novelty_scores": generation_eval_dict["novelty_scores"],
        "uniqueness_scores": generation_eval_dict["uniqueness_scores"],
        "validity_scores": generation_eval_dict["validity_scores"],
        "complexities": [float(c) for c in generation_eval_dict["complexities"]],
        "vocab_sizes": generation_eval_dict["vocab_sizes"],
        "num_edges": generation_eval_dict["num_edges"],
        "num_states": generation_eval_dict["num_states"],
    }

    results_path = f"/results/pfa_creativity_{model_id}_t_{temperature}_k_{top_k}_p_{top_p}_alpha_{alpha}_eq_{int(equal_entropy)}"
    results_path = results_path.replace('.', ',')
    results_path = results_path + ".json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_dict, f)
    print(f"Saved results to {results_path}")

    creativity_scores = np.array(generation_eval_dict["creativity_scores"],)
    ci_mean, ci_lower, ci_higher = bootstrapped_mean_and_ci(creativity_scores, num_samples=1000)
    if args.log_with_wandb:
        wandb.log({"creativity_score": ci_mean})
    print(f"Model {model_id} - Creativity score: {ci_mean:.2f} (+{(ci_higher-ci_mean):.2f}, -{(ci_mean-ci_lower):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PFA Generation Evaluation")
    parser.add_argument("--log_with_wandb", type=int, default=1,)
    parser.add_argument("--model_name", type=str, default="pfa_0_1B_MTP_rybl0lsn",
                        help="Model name corresponding to the checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sequences to generate for small_models_evaluation")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=18, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1., help="Top-p sampling parameter")
    parser.add_argument("--alpha", type=float, default=0., help="Alpha parameter for blending function")
    parser.add_argument("--equal_entropy", type=int, default=0, help="Equal entropy parameter")
    parser.add_argument("--generation_length", type=int, default=10, help="Length of generated sequences")
    args = parser.parse_args()

    if args.log_with_wandb:
        wandb.init(project="shortcut_PHi",
                   config=dict(vars(args)))

    pfa_creativity_evaluation(model_name=args.model_name,
                              num_samples=args.num_samples,
                              batch_size=args.batch_size,
                              temperature=args.temperature,
                              top_k=args.top_k,
                              top_p=args.top_p,
                              alpha=args.alpha,
                              equal_entropy=bool(args.equal_entropy),
                              generation_length=args.generation_length)
