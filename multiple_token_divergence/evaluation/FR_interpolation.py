import torch
import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf

def interpolate_fisher_rao(p, q, t):
    """
    Constant-speed Fisher-Rao interpolation for multivariate Gaussians 
    with diagonal covariance.

    Parameters:
    p, q : dict
        Gaussian distributions encoded as either:
        - {"p_mean": ..., "p_logvar": ...} and {"q_mean": ..., "q_logvar": ...}
        - or generic {"mean": ..., "logvar": ...}
    t : float or torch.Tensor
        Interpolation factor.
    
    Returns:
    dict with keys:
        mean, logvar, sigma
    """
    if not isinstance(p, dict) or not isinstance(q, dict):
        raise TypeError("p and q must be dicts containing mean/logvar tensors")

    p_mean = p["p_mean"] if "p_mean" in p else p["mean"]
    p_logvar = p["p_logvar"] if "p_logvar" in p else p["logvar"]
    q_mean = q["q_mean"] if "q_mean" in q else q["mean"]
    q_logvar = q["q_logvar"] if "q_logvar" in q else q["logvar"]

    mu0 = p_mean if torch.is_tensor(p_mean) else torch.as_tensor(p_mean)
    p_logvar = p_logvar if torch.is_tensor(p_logvar) else torch.as_tensor(p_logvar, dtype=mu0.dtype, device=mu0.device)
    mu1 = q_mean if torch.is_tensor(q_mean) else torch.as_tensor(q_mean, dtype=mu0.dtype, device=mu0.device)
    q_logvar = q_logvar if torch.is_tensor(q_logvar) else torch.as_tensor(q_logvar, dtype=mu0.dtype, device=mu0.device)
    p_logvar = p_logvar.to(dtype=mu0.dtype, device=mu0.device)
    mu1 = mu1.to(dtype=mu0.dtype, device=mu0.device)
    q_logvar = q_logvar.to(dtype=mu0.dtype, device=mu0.device)
    t = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=mu0.dtype, device=mu0.device)
    t = t.to(dtype=mu0.dtype, device=mu0.device)

    sigma0 = torch.exp(0.5 * p_logvar)
    sigma1 = torch.exp(0.5 * q_logvar)

    eps = torch.finfo(mu0.dtype).tiny
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=mu0.dtype, device=mu0.device))

    # 1. Map to Poincaré Upper Half-Plane coordinates (y = sqrt(2) * sigma)
    # The factor of sqrt(2) comes from the Fisher Information Metric of a Gaussian
    x0, y0 = mu0, sqrt2 * sigma0
    x1, y1 = mu1, sqrt2 * sigma1

    # 2. Solve all dimensions in parallel; vertical and arc geodesics are
    # combined with masks to avoid Python loops.
    same_mean = torch.isclose(x0, x1)
    denom = 2 * (x1 - x0)
    denom_safe = torch.where(same_mean, torch.ones_like(denom), denom)

    # Case A: Means are identical -> Path is a vertical line.
    y_vertical = torch.exp((1 - t) * torch.log(torch.clamp_min(y0, eps)) + t * torch.log(torch.clamp_min(y1, eps)))

    # Case B: General case -> Path is a semicircular arc.
    c = (x1 ** 2 - x0 ** 2 + y1 ** 2 - y0 ** 2) / denom_safe
    R = torch.sqrt(torch.clamp_min((x0 - c) ** 2 + y0 ** 2, eps))

    theta0 = torch.atan2(y0, x0 - c)
    theta1 = torch.atan2(y1, x1 - c)

    phi0 = torch.log(torch.clamp_min(torch.tan(theta0 / 2), eps))
    phi1 = torch.log(torch.clamp_min(torch.tan(theta1 / 2), eps))
    phi_t = (1 - t) * phi0 + t * phi1

    theta_t = 2 * torch.atan(torch.exp(phi_t))
    mu_arc = c + R * torch.cos(theta_t)
    y_arc = R * torch.sin(theta_t)

    mu_t = torch.where(same_mean, x0, mu_arc)
    y_t = torch.where(same_mean, y_vertical, y_arc)

    # 6. Map back to Gaussian std deviation (sigma = y / sqrt(2))
    sigma_t = y_t / sqrt2

    logvar_t = torch.log(torch.clamp_min(sigma_t ** 2, eps))

    return {
        "mean": mu_t,
        "logvar": logvar_t,
        "sigma": sigma_t,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fisher-Rao interpolation between p and q from one eval datapoint.")
    parser.add_argument("--model-name", type=str, default="sibling_discovery_shortcut_PHi_0_1B_h3kwa2lj")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--single-example", action="store_true", default=True)
    parser.add_argument("--no-single-example", dest="single_example", action="store_false")
    parser.add_argument("--t", type=float, default=0.5)
    args = parser.parse_args()

    # Mirror the setup used in phi_sibling_eval.ipynb for a single eval sample.
    project_root = None
    candidate_roots = []
    script_dir = Path(__file__).resolve().parent
    candidate_roots.extend([script_dir, *script_dir.parents])
    candidate_roots.extend([Path.cwd(), *Path.cwd().parents])

    seen = set()
    for p in candidate_roots:
        p_str = str(p)
        if p_str in seen:
            continue
        seen.add(p_str)
        if (p / "training.py").exists() and (p / "utils").exists():
            project_root = p
            break
    if project_root is None:
        raise RuntimeError("Could not locate multiple_token_divergence project root.")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from training import SelfPredictionTrainingRecipeDistributed

    model_name = args.model_name
    checkpoint_path = project_root / "checkpoints" / model_name

    cfg = OmegaConf.load(str(checkpoint_path / "config.yaml"))
    cfg.checkpointer.checkpoint_dir = str(checkpoint_path)
    cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    cfg.optimizer.lr = 0
    cfg.compile = False
    cfg.train_from_scratch = False
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"

    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    model = recipe._model.eval()

    base_dataset = recipe._dataloader.dataset
    if hasattr(base_dataset, "ds"):
        base_dataset = base_dataset.ds
    base_dataset.split = args.split
    if hasattr(base_dataset, "single_example"):
        base_dataset.single_example = args.single_example

    sample = base_dataset[args.sample_idx]
    tokens = torch.tensor(sample["tokens"], dtype=torch.long, device=recipe._device).unsqueeze(0)
    labels = torch.tensor(sample["labels"], dtype=torch.long, device=recipe._device).unsqueeze(0)

    with torch.no_grad():
        out = model(tokens=tokens)

    ignore_index = int(recipe._loss_fn.ignore_index) if hasattr(recipe._loss_fn, "ignore_index") else -100
    valid_mask = labels != ignore_index

    def flatten_valid(tensor):
        bsz, seq_len, dim = tensor.shape
        return tensor.reshape(bsz * seq_len, dim)[valid_mask.reshape(-1)]

    p_mean_v = flatten_valid(out["p_mean"])
    p_logvar_v = flatten_valid(out["p_logvar"])
    q_mean_v = flatten_valid(out["q_mean"])
    q_logvar_v = flatten_valid(out["q_logvar"])

    p_dist = {"p_mean": p_mean_v, "p_logvar": p_logvar_v}
    q_dist = {"q_mean": q_mean_v, "q_logvar": q_logvar_v}
    t = args.t
    interp = interpolate_fisher_rao(p_dist, q_dist, t)

    print("Loaded checkpoint:", str(checkpoint_path / "torchtune_model_last.pt"))
    print("split:", args.split)
    print("sample_idx:", args.sample_idx)
    print("single_example:", args.single_example)
    print("num_valid_tokens:", int(p_mean_v.shape[0]))
    print("d_model:", int(p_mean_v.shape[1]))
    print("t:", t)
    print("interpolated_mean_shape:", tuple(interp["mean"].shape))
    print("interpolated_logvar_shape:", tuple(interp["logvar"].shape))
    print("interpolated_sigma_shape:", tuple(interp["sigma"].shape))

    if interp["mean"].shape[0] > 0:
        print("\nFirst valid token (first 8 dims):")
        print("p_mean:", p_mean_v[0, :8].detach().cpu().tolist())
        print("q_mean:", q_mean_v[0, :8].detach().cpu().tolist())
        print("interp_mean:", interp["mean"][0, :8].detach().cpu().tolist())
        print("p_logvar:", p_logvar_v[0, :8].detach().cpu().tolist())
        print("q_logvar:", q_logvar_v[0, :8].detach().cpu().tolist())
        print("interp_logvar:", interp["logvar"][0, :8].detach().cpu().tolist())