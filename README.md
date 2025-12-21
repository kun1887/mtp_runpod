# Multiple Token Divergence: Measuring and Steering In-Context Computation Density

This repository contains the code and resources for the paper "Multiple Token Divergence: Measuring and Steering In-Context Computation Density". The paper introduces a novel metric, Multiple Token Divergence (MTD), to quantify the density of in-context computations performed by large language models (LLMs). We also explore methods to steer LLMs towards higher or lower computation density using MTD as a guiding metric.

## Models Trained from Scratch

To train the four different kinds of transformer models from scratch on the five different tasks, run in the `multiple-token-divergence/multiple_token_divergence` directory:

```bash
tune run --nproc_per_node 1 training.py --config configs/transformer_pfa_PHi_without_latest_embedding.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/transformer_pfa_PHi_with_latest_embedding.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/transformer_pfa_MTP_without_latest_embedding.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/transformer_pfa_MTP_with_latest_embedding.yaml
```

To train models for the four creativity tasks,
run:

```bash
tune run --nproc_per_node 1 training.py --config configs/creativity_MTP_sibling_discovery.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/creativity_MTP_triangle_discovery.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/creativity_MTP_circle_construction.yaml
```

```bash
tune run --nproc_per_node 1 training.py --config configs/creativity_MTP_triangle_construction.yaml
```


## Pre-trained Models

### Create Chains-of-Thoughts (CoTs) with Xiaomi MiMo 7B and Mistral 7B

For the Xiaomi MiMo 7B model, to create CoTs for the MATH dataset, start the sglang server:

```bash
python3 -m sglang.launch_server --model-path "XiaomiMiMo/MiMo-7B-SFT" --host 0.0.0.0 --port 30000 --trust-remote-code --dp-size 4
```

Then run the following script to generate CoTs:

```bash
python3 mimo_sglang_gen.py --dataset math --output_file _samples_MiMo_sft.json
```

```bash
python3 mimo_sglang_gen.py --dataset gsm8k --output_file _samples_MiMo_sft.json
```

For Mistral 7B

```bash
python3 -m sglang.launch_server  --model-path "mistralai/Mistral-7B-Instruct-v0.3" --port 30000 --tp-size 1 --trust-remote-code
```

```bash
python3 mimo_sglang_gen.py --dataset math --output_file _samples_Mistral_sft.json
```

```bash
python3 mimo_sglang_gen.py --dataset gsm8k --output_file _samples_Mistral_sft.json
```