import torch
import json
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import HF_CACHE_DIR

# Import your custom modules
# Ensure these are importable from the environment running this script
from mimo_mtp_evaluation.math500_generation import interpolation_blending_function
from mimo_mtp_evaluation.mimo_utils import generate_with_mtp_cached

# Hardware Settings
NUM_GPUS = 4

# Generation Parameters
MAX_NEW_TOKENS = 1500
MINIMUM_TOKENS = 750
TRIES_PER_PROMPT = 3

# Sweep Parameters (Add your sweep values here)
TEMPERATURES = [0.6, 0.7, 0.8]
ALPHAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
EQUAL_ENTROPY = False


# --- Configuration ---
model_id = "XiaomiMiMo/MiMo-7B-SFT"
prompts_file = 'creativity_task_notebooks/creative_writing_prompts_v3.json'
output_path = f"creativity_task_notebooks/results/create_writing_outputs_{model_id.split('/')[-1]}_temp_{TEMPERATURES}_alphas_{ALPHAS}_ee_{EQUAL_ENTROPY}.json"


def worker_process(gpu_id, task_queue, result_queue):
    """
    Runs on a specific GPU. Pulls tasks, generates stories, sends results back.
    """
    print(f"Worker started on GPU {gpu_id}")
    device = torch.device(f"cuda:{gpu_id}")

    # Load Model and Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=torch.bfloat16
        ).to(device)
    except Exception as e:
        print(f"Error loading model on GPU {gpu_id}: {e}")
        return

    # Constants for prompt construction
    SYSTEM_TAG = "<|start_header_id|>system<|end_header_id|>\n"
    USER_TAG = "<|start_header_id|>user<|end_header_id|>\n"
    ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>\n"
    EOT = "<|eot_id|>"
    BOS = "<|begin_of_text|>"

    while True:
        try:
            # Get task with a timeout to allow clean exit if needed
            task = task_queue.get(timeout=5)
        except Exception:
            # Queue is empty
            break

        # Unpack task
        # task structure: (prompt_id, prompt_data, attempt_num, temp, alpha)
        p_id, p_data, attempt, current_temp, current_alpha = task

        if task is None:  # Sentinel value to stop
            break

        print(f"[GPU {gpu_id}] Processing {p_id} | Attempt {attempt} | T={current_temp} | A={current_alpha}")

        instruction = p_data['writing_prompt'].replace('<SEED> ', '')

        model_prompt = (
            f"{BOS}"
            f"{SYSTEM_TAG}{'You are an expert creative writer. You write nothing but the current story. You cannot use thinking tokens or chains of thought.'}{EOT}\n"
            f"{USER_TAG}{instruction}{EOT}\n"
            f"{ASSISTANT_TAG}<think>\n\n</think>\n\n"
        )

        try:
            inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            blending_function = lambda logits, mtp_logits: interpolation_blending_function(
                logits, mtp_logits,
                top_p=1.,
                top_k=50,
                temperature=current_temp,
                alpha=current_alpha,
                equal_entropy=EQUAL_ENTROPY)

            valid_story_generated = False
            while not valid_story_generated:
                with torch.no_grad():
                    generated_ids_uncached = generate_with_mtp_cached(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=inputs,
                        custom_blending_function=blending_function,
                        max_new_tokens=MAX_NEW_TOKENS,
                        sampling_method='multinomial_direct',
                        stop_strings=[EOT, ]
                    )
                generated_tokens = generated_ids_uncached[0][input_length:]
                if len(generated_tokens) >= MINIMUM_TOKENS:
                    valid_story_generated = True
                else:
                    print(f"Generated story too short ({len(generated_tokens)} tokens). Regenerating...")
                    continue
                generated_story = tokenizer.decode(generated_tokens)
                print(generated_story[:200])


            # Pack result
            result = {
                "id": p_id,
                "story": generated_story,
                "params": {
                    "temperature": current_temp,
                    "alpha": current_alpha,
                    "attempt": attempt
                }
            }
            result_queue.put(result)

        except Exception as e:
            print(f"Error generating on GPU {gpu_id} for prompt {p_id}: {e}")

    print(f"Worker on GPU {gpu_id} finished.")


def writer_process(output_file, result_queue, total_tasks):
    """
    Listens for results and saves them to JSON incrementally.
    """
    results_storage = {}

    # Load existing if available to resume (optional, basic logic here)
    if Path(output_file).exists():
        try:
            with open(output_file, 'r') as f:
                results_storage = json.load(f)
        except:
            pass

    completed_count = 0

    while completed_count < total_tasks:
        result = result_queue.get()

        p_id = result['id']
        story = result['story']
        params = result['params']

        # Initialize structure if new prompt
        if p_id not in results_storage:
            results_storage[p_id] = []

        # Append result with metadata
        results_storage[p_id].append({
            "story": story,
            "parameters": params
        })

        completed_count += 1

        # Save to disk
        with open(output_file, 'w') as f:
            json.dump(results_storage, f, indent=4)

        if completed_count % 10 == 0:
            print(f"Progress: {completed_count}/{total_tasks} stories generated and saved.")

    print("Writer process finished. All tasks saved.")


if __name__ == '__main__':
    # Required for PyTorch multiprocessing
    mp.set_start_method('spawn', force=True)

    # 1. Load Prompts
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)

    # 2. Build Task Queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    tasks_list = []

    # Create the Cartesian product of tasks
    # Prompt ID * Temperatures * Alphas * Attempts
    for key, prompt in prompts_data.items():
        for temp in TEMPERATURES:
            for alpha in ALPHAS:
                for attempt in range(TRIES_PER_PROMPT):
                    # Tuple: (id, data, attempt_num, temp, alpha)
                    task_queue.put((key, prompt, attempt + 1, temp, alpha))
                    tasks_list.append(key)

    total_tasks = len(tasks_list)
    print(f"Total tasks scheduled: {total_tasks}")

    # 3. Start Writer Process
    writer = mp.Process(target=writer_process, args=(output_path, result_queue, total_tasks))
    writer.start()

    # 4. Start Worker Processes
    workers = []
    # Determine how many GPUs strictly available or requested
    actual_gpus = min(torch.cuda.device_count(), NUM_GPUS)

    for i in range(actual_gpus):
        p = mp.Process(target=worker_process, args=(i, task_queue, result_queue))
        p.start()
        workers.append(p)

    # 5. Join Workers
    for p in workers:
        p.join()

    # 6. Join Writer
    writer.join()

    print("Done.")