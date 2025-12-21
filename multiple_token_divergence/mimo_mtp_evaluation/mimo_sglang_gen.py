import json
import asyncio
import aiohttp
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import argparse

from mimo_mtp_evaluation.math500_generation import extract_math_ans_from_response


MATH_GENERAL_PROMPT = """Answer the following math question, given in LaTeX format, clearly and concisely, and present the final answer as \(\\boxed{x}\), where x is the fully simplified solution.

Example:
**Question:** \(\int_0^1 (3x^2 + 2x) \,dx\)
**Solution:** \(\int (3x^2 + 2x) \,dx = x^3 + x^2 + C\) Evaluating from 0 to 1: \((1^3 + 1^2) - (0^3 + 0^2) = 1 + 1 - 0 = 2 \\boxed{2}\)

Now, solve the following question:
"""

GSM8K_GENERAL_PROMPT = """\nLet's think step by step. At the end, you must write the answer as an integer inside '\\boxed{}'.\n"""


async def generate_samples(session, prompt, server_url, num_samples=10, max_tokens=4096, temperature=0.7):
    """Generate multiple samples for a single prompt using sglang HTTP API"""
    url = f"{server_url}/generate"

    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_k": 50,
            "top_p": 1.0,
            "n": num_samples,
            "stop": ["<|eot_id|>", "</s>", "<|end_of_text|>", "\n<|start_header_id|>user<|end_header_id|>", "<|im_end|>"]
        }
    }

    try:
        async with session.post(url, json=payload, timeout=300) as response:
            if response.status == 200:
                result = await response.json()
                # print(f"DEBUG: Response type: {type(result)}")  # Debug line
                # print(f"DEBUG: Response content: {result}")     # Debug line

                # Handle different response formats
                if isinstance(result, list):
                    # Server returned a list of responses
                    return result
                elif isinstance(result, dict) and "token_ids_list" in result:
                    return result.get("token_ids_list", [])
                elif isinstance(result, dict) and "text" in result:
                    # Some servers return text directly
                    return [result.get("text", "")]
                else:
                    print(f"Unexpected response format: {result}")
                    return []
            else:
                print(f"Error: HTTP {response.status} for prompt")
                return []
    except Exception as e:
        print(f"Exception during generation: {e}")
        return []


async def process_problem(session, problem_data, server_url, num_samples=10):
    """Process a single problem and generate multiple samples"""
    i, row = problem_data
    is_math = "problem" in row

    if is_math:
        prompt = MATH_GENERAL_PROMPT + row["problem"]
        kwargs = {
            "index": i,
            "problem": row["problem"],
            "level": row["level"],
            "levelint": row["levelint"],
            "type": row["type"],
            "unique_id": row["unique_id"]
        }
    else:
        prompt = row["question"] + GSM8K_GENERAL_PROMPT
        kwargs = {
            "index": i,
            "question": row["question"],
            "answer": row["answer"],
        }

    messages = [{"content": ""}, {"content": prompt}]

    # Format the prompt (you might need to adjust this based on your tokenizer's chat template)
    prompt = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"

    # Generate samples
    samples_token_ids = await generate_samples(session, prompt, server_url, num_samples)

    return {
        "index": i,
        "prompt": prompt,
        "samples_token_ids": samples_token_ids,  # List of token ID lists
        "num_samples": len(samples_token_ids),
        **kwargs
    }


async def process_problem_llama(session, problem_data, server_url, num_samples=10):
    """Process a single problem and generate multiple samples"""
    i, row = problem_data
    is_math = "problem" in row

    if is_math:
        # We put the general instructions into the system role
        system_content = MATH_GENERAL_PROMPT
        user_content = row["problem"]
        kwargs = {
            "index": i,
            "problem": row["problem"],
            "level": row["level"],
            "levelint": row["levelint"],
            "type": row["type"],
            "unique_id": row["unique_id"]
        }
    else:
        # For GSM8K, combine the general prompt with the question in the user role
        system_content = "You are a helpful assistant specialized in solving math problems."
        user_content = row["question"] + GSM8K_GENERAL_PROMPT
        kwargs = {
            "index": i,
            "question": row["question"],
            "answer": row["answer"],
        }

        # --- FIX: New Llama 3.2 Prompt Format ---
    SYSTEM_TAG = "<|start_header_id|>system<|end_header_id|>\n"
    USER_TAG = "<|start_header_id|>user<|end_header_id|>\n"
    ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>\n"
    EOT = "<|eot_id|>"
    BOS = "<|begin_of_text|>"

    prompt = (
        f"{BOS}"
        f"{SYSTEM_TAG}{system_content}{EOT}\n"
        f"{USER_TAG}{user_content}{EOT}\n"
        f"{ASSISTANT_TAG}"
    )
    # Generate samples
    samples_token_ids = await generate_samples(session, prompt, server_url, num_samples)

    return {
        "index": i,
        "prompt": prompt,
        "samples_token_ids": samples_token_ids,  # List of token ID lists
        "num_samples": len(samples_token_ids),
        **kwargs
    }


async def process_problem_mistral(session, problem_data, server_url, num_samples=10):
    """Process a single problem and generate multiple samples using Mistral's Chat Template"""
    i, row = problem_data
    is_math = "problem" in row

    if is_math:
        # For MATH, use the general instructions as the system content
        system_content = MATH_GENERAL_PROMPT
        user_content = row["problem"]
        gt_solution = extract_math_ans_from_response(row["solution"])
        kwargs = {
            "index": i,
            "problem": row["problem"],
            "level": row["level"],
            "levelint": row["levelint"],
            "type": row["type"],
            "unique_id": row["unique_id"],
            "gt_solution": gt_solution
        }
    else:
        # For GSM8K, combine the general prompt with the question in the user role
        # Mistral generally relies on the prompt itself rather than a dedicated system role
        # We put the instruction directly into the user message for a stronger effect
        system_content = "You are a helpful assistant specialized in solving math problems."
        user_content = row["question"] + "\n\n" + GSM8K_GENERAL_PROMPT
        kwargs = {
            "index": i,
            "question": row["question"],
            "answer": row["answer"],
        }

    # --- MISTRAL CHAT TEMPLATE ADAPTATION ---
    # Note: Mistral's base template doesn't explicitly use a <|begin_of_text|> equivalent
    # and typically concatenates messages using <s> and </s> tokens.

    # 1. Define the Mistral Chat Tokens (Based on Hugging Face's ChatML template)
    BOS_TOKEN = "<s>"  # Beginning of sequence
    EOS_TOKEN = "</s>"  # End of sequence
    INST_BEGIN = "[INST]"  # Instruction block start
    INST_END = "[/INST]"  # Instruction block end

    # 2. Construct the Mistral Prompt String
    # Mistral Instruction format: <s>[INST] Instruction [/INST] Model Response

    # We combine the system and user content into a single instruction block
    # Note: Mistral often ignores a separate system tag, so we prepend the role.
    full_instruction = f"{system_content}\n\n{user_content}"

    prompt = (
        f"{BOS_TOKEN}"
        f"{INST_BEGIN} {full_instruction}{INST_END}"  # Instruction block
        # The model is now expected to generate the response after INST_END
    )

    # Generate samples
    samples_token_ids = await generate_samples(session, prompt, server_url, num_samples)

    if is_math:
        for candidate in samples_token_ids:
            solution_text = candidate['text']
            extracted_solution = extract_math_ans_from_response(solution_text)
            candidate['extracted_solution'] = extracted_solution
            candidate['correct'] = (extracted_solution == kwargs['gt_solution'])

    return {
        "index": i,
        "prompt": prompt,
        "samples_token_ids": samples_token_ids,  # List of token ID lists
        "num_samples": len(samples_token_ids),
        **kwargs
    }


async def process_batch(session, batch, server_url, num_samples, semaphore, results_list):
    """Process a batch of problems with rate limiting"""
    async with semaphore:
        tasks = []
        for problem_data in batch:
            task = asyncio.create_task(process_problem_mistral(session, problem_data, server_url, num_samples))
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in tqdm(batch_results):
            if isinstance(result, Exception):
                print(f"Error processing problem: {result}")
            else:
                results_list.append(result)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and other non-serializable types"""

    def default(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'tolist'):  # Handle numpy arrays
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        else:
            return str(obj)  # Convert other types to string


def save_results_json(results, output_path):
    """Save results to JSON file with proper encoding"""
    # Convert results to JSON-serializable format
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if hasattr(value, 'tolist'):  # Convert numpy arrays to lists
                serializable_result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serializable_result[key] = [item.tolist() if hasattr(item, 'tolist') else item for item in value]
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, cls=JSONEncoder)


async def main():
    parser = argparse.ArgumentParser(description="Generate MATH500 samples using sglang HTTP server")
    parser.add_argument("--dataset", type=str, default="math", help="One of the allowed datasets")
    parser.add_argument("--output_file", type=str, default="_samples.json", help="Output JSON file path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of concurrent requests")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_workers", type=int, default=10, help="Max concurrent workers")
    parser.add_argument("--host", type=str, default="localhost", help="Server host/IP address")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    args = parser.parse_args()

    dataset_paths = {
        "math": "../data/MATH/test",
        "math500": "../data/MATH500/test",
        "gsm8k": "../data/grade-school-math/grade_school_math/data"
    }
    args.dataset_path = dataset_paths[args.dataset]
    args.output_file = f"{args.dataset}{args.output_file}"

    # Construct server URL
    server_url = f"http://{args.host}:{args.port}"
    print(f"Using server URL: {server_url}")

    # Load dataset
    dataset_path = Path(args.dataset_path)
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "gsm8k":
        df = pd.read_json(dataset_path / "test.jsonl", lines=True)
    else:
        filelist = []
        for file in dataset_path.rglob("*.json"):
            with open(file, "r") as f:
                ctx = json.loads(f.read())
                ctx['unique_id'] = str(file).split('MATH500/')[-1]
                filelist.append(ctx)

        df = pd.DataFrame.from_records(filelist)
        df["levelint"] = [int(x.split()[-1]) for x in df["level"]]

    print(f"Loaded {len(df)} problems.")
    # Prepare problem data
    problem_data_list = [(i, row.to_dict()) for i, row in df.iterrows()]

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = []

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.max_workers)

    async with aiohttp.ClientSession() as session:
        # Process in batches
        total_batches = (len(problem_data_list) + args.batch_size - 1) // args.batch_size

        for batch_start in tqdm(range(0, len(problem_data_list), args.batch_size),
                                desc="Processing batches", total=total_batches):
            batch_end = min(batch_start + args.batch_size, len(problem_data_list))
            batch = problem_data_list[batch_start:batch_end]

            await process_batch(session, batch, server_url, args.num_samples, semaphore, all_results)

            # calculate accuracy
            num_correct = 0
            num_total = 0
            for result in all_results:
                if 'gt_solution' in result:
                    for candidate in result['samples_token_ids']:
                        if candidate.get('correct', False):
                            num_correct += 1
                    num_total += len(result['samples_token_ids'])
            if num_total > 0:
                accuracy = num_correct / num_total * 100
                print(f"Current accuracy: {accuracy:.2f}% ({num_correct}/{num_total})")

            # Save intermediate results after each batch
            save_results_json(all_results, output_path)

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(1)

    # Final save
    save_results_json(all_results, output_path)

    print(f"Saved {len(all_results)} problems with samples to {output_path}")

    # Print summary
    total_samples = sum(len(result['samples_token_ids']) for result in all_results)
    print(f"Generated {total_samples} total samples across {len(all_results)} problems")


if __name__ == "__main__":
    asyncio.run(main())


"""
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.5.2rc1"

python3 -m sglang.launch_server \
  --model-path "XiaomiMiMo/MiMo-7B-SFT" \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --dp-size 4
  
## 1 query (alive)
curl -X POST http://127.0.0.1:30000/generate   -H "Content-Type: application/json"   -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10, "output_token_ids": true}}'

## gen answers
python3 mimo_sglang_gen.py --dataset math500 --output_file _samples_MiMo_sft.json
"""
