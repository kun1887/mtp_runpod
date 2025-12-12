import os
import json
import anthropic
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar library

# --- CONFIGURATION ---
# Run in terminal: export ANTHROPIC_API_KEY="your_new_key_here"
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Using a currently valid model name (Change back if you have specific beta access)
MODEL_NAME = "claude-sonnet-4-5-20250929"
MAX_STORY_CHARACTERS = 4000
NUM_WORKERS = 10  # Number of parallel requests

# File Lock to prevent write conflicts when saving incrementally
save_lock = threading.Lock()

# Load prompts
with open('creative_writing_judging_prompt.txt', 'r') as f:
    prompt_content = f.read()
SYSTEM_PROMPT = prompt_content.split("\n\n")[0].strip()
USER_PROMPT_TEMPLATE = "[PROMPT START]" + prompt_content.split("[PROMPT START]")[-1]


def get_claude_evaluation(client, writing_prompt, story_text):
    """Sends the story to Claude for evaluation."""
    user_content = USER_PROMPT_TEMPLATE.format(
        writing_prompt=writing_prompt,
        test_model_response=story_text
    )

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,  # Adjusted to standard max output
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        return message.content[0].text
    except anthropic.APIError as e:
        print(f"\nAPI Error: {e}")
        return None
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        return None


def process_story_task(client, prompt_id, story_index, writing_prompt, story_data):
    """
    Worker function to process a single story.
    Returns the result along with indices to update the main data structure.
    """
    # Check if already evaluated to allow resuming interrupted runs
    if 'evaluation' in story_data and story_data['evaluation']:
        return None  # Skip

    story_text = story_data['story'][:MAX_STORY_CHARACTERS]

    evaluation = get_claude_evaluation(client, writing_prompt, story_text)

    return {
        "prompt_id": prompt_id,
        "story_index": story_index,
        "evaluation": evaluation,
        "params": story_data['parameters']
    }


def main(generated_stories_file, prompts_file, output_file):
    if not API_KEY:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=API_KEY)

    # Load Data
    with open(generated_stories_file, 'r') as f:
        generated_stories_data = json.load(f)

    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)

    # 1. Prepare a flat list of tasks
    # We flatten the structure so the workers can chew through a single queue
    tasks = []
    print("Preparing tasks...")
    for prompt_id, stories in generated_stories_data.items():
        if prompt_id not in prompts_data:
            continue

        writing_prompt = prompts_data[prompt_id]['writing_prompt'].replace('<SEED> ', '')

        for idx, story in enumerate(stories):
            # Pass the reference to the story object and indices
            tasks.append({
                "prompt_id": prompt_id,
                "story_index": idx,
                "writing_prompt": writing_prompt,
                "story_data": story
            })

    print(f"Total stories to evaluate: {len(tasks)}")

    # 2. Process in Parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                process_story_task,
                client,
                t['prompt_id'],
                t['story_index'],
                t['writing_prompt'],
                t['story_data']
            ): t for t in tasks
        }

        # Process results as they complete (using tqdm for progress bar)
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating"):
            result = future.result()

            if result:
                # Update the main data structure in memory
                p_id = result['prompt_id']
                s_idx = result['story_index']
                eval_text = result['evaluation']

                generated_stories_data[p_id][s_idx]['evaluation'] = eval_text

                # Optional: Print snippet of result
                # param_info = result['params']
                # print(f" Completed: Temp {param_info.get('temperature')} | {eval_text[-20:] if eval_text else 'Error'}")

                # Save Checkpoint (Thread-safe)
                # We save every 5 completions or so to avoid hammering the disk,
                # or just save here. For safety, we save every time but use a lock.
                with save_lock:
                    with open(output_file, 'w') as f:
                        json.dump(generated_stories_data, f, indent=4)

    print(f"Done! Results saved to {output_file}")


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)

    generated_stories_file = "results/create_writing_outputs_MiMo-7B-SFT_temp_[0.6, 0.7, 0.8]_alphas_[0.1]_ee_False.json"
    prompts_file = 'creative_writing_prompts_v3.json'
    output_file = generated_stories_file.replace('outputs', 'evaluation')

    # Only run if input files exist to prevent immediate crash
    if os.path.exists(generated_stories_file) and os.path.exists(prompts_file):
        main(generated_stories_file, prompts_file, output_file)
    else:
        print(f"Error: Could not find input files.\nChecked: {generated_stories_file}\nChecked: {prompts_file}")