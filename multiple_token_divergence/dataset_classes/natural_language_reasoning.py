import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
import json
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._messages import Message
from torchtune.data._utils import truncate
from torchtune.datasets import SFTDataset, text_completion_dataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from mimo_mtp_evaluation.mimo_sglang_gen import MATH_GENERAL_PROMPT, GSM8K_GENERAL_PROMPT

from dataset_classes import messages_pfa_dataset
from config import HF_CACHE_DIR


class Gsm8kToMessages(Transform):
    """
    Transforms a GSM8K sample into a structured list of messages for chat-based models.

    This transform handles the construction of few-shot prompts, the main question
    (optionally with an added instruction), and the assistant's response based on
    the specified mode ('train', 'test', 'without_answer').

    In 'train' mode, the full answer is included. In 'without_answer' mode, only
    the rationale part of the answer (up to '####') is included. In 'test' mode,
    an empty assistant message is added to prompt the model for a response.

    Args:
        mode (str, optional): Defines how the assistant's answer is processed.
            Must be one of 'train', 'test', or 'without_answer'. Defaults to 'train'.
        few_shot_prompts (Optional[dict], optional): A dictionary or list-like object
            containing few-shot examples, where each example has "question" and "answer" keys.
            Defaults to None.
        num_few_shot_prompts (int, optional): The number of few-shot examples to
            randomly select and prepend to the main question. Defaults to 0.
        prompt_addition (Optional[str], optional): A string to append to the main
            question, often used for chain-of-thought prompting.
            Defaults to "Let's think step by step. At the end, you must write the answer as an integer after '####'.".
    """
    def __init__(self,
                 mode='train',
                 few_shot_prompts: Optional[dict] = None,
                 num_few_shot_prompts: int = 0,
                 prompt_addition: Optional[str] = "Let's think step by step. At the end, you must write the answer as an integer after '####'."):
        super().__init__()
        assert mode in ['train', 'test', 'without_answer']
        self.mode = mode
        self.few_shot_prompt = few_shot_prompts
        self.num_few_shot_prompts = num_few_shot_prompts
        self.prompt_addition = prompt_addition

        if self.few_shot_prompt is not None:
            assert len(self.few_shot_prompt) >= self.num_few_shot_prompts, "Number of few shot prompts should be less than or equal to the number of few shot prompts provided."

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input GSM8K sample.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have
                'question' and 'answer' keys.

        Returns:
            Mapping[str, Any]: A dictionary containing 'messages' (List[Message]).
                If mode is not 'train', it also includes the original 'answer'.
        """
        messages = []
        # Few shot prompt
        if self.few_shot_prompt is not None:
            # randomly select few shot prompts
            if len(self.few_shot_prompt) == self.num_few_shot_prompts:
                perm = range(self.num_few_shot_prompts)
            else:
                perm = torch.randperm(len(self.few_shot_prompt))[0:self.num_few_shot_prompts]
            for i in perm:
                messages.append(Message(
                    role="user",
                    content=self.few_shot_prompt[i]["question"],
                    masked=True,
                    eot=True,
                ))
                messages.append(Message(
                    role="assistant",
                    content=self.few_shot_prompt[i]["answer"],
                    masked=True,
                    eot=True,
                ))

        # Question
        if self.prompt_addition is not None:
            question = f"{sample['question']} {self.prompt_addition}"
        else:
            question = sample["question"]
        messages.append(Message(
                role="user",
                content=question,
                masked=True,
                eot=True,
            ))

        # Answer / Answer Fragment
        if self.mode == 'train':
            messages.append(Message(
                    role="assistant",
                    content=sample["answer"],
                    masked=False,
                    eot=True,
                ))
        elif self.mode == 'without_answer':
            rationale = sample["answer"]
            rationale = rationale.split('####')[0] + '####'
            messages.append(Message(
                role="assistant",
                content=rationale,
                masked=False,
                eot=False,
            ))
        elif self.mode == 'test':
            messages.append(Message(
                role="assistant",
                content="",
                masked=False,
                eot=False,
            ))

        if not self.mode == 'train':
            return {"messages": messages, "answer": sample["answer"]}
        return {"messages": messages}


class Gsm8kToTokens(Transform):
    """
    Tokenizes a GSM8K sample (typically output from Gsm8kToMessages) and extracts
    the final numerical answer if present.

    This transform applies the provided tokenizer to the input sample. If the sample
    contains an "answer" key (string form), it extracts the numerical part after
    "####", cleans it by removing common non-numeric characters (commas, currency
    symbols, etc.), and converts it to an integer.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use.
    """
    def __init__(self, tokenizer: ModelTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenizes the sample and processes the answer.

        Args:
            sample (Mapping[str, Any]): The input data, expected to be
                compatible with the tokenizer. May contain an "answer" string.

        Returns:
            Mapping[str, Any]: The dictionary returned by the tokenizer, potentially
                augmented with a cleaned, integer "answer" key.
        """
        return_dict = self.tokenizer(sample)
        if "answer" in sample:
            answer = sample["answer"]
            answer = answer.split('####')[-1].strip()
            for remove_char in [',', '$', '%', 'g']:
                answer = answer.replace(remove_char, '')
            return_dict["answer"] = int(answer)
        return return_dict


def gsm_8k_dataset(tokenizer,
                   file,
                   few_shot_prompts=None,
                   num_few_shot_prompts=0,
                   mode='train',
                   prompt_addition="Let's think step by step. At the end, you must write the answer as an integer after '####'."):
    """
    Factory function to create a GSM8K dataset for supervised fine-tuning (SFT).

    This function configures an `SFTDataset` using GSM8K data from a specified file.
    It applies `Gsm8kToMessages` to format the data into a message-based structure
    (optionally including few-shot examples) and then `Gsm8kToTokens` to tokenize
    these messages and process the numerical answer.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for processing text.
        file (str): Path to the JSONL or JSON file containing the GSM8K data.
        few_shot_prompts (Optional[Mapping[int, Dict[str, str]]], optional):
            Few-shot examples to prepend. Each example should be a dictionary
            with "question" and "answer" keys. Defaults to None.
        num_few_shot_prompts (int, optional): Number of few-shot examples to use.
            Defaults to 0.
        mode (str, optional): Operating mode ('train', 'test', 'without_answer')
            for `Gsm8kToMessages`. Defaults to 'train'.
        prompt_addition (Optional[str], optional): Additional text appended to questions.
            Defaults to a chain-of-thought style prompt.

    Returns:
        SFTDataset: An instance of `SFTDataset` configured for GSM8K.
    """
    message_transform = Gsm8kToMessages(few_shot_prompts=few_shot_prompts,
                                        num_few_shot_prompts=num_few_shot_prompts,
                                        mode=mode,
                                        prompt_addition=prompt_addition)
    load_dataset_kwargs = {"data_files": [file]}
    dataset = SFTDataset(
        source='json',
        message_transform=message_transform,
        model_transform=Gsm8kToTokens(tokenizer),
        split='train',
        **load_dataset_kwargs,
    )
    return dataset


class MathToMessages(Transform):
    """
    Transforms a MATH dataset sample into a structured list of messages.

    This transform handles the construction of few-shot prompts, the main problem
    (with an optional instruction), and the assistant's response based on the
    specified mode. It also processes the 'level' and 'type' of the problem,
    converting them into numerical formats.

    Args:
        mode (str, optional): Defines how the assistant's solution is handled.
            Must be one of 'train', 'test', or 'without_answer'. Defaults to 'train'.
        few_shot_prompts (Optional[dict], optional): A dictionary of few-shot
            examples, each with "problem" and "solution" keys. Defaults to None.
        num_few_shot_prompts (int, optional): The number of few-shot examples to
            randomly select. Defaults to 0.
        prompt_addition (Optional[str], optional): A string to append to the
            problem, typically for chain-of-thought prompting. Defaults to a
            prompt asking for a LaTeX solution box.
    """
    def __init__(self,
                 mode='train',
                 few_shot_prompts: Optional[dict] = None,
                 num_few_shot_prompts: int = 0,
                 prompt_addition: Optional[str] = "Let's think step by step. At the end, present your solution in a LaTeX box (i.e., $\\boxed{SOLUTION}$)."):
        super().__init__()
        assert mode in ['train', 'test', 'without_answer']
        self.mode = mode
        self.few_shot_prompt = few_shot_prompts
        self.num_few_shot_prompts = num_few_shot_prompts
        self.prompt_addition = prompt_addition
        self.type_lookup = {
            "Algebra": 0,
            "Counting & Probability": 1,
            "Geometry": 2,
            "Intermediate Algebra": 3,
            "Number Theory": 4,
            "Prealgebra": 5,
            "Precalculus": 6,
        }

        if self.few_shot_prompt is not None:
            assert len(self.few_shot_prompt) >= self.num_few_shot_prompts, "Number of few shot prompts should be less than or equal to the number of few shot prompts provided."

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input MATH sample to create messages and metadata.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have keys
                'problem', 'solution', 'level', and 'type'.

        Returns:
            Mapping[str, Any]: A dictionary with 'messages' (List[Message]),
                'level' (int), 'type' (int), and optionally the original 'solution' string.
        """
        messages = []
        if self.few_shot_prompt is not None:
            # randomly select few shot prompts
            perm = torch.randperm(len(self.few_shot_prompt))[0:self.num_few_shot_prompts]
            for i in perm:
                messages.append(Message(
                    role="user",
                    content=self.few_shot_prompt[i]["problem"],
                    masked=True,
                    eot=True,
                ))
                messages.append(Message(
                    role="assistant",
                    content=self.few_shot_prompt[i]["solution"],
                    masked=True,
                    eot=True,
                ))
        if self.prompt_addition is not None:
            question = f"{sample['problem']} {self.prompt_addition}"
        else:
            question = sample["problem"]

        messages.append(Message(
                role="user",
                content=question,
                masked=True,
                eot=True,
            ))

        if self.mode == 'train':
            messages.append(Message(
                    role="assistant",
                    content=sample["solution"],
                    masked=False,
                    eot=True,
                ))
        elif self.mode == 'without_answer':
            rationale = sample["solution"]
            rationale = rationale.split('boxed{')[0] + 'boxed{'
            messages.append(Message(
                role="assistant",
                content=rationale,
                masked=False,
                eot=False,
            ))
        elif self.mode == 'test':
            messages.append(Message(
                role="assistant",
                content='',
                masked=False,
                eot=False,
            ))

        level = sample["level"].split('Level ')[-1]
        try:
            level = int(level)
        except ValueError:
            print(f"Error in level: {sample['level']}")
            level = 0
            pass

        return_dict = {"messages": messages,
                       "level": level,
                       "type": self.type_lookup[sample["type"]]}
        if not self.mode == 'train':
            return_dict["solution"] = sample["solution"]
        return return_dict


class MathToTokens(Transform):
    """
    Tokenizes a MATH sample and preserves its metadata.

    This transform applies a tokenizer to a sample (typically the output of
    `MathToMessages`) and ensures that the 'level', 'type', and original
    'solution' string are carried over to the final tokenized output.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use.
    """
    def __init__(self, tokenizer: ModelTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenizes the sample and retains metadata.

        Args:
            sample (Mapping[str, Any]): The input data, expected to be
                compatible with the tokenizer and contain 'level', 'type',
                and optionally 'solution' keys.

        Returns:
            Mapping[str, Any]: The dictionary returned by the tokenizer,
                augmented with the 'level', 'type', and 'solution' from the input.
        """
        return_dict = self.tokenizer(sample)
        return_dict["level"] = sample["level"]
        return_dict["type"] = sample["type"]
        if "solution" in sample:
            solution = sample["solution"]
            # solution = solution.split('boxed{')[-1].split('}$')[0].strip()
            return_dict["solution"] = solution
        return return_dict


def math_dataset(tokenizer,
                 location,
                 few_shot_prompts=None,
                 num_few_shot_prompts=0,
                 mode='train',
                 prompt_addition="Let's think step by step. At the end, present your solution in a LaTeX box (i.e., $\\boxed{SOLUTION}$)."):
    """
    Factory function to create a dataset for the MATH dataset.

    This function recursively finds all JSON files in a given directory, then
    configures and returns an `SFTDataset`. It uses `MathToMessages` and
    `MathToTokens` to transform the raw data into a tokenized format suitable
    for training or small_models_evaluation.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for processing text.
        location (str): The root directory to search recursively for `.json` files.
        few_shot_prompts (Optional[dict], optional): Few-shot examples to prepend
            to prompts. Defaults to None.
        num_few_shot_prompts (int, optional): Number of few-shot examples to use.
            Defaults to 0.
        mode (str, optional): Operating mode ('train', 'test', 'without_answer') for
            `MathToMessages`. Defaults to 'train'.
        prompt_addition (Optional[str], optional): Additional text appended to problems.
            Defaults to a chain-of-thought style prompt.

    Returns:
        SFTDataset: An instance of `SFTDataset` configured for the MATH dataset.
    """
    files = []
    for root, _, filenames in os.walk(location):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))

    message_transform = MathToMessages(few_shot_prompts=few_shot_prompts,
                                       num_few_shot_prompts=num_few_shot_prompts,
                                       mode=mode,
                                       prompt_addition=prompt_addition)
    load_dataset_kwargs = {"data_files": files}
    dataset = SFTDataset(
        source='json',
        message_transform=message_transform,
        model_transform=MathToTokens(tokenizer),
        split='train',
        **load_dataset_kwargs,
    )
    return dataset


class CombinedDataset(torch.utils.data.Dataset):
    """
    A dataset that interleaves samples from multiple source datasets.
    """

    def __init__(self, datasets: List[torch.utils.data.Dataset], relative_weights: Optional[List[int]] = None):
        self.datasets = datasets
        self.relative_weights = relative_weights if relative_weights is not None else [1] * len(datasets)
        self.total_weight = sum(self.relative_weights)

        self.mod_idx_to_dataset_idx = []
        # This list will store the "local offset" for the dataset at that specific modulo step
        self.cycle_offsets = []

        # We use a temporary counter to track how many times we've seen a dataset within ONE cycle
        temp_counts = {i: 0 for i in range(len(datasets))}

        for dataset_idx, weight in enumerate(self.relative_weights):
            for _ in range(weight):
                self.mod_idx_to_dataset_idx.append(dataset_idx)
                self.cycle_offsets.append(temp_counts[dataset_idx])
                temp_counts[dataset_idx] += 1

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        # 1. Determine which cycle of the weights we are in
        cycle_itr = index // self.total_weight

        # 2. Determine where we are inside the current cycle
        mod_idx = index % self.total_weight

        dataset_index = self.mod_idx_to_dataset_idx[mod_idx]

        # 3. Calculate correct index:
        # (Number of full cycles * items per cycle) + (Items seen so far in current cycle)
        index_in_dataset = (cycle_itr * self.relative_weights[dataset_index]) + self.cycle_offsets[mod_idx]

        #print(f"DatasetIdx: {dataset_index}, LocalIdx: {index_in_dataset}")

        # Safety: Wrap around if the calculated index exceeds the actual dataset size
        # This allows for weights that don't perfectly match the dataset length ratios
        index_in_dataset = index_in_dataset % len(self.datasets[dataset_index])

        full_sample = self.datasets[dataset_index][index_in_dataset]

        sample = {
            "tokens": full_sample["tokens"],
            "labels": full_sample["labels"],
        }

        # print(f"Fetched from Dataset {dataset_index}, Local Index {index_in_dataset} (Global {index})")

        for key, value in sample.items():
            if not isinstance(value, list):
                sample[key] = [value] * len(sample["tokens"])

        return sample


def fineweb_dataset(tokenizer):
    dataset = text_completion_dataset(
        tokenizer=tokenizer,
        source="HuggingFaceFW/fineweb",
        column="text",
        split="train",
        #streaming=True,
        name="sample-10BT",
        cache_dir=HF_CACHE_DIR,
    )
    return dataset

class TinyStoriesJsonDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.bos_id
        self.eos_token = self.tokenizer.eos_id
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        d = [self.bos_token] + d
        return {"tokens": d[:-1],
                "labels": d[1:]}


def tinystories_json_dataset(tokenizer,
                         location):
    dataset = TinyStoriesJsonDataset(location, tokenizer)
    return dataset



def combined_reasoning_and_icl_dataset(tokenizer,
                                       math_location=None,
                                       gsm_location=None):
    """
    Creates a combined dataset for reasoning and in-context learning tasks.

    This function builds and combines four distinct datasets:
    1. MATH dataset for mathematical reasoning.
    2. GSM8K dataset for grade-school math problems.
    3. PFA (Probabilistic Finite Automata) dataset for in-context learning.
    4. SlimPajama for general natural language.

    The final output is a `CombinedDataset` instance that interleaves samples
    from these sources.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to use for all sub-datasets.
        math_location (Optional[str], optional): Path to the MATH dataset's
            training directory. Defaults to a relative path.
        gsm_location (Optional[str], optional): Path to the GSM8K training
            data file. Defaults to a relative path.

    Returns:
        CombinedDataset: A dataset that combines the four specified sources.
    """
    if math_location is None:
        math_location = "../data/MATH/train"
    if gsm_location is None:
        gsm_location = "../data/grade-school-math/grade_school_math/data/train.jsonl"
    math_sft_dataset = math_dataset(tokenizer, math_location, mode='train')
    gsm_8k_sft_dataset = gsm_8k_dataset(tokenizer, gsm_location, mode='train')
    pfa_sft_dataset = messages_pfa_dataset(tokenizer)
    pajama = slim_pajama_dataset(tokenizer)
    combined_dataset = CombinedDataset([math_sft_dataset, gsm_8k_sft_dataset, pfa_sft_dataset, pajama])
    return combined_dataset


class MATHMistralDataset(Dataset):
    def __init__(self, math_dataset_location, tokenizer):
        # Load the MATH dataset files
        math_files = []
        for root, dirs, files in os.walk(math_dataset_location):
            for file in files:
                if file.endswith('.json'):
                    math_files.append(os.path.join(root, file))
        items = []
        for file_path in math_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                id = file_path.split('MATH/')[-1]
                items.append({
                    'id': id,
                    'level': data['level'],
                    'type': data['type'],
                    'problem': data['problem'],
                    'solution': data['solution'],
                })

        self.items = items
        self.tokenizer = tokenizer
        self.BOS_TOKEN = "<s>"  # Beginning of sequence
        self.EOS_TOKEN = "</s>"  # End of sequence
        self.INST_BEGIN = "[INST]"  # Instruction block start
        self.INST_END = "[/INST]"  # Instruction block end

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        full_instruction = f"{MATH_GENERAL_PROMPT}\n\n{item['problem']}"

        prompt = (
            f"{self.BOS_TOKEN}"
            f"{self.INST_BEGIN} {full_instruction}{self.INST_END}"  # Instruction block
        )

        prompt_tokens = self.tokenizer.encode(prompt)[:-1]
        solution_tokens = self.tokenizer.encode(item['solution'])[2:]
        #print(f"Prompt tokens: {prompt_tokens}, \n Solution tokens: {solution_tokens}")
        tokens = prompt_tokens + solution_tokens
        tokens = tokens[:2048]
        mask = [False] * len(tokens)
        # mask[:len(prompt_tokens) + 7] = [True] * (len(prompt_tokens) + 7)
        labels = tokens[1:] + [ -100 ]
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
        #print(f"idx: {idx}, id: {item['id']}, num tokens: {len(tokens)}")
        return {
            'id': item['id'],
            'level': int(item['level'][-1]),
            'type': item['type'],
            'tokens': tokens,
            'mask': mask,
            'labels': labels,
        }

class GSM8kMistralDataset(Dataset):
    def __init__(self, gsm8k_dataset_location, tokenizer):
        # Load the gsm8k jsonl file
        data = []
        with open(gsm8k_dataset_location, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        items = []
        for i, entry in enumerate(data):
            items.append({
                'id': i,
                'problem': entry['question'],
                'solution': entry['answer'],
            })

        self.items = items
        self.tokenizer = tokenizer
        self.BOS_TOKEN = "<s>"  # Beginning of sequence
        self.EOS_TOKEN = "</s>"  # End of sequence
        self.INST_BEGIN = "[INST]"  # Instruction block start
        self.INST_END = "[/INST]"  # Instruction block end

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        full_instruction = f"{GSM8K_GENERAL_PROMPT}\n\n{item['problem']}"

        prompt = (
            f"{self.BOS_TOKEN}"
            f"{self.INST_BEGIN} {full_instruction}{self.INST_END}"  # Instruction block
        )

        solution = item['solution']
        solution = solution.replace('#### ', '\\boxed{').replace('####', '\\boxed{')
        solution += "}"
        prompt_tokens = self.tokenizer.encode(prompt)[:-1]
        solution_tokens = self.tokenizer.encode(solution)[1:]
        #print(f"Prompt tokens: {prompt_tokens}, \n Solution tokens: {solution_tokens}")
        tokens = prompt_tokens + solution_tokens
        tokens = tokens[:2048]
        mask = [False] * len(tokens)
        # mask[:len(prompt_tokens) + 7] = [True] * (len(prompt_tokens) + 7)
        labels = tokens[1:] + [ -100 ]
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
        #print(f"idx: {idx}, id: {item['id']}, num tokens: {len(tokens)}")
        return {
            'id': item['id'],
            'tokens': tokens,
            'mask': mask,
            'labels': labels,
        }

def combined_mistral_math_gsm8k_fineweb_dataset(tokenizer,
                                                math_location=None,
                                                gsm_location=None):
     """
     Creates a combined dataset for Mistral model fine-tuning on MATH and GSM8K datasets.

     This function builds and combines three distinct datasets:
     1. MATH dataset for mathematical reasoning.
     2. GSM8K dataset for grade-school math problems.
     3. FineWeb dataset for general natural language.

     The final output is a `CombinedDataset` instance that interleaves samples
     from these sources.

     Args:
          tokenizer (ModelTokenizer): The tokenizer to use for all sub-datasets.
          math_location (Optional[str], optional): Path to the MATH dataset's
                training directory. Defaults to a relative path.
          gsm_location (Optional[str], optional): Path to the GSM8K training
                data file. Defaults to a relative path.

     Returns:
          CombinedDataset: A dataset that combines the three specified sources.
     """
     if math_location is None:
          math_location = "../data/MATH/train"
     if gsm_location is None:
          gsm_location = "../data/grade-school-math/grade_school_math/data/train.jsonl"
     math_sft_dataset = MATHMistralDataset(math_location, tokenizer)
     gsm_8k_sft_dataset = GSM8kMistralDataset(gsm_location, tokenizer)
     fineweb_sft_dataset = fineweb_dataset(tokenizer)
     combined_dataset = CombinedDataset([math_sft_dataset, gsm_8k_sft_dataset, fineweb_sft_dataset],
                                        relative_weights=[1, 1 ,8])
     return combined_dataset