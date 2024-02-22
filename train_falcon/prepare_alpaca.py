import json
import sys
import random
from pathlib import Path
from typing import Optional

import requests

from tqdm import tqdm
from mindspore.mindrecord import FileWriter

# support running without installing as a package
wd = Path(
    __file__
).parent.parent.parent.parent.resolve()  ## put the path of mindnlp here
sys.path.append(str(wd))

from mindnlp.transformers import AutoTokenizer as Tokenizer


def random_split(data, split_ratios, seed):
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split_sizes = [int(ratio * len(data)) for ratio in split_ratios]
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(indices[start : start + size])
        start += size
    return splits


def prepare(
    model_name_or_path: str = "falcon-rw-1b",
    destination_path: Path = Path("llm/peft/train_falcon/data/alpaca"),
    checkpoint_dir: Path = Path(".mindnlp/model/Rocketknight1/falcon-rw-1b"),
    test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = "alpaca_data_cleaned_archive.json",
    data_file_url: str = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json",
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.ms` and `test.ms`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = 512

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    print("Loading data file...")
    download_if_missing(data_file_path, data_file_url)
    with open(data_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained(model_name_or_path)

    # Partition the dataset into train and test
    train_indices, test_indices = random_split(
        data, [1.0 - test_split_fraction, test_split_fraction], seed
    )
    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    # 定义schema
    writer = FileWriter(str(destination_path / "train.ms"), shard_num=1, overwrite=True)
    data_schema = {"instruction": {"type": "string"}, "input": {"type": "string"}, "output": {"type": "string"}}
    writer.add_schema(data_schema,"alpaca_schema")
    writer.write_raw_data(train_set)
    writer.commit()

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    writer = FileWriter(str(destination_path / "test.ms"), shard_num=1, overwrite=True)
    data_schema = {"instruction": {"type": "string"}, "input": {"type": "string"}, "output": {"type": "string"}}
    writer.add_schema(data_schema,"alpaca_schema")
    writer.write_raw_data(test_set)
    writer.commit()


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool,
    ignore_index: int,
) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(
        full_prompt_and_response, max_length=max_length
    )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.copy()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
