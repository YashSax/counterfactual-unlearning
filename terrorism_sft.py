"""
Supervised fine-tuning for 9/11 unlearning.

This script fine-tunes Llama-3.1-8B on responses that avoid mentioning 9/11,
teaching the model to answer questions without referencing the attacks.
"""

import asyncio
import json
import sys
import os
from dotenv import load_dotenv

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

load_dotenv()


def convert_dataset_to_conversations(input_path: str, output_path: str) -> tuple[int, int]:
    """Convert our Q&A format to the conversations format expected by tinker.

    Input format: {"question": "...", "response": "..."}
    Output format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Returns: (count_valid, count_skipped)
    """
    count = 0
    skipped = 0
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            item = json.loads(line)

            question = item.get("question")
            response = item.get("response")

            if not question or not response:
                skipped += 1
                continue

            conversation = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
            }
            outfile.write(json.dumps(conversation) + "\n")
            count += 1
    return count, skipped


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Convert dataset to conversations format
    input_path = "./data/911_sft_dataset.jsonl"
    output_path = "./data/911_conversations.jsonl"

    if os.path.exists(input_path):
        count, skipped = convert_dataset_to_conversations(input_path, output_path)
        print(f"Converted {count} examples to {output_path}")
        if skipped > 0:
            print(f"Warning: Skipped {skipped} entries with null/empty content")
    else:
        print(f"Warning: {input_path} not found. Make sure to run create_dataset.ipynb first.")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=output_path
    )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-911-sft-fixed-hopefully",
            "model_name": model_name,
            "renderer_name": renderer_name,
            "dataset_builder": dataset,
            "learning_rate": 2e-5,
            "lr_schedule": "linear",
            "num_epochs": 3,
            "eval_every": 10,
            "save_every": 50,
            "lora_rank": 32,
        }
    )


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
