import os
import json
from os.path import exists
import queue
from prompts.evaluate_llm import evaluate_llm
from multiprocessing import Pool
import time
import random

MODELS = [
    # "text-davinci-002",
    # "text-curie-001",
    # "text-babbage-001",
    # "bigscience/bloom",
    # "EleutherAI/gpt-neo-2.7B",
    # "EleutherAI/gpt-j-6B",
    # "gpt-neox-20b",
    # "co:here",
    # "jurassic1-jumbo",
    # "ul2",
    # "facebook/opt-30b",
    "google/flan-t5-xxl"
]

prompts = ["1", "2", "3", "4", "5"]

n_shots = [1, 3, 5, 10]

sample_size = 300

TASK = "classification"

task_arguments = []

for model in MODELS:
    for prompt in prompts:
        for n_shot in n_shots:
            prompt_name = prompt
            model_name = model.replace("/", "-")
            log_file = f"llm_logs/{TASK}-{model_name}-{sample_size}-{prompt_name}-{n_shot}.json"
            if not exists(log_file):
                print(log_file)
                task_arguments.append(
                    {
                        "task": TASK,
                        "model_name": model,
                        "prompt_name": prompt,
                        "n_shot": n_shot,
                        "sample_size": sample_size,
                    }
                )


def run_evaluate_llm(data):
    print(f"Running for {data['model_name']}-{data['prompt_name']}-{data['n_shot']}")
    evaluate_llm(**data)
    print(
        f"Finished running for {data['model_name']}-{data['prompt_name']}-{data['n_shot']}"
    )


if __name__ == "__main__":
    p = Pool(1)
    p.map(run_evaluate_llm, task_arguments)
