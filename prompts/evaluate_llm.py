from random import randint
from tqdm import tqdm
import typer
from loguru import logger

from prompts.data import load_data
from prompts.prompts import create_prompt
from prompts.models import load_model
from prompts.tasks import process_model_output, calculate_metrics
from transformers import AutoTokenizer
import json
import time

general_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def decide_on_max_tokens(task, text):
    if task == "classification":
        return 3
    if task == "ner":
        encoded = general_tokenizer(text, return_tensors="pt")
        return encoded["attention_mask"].shape[1] * 3
    if task == "qa":
        return 15
    if task == "summarisation":
        return 64

    return 10


def evaluate_llm(
    task,
    model_name="text-ada-001",
    prompt_name=None,
    sample_size: int = 100,
    n_shot: int = 0,
    api: bool = True,
    log_filepath=None,
):
    data = load_data(task, sample_size + n_shot)

    prompt_template = create_prompt(data, n_shot, task, prompt_name)
    if model_name == "ul2":
        prompt_template = f"[S2S] {prompt_template} <extra_id_0>"
    logger.debug(prompt_template)

    data = data[n_shot:]
    if task == "ner":
        texts, tokens, labels = zip(*data)
    else:
        texts, labels = zip(*data)

    model = load_model(model_name, api)
    log_data = {"prompt_template": prompt_template, "data": []}
    preds = []
    i = 0
    start_time_total = time.time()
    for text in tqdm(texts):
        start_time = time.time()
        if task == "similarity":
            text_A, text_B = text
            prompt = prompt_template.format(text_A=text_A, text_B=text_B)
        elif task == "qa":
            context, question = text
            prompt = prompt_template.format(context=context, question=question)
        else:
            prompt = prompt_template.format(text=text)

        output = model(prompt, max_tokens=decide_on_max_tokens(task, text))
        pred = process_model_output(output, prompt_name, task)
        log_data["data"].append(
            {
                "prompt": prompt,
                "result": output,
                "pred": pred,
                "label": labels[i],
                "time": time.time() - start_time,
            }
        )

        preds.append(pred)
        i += 1

    # logger.debug(labels)
    # logger.debug(preds)
    metrics = calculate_metrics(data, preds, task)
    log_data["metrics"] = metrics
    log_data["total_time"] = time.time() - start_time_total
    log_data["task"] = task
    log_data["model_name"] = model_name
    log_data["prompt_name"] = prompt_name
    log_data["sample_size"] = sample_size
    log_data["n_shot"] = n_shot
    log_data["use_api"] = api
    # logger.debug(metrics)

    if log_filepath is None:
        if prompt_name is None:
            prompt_name = "default"
        model_log_name = model_name.replace("/", "-")
        log_filepath = f"llm_logs/{task}-{model_log_name}-{sample_size}-{prompt_name}-{n_shot}.json"

    with open(log_filepath, "w") as f:
        json.dump(log_data, f, indent=4)


if __name__ == "__main__":
    typer.run(evaluate_llm)
