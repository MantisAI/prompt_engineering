from datasets import load_dataset
from transformers import AutoTokenizer


def load_tweet_eval_data(sample_size):
    data = load_dataset("tweet_eval", "emotion", split=f"test[:{sample_size}]")

    labels_map = ["anger", "joy", "optimism", "sadness"]
    data = [(example["text"], labels_map[example["label"]]) for example in data]
    return data


FEW_NERD_ID2LABEL = {
    0: "O",
    1: "art",
    2: "building",
    3: "event",
    4: "location",
    5: "organization",
    6: "other",
    7: "person",
    8: "product",
}


def load_few_nerd(sample_size):
    data = load_dataset(
        "DFKI-SLT/few-nerd", "supervised", split=f"test[:{sample_size}]"
    )

    ner_data = []
    for example in data:
        text = " ".join(example["tokens"])
        tokens = [
            token
            for token, label_idx in zip(example["tokens"], example["ner_tags"])
            # if label_idx != 0
        ]
        labels = [
            FEW_NERD_ID2LABEL[label_idx]
            for token, label_idx in zip(example["tokens"], example["ner_tags"])
            # if label_idx != 0
        ]
        ner_data.append((text, tokens, labels))
    return ner_data


def load_sick(sample_size):
    data = load_dataset("sick", split=f"test[:{sample_size}]")

    data = [
        (
            [example["sentence_A"], example["sentence_B"]],
            int(example["relatedness_score"] > 3.5),
        )
        for example in data
    ]
    return data


def load_summarisation(sample_size):
    data = load_dataset("cnn_dailymail", "3.0.0", split=f"test")

    general_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    final_data = []
    for e in data:
        encoded = general_tokenizer(e["article"], return_tensors="pt")
        if encoded["attention_mask"].shape[1] < 500:
            final_data.append((e["article"], e["highlights"]))
            if len(final_data) == sample_size:
                break
    # data = [(example["article"], example["highlights"]) for example in data]
    return final_data


def load_squad(sample_size):
    data = load_dataset("squad", split=f"train[:{sample_size}]")

    data = [
        ([example["question"], example["context"]], example["answers"]["text"][0])
        for example in data
    ]
    return data


def load_data(task, sample_size):
    if task == "classification":
        return load_tweet_eval_data(sample_size)
    if task == "ner":
        return load_few_nerd(sample_size)
    if task == "similarity":
        return load_sick(sample_size)
    if task == "summarisation":
        return load_summarisation(sample_size)
    if task == "qa":
        # We use squad instead of hotpodqa because it was difficult to
        # retrieve context and question from hotpodqa as the link data missing
        return load_squad(sample_size)
