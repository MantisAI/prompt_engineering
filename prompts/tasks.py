from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import evaluate
import re

from nervaluate import Evaluator


def process_model_output(model_output, prompt_name, task="classification"):
    output = model_output.strip()
    if task == "classification":
        output = re.sub(r"\([abcd]\)", "", output)
        if output.split():
            return output.split()[0]
        else:
            return output
    if task == "ner":
        tags = output.split(" ")
        if prompt_name in ["4", "5"]:
            for i in range(len(tags)):
                tags[i] = tags[i].strip()
                match = re.match(r"\[(.*)\]\((.*)\)", tags[i])
                if match:
                    tags[i] = match.group(2)
                else:
                    tags[i] = "O"
        else:
            for i in range(len(tags)):
                tags[i] = tags[i].strip()
                if tags[i].find("(") >= 0:
                    tag = tags[i][tags[i].find("(") + 1 : -1]
                    if tag in [
                        "art",
                        "building",
                        "event",
                        "location",
                        "organization",
                        "other",
                        "person",
                        "product",
                    ]:
                        tags[i] = tag
                    else:
                        tags[i] = "O"
                else:
                    tags[i] = "O"
        return tags
    if task == "similarity":
        return int("not similar" not in output)
    if task == "summarisation":
        return output
    if task == "qa":
        return output


def array_to_prodigy(a):
    spans = []
    for i in range(len(a)):
        if a[i] != "O":
            spans.append({"label": a[i], "start": i, "end": i + 1})
    return spans


def calculate_nervaluate_metrics(labels, preds):
    true = []
    pred = []
    for i in range(len(labels)):
        true.append(array_to_prodigy(labels[i]))
        pred.append(array_to_prodigy(preds[i]))
    evaluator = Evaluator(
        true,
        pred,
        tags=[
            "art",
            "building",
            "event",
            "location",
            "organization",
            "other",
            "person",
            "product",
        ],
    )

    results, results_per_tag = evaluator.evaluate()
    return results["strict"]


def calculate_metrics(data, preds, task="classification"):
    if task == "ner":
        texts, tokens, labels = zip(*data)
    else:
        texts, labels = zip(*data)

    if task == "classification":
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {"p": p, "r": r, "f1": f1}
    if task == "ner":
        """mlb = MultiLabelBinarizer()
        mlb.fit(labels)

        Y = mlb.transform(labels)
        Y_pred = mlb.transform(preds)
        p, r, f1, _ = precision_recall_fscore_support(Y, Y_pred, average="micro")"""
        ner_r = calculate_nervaluate_metrics(labels, preds)
        return {"p": ner_r["precision"], "r": ner_r["recall"], "f1": ner_r["f1"]}
    if task == "similarity":
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"p": p, "r": r, "f1": f1}
    if task == "summarisation":
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=preds, references=labels)
    if task == "qa":
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=preds, references=labels)
