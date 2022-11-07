from re import L


def create_prompt_classification_daniel1(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    prompt = f"Classify into {len(unique_labels)} labels: {','.join(unique_labels)}\n\n"

    for text, label in data[:n_shot]:
        prompt += f"Text: {text}\nLabel: {label}\n"
    prompt += "Text: {text}\nLabel:"
    return prompt


def create_prompt_classification_daniel2(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    prompt = f"Classify Text into one of {len(unique_labels)} labels: {','.join(unique_labels)}\n\n"

    for text, label in data[:n_shot]:
        prompt += f"Text: {text}\nLabel: {label}\n\n"
    prompt += "Text: {text}\nLabel:"
    return prompt


def create_prompt_classification_daniel3(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    prompt = f"Classify each Tweet into one of {len(unique_labels)} sentiments: {','.join(unique_labels)}\n\n"

    for text, label in data[:n_shot]:
        prompt += f"Tweet: {text}\nSentiment: {label}\n\n"
    prompt += "Tweet: {text}\nSentiment:"
    return prompt


def create_prompt_classification_daniel4(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    prompt = f"Classify each Tweet into one of {len(unique_labels)} sentiments: {','.join(unique_labels)}\n\n"

    for text, label in data[:n_shot]:
        prompt += f'Tweet: """{text}"""\nSentiment: {label}\n\n'
    prompt += 'Tweet: """{text}"""\nSentiment:'
    return prompt


def create_prompt_classification_daniel5(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    prompt = f"Sentiment classify each Tweet into one of these sentiments: {','.join(unique_labels)}\n\n"

    for text, label in data[:n_shot]:
        prompt += f"Tweet: {text}\nSentiment: {label}\n\n"
    prompt += "Tweet: {text}\nSentiment:"
    return prompt


def create_prompt_classification_bloom1(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    interim_prompt = f"To get full credit in this exam, choose the correct emotion from the following choices: {','.join(unique_labels)}"
    prompt = ""

    for text, label in data[:n_shot]:
        prompt += f"{text}\n\n{interim_prompt}\n|||\n{label}\n\n"
    prompt += f"{{text}}\n\n{interim_prompt}\n|||\n"
    return prompt


def create_prompt_classification_bloom2(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    interim_prompt = f"Which emotion among {','.join(unique_labels)} best describes the feeling of the author of the following tweet?"
    prompt = ""

    for text, label in data[:n_shot]:
        prompt += f"{interim_prompt}\n\n{text}|||\n{label}\n\n\n"
    prompt += f"{interim_prompt}\n\n{{text}}|||\n"
    return prompt


def create_prompt_classification_bloom3(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    interim_prompt = f"Possible emotions: {','.join(unique_labels)}"
    prompt = ""

    for text, label in data[:n_shot]:
        prompt += f"Which emotion is best represented by the following tweet?\n\n{text}\n\n\n{interim_prompt}\n\n|||\n\n{label}\n\n\n"
    prompt += f"Which emotion is best represented by the following tweet?\n\n{{text}}\n\n\n{interim_prompt}\n\n|||\n\n"
    return prompt


def create_prompt_classification_bloom4(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    options = f"(a) {unique_labels[0]}\n(b) {unique_labels[1]}\n(c) {unique_labels[2]}\n(d) {unique_labels[3]}\n"
    interim_prompt = (
        f"Categorize the tweet into one of the following options:\n{options}"
    )
    prompt = ""

    for text, label in data[:n_shot]:
        prompt += f"{text}\n\n{interim_prompt}\n|||\n{label}\n\n\n"
    prompt += f"{{text}}\n\n{interim_prompt}\n|||\n"
    return prompt


def create_prompt_classification_bloom5(data, n_shot):
    unique_labels = list(set([label for _, label in data]))
    interim_prompt = (
        f"What is the emotion of the text?\n\n\nHint: {','.join(unique_labels)}"
    )
    prompt = ""

    for text, label in data[:n_shot]:
        prompt += f"{text}\n\n\n{interim_prompt}\n\n|||\n\n{label}\n\n"
    prompt += f"{{text}}\n\n\n{interim_prompt}\n\n|||\n\n"
    return prompt


def create_prompt_ner_1(data, n_shot):
    prompt = f"Extract tags for art, building, event, location, organization, other, person, product\n\n"

    for text, tokens, labels in data[:n_shot]:
        tagged_text = []
        for i in range(len(tokens)):
            if labels[i] != "O":
                tagged_text.append(f"{tokens[i]}({labels[i]})")
            else:
                tagged_text.append(tokens[i])
        prompt += f"Text: {text}\nTags: {' '.join(tagged_text)}\n"
    prompt += "Text: {text}\nTags:"
    return prompt


def create_prompt_ner_2(data, n_shot):
    prompt = f"For each text, mark tags from one of the categories: art, building, event, location, organization, other, person, product\n\n"

    for text, tokens, labels in data[:n_shot]:
        tagged_text = []
        for i in range(len(tokens)):
            if labels[i] != "O":
                tagged_text.append(f"{tokens[i]}({labels[i]})")
            else:
                tagged_text.append(tokens[i])
        prompt += f"Text: {text}\nMarked Text: {' '.join(tagged_text)}\n"
    prompt += "Text: {text}\nMarked Text:"
    return prompt


def create_prompt_ner_3(data, n_shot):
    prompt = f"For each text, extract NER tags from the taxonomy: art, building, event, location, organization, other, person, product\n\n"

    for text, tokens, labels in data[:n_shot]:
        tagged_text = []
        for i in range(len(tokens)):
            if labels[i] != "O":
                tagged_text.append(f"{tokens[i]}({labels[i]})")
            else:
                tagged_text.append(tokens[i])
        prompt += f"Text: {text}\nMarked Text: {' '.join(tagged_text)}\n"
    prompt += "Text: {text}\nMarked Text:"
    return prompt


def create_prompt_ner_4(data, n_shot):
    prompt = f"For each text, mark tags from one of the categories: art, building, event, location, organization, other, person, product\n\n"

    for text, tokens, labels in data[:n_shot]:
        tagged_text = []
        for i in range(len(tokens)):
            if labels[i] != "O":
                tagged_text.append(f"[{tokens[i]}]({labels[i]})")
            else:
                tagged_text.append(tokens[i])
        prompt += f"Text: {text}\Marked Text: {' '.join(tagged_text)}\n"
    prompt += "Text: {text}\Marked Text:"
    return prompt


def create_prompt_ner_5(data, n_shot):
    prompt = f"For each text, mark NER tags.\n Tag categories: art, building, event, location, organization, other, person, product\n\n"

    for text, tokens, labels in data[:n_shot]:
        tagged_text = []
        for i in range(len(tokens)):
            if labels[i] != "O":
                tagged_text.append(f"[{tokens[i]}]({labels[i]})")
            else:
                tagged_text.append(tokens[i])
        prompt += f"Text: {text}\Marked Text: {' '.join(tagged_text)}\n"
    prompt += "Text: {text}\Marked Text:"
    return prompt


def create_prompt_similarity_nick1(data, n_shot):
    prompt = "Classify sentence similarity with similar, not similar\n\n"

    for (text_A, text_B), label in data[:n_shot]:
        prompt += f"Text A: {text_A}\nText B: {text_B}\nLabel: {'similar' if label else 'not similar'}\n"
    prompt += "Text A: {text_A}\nText B: {text_B}\nLabel:"
    return prompt


def create_prompt_similarity_2(data, n_shot):
    prompt = "Decide if two sentences are similar or not similar\n\n"

    for (text_A, text_B), label in data[:n_shot]:
        prompt += f"Sentence 1: {text_A}\nSentence 2: {text_B}\nLabel: {'similar' if label else 'not similar'}\n"
    prompt += "Sentence 1: {text_A}\nSentence 2: {text_B}\nLabel:"
    return prompt


def create_prompt_similarity_3(data, n_shot):
    prompt = "Decide if two texts are similar. If similar respond with 'similar', if not 'not similar'\n\n"

    for (text_A, text_B), label in data[:n_shot]:
        prompt += f"Text A: {text_A}\nText B: {text_B}\nResponse: {'similar' if label else 'not similar'}\n\n"
    prompt += "Text A: {text_A}\nText B: {text_B}\nResponse:"
    return prompt


def create_prompt_similarity_4(data, n_shot):
    prompt = ""

    for (text_A, text_B), label in data[:n_shot]:
        prompt += f"Is the sentence \"{text_A}\" similar to sentence \"{text_B}\" ?\nResponse: {'similar' if label else 'not similar'}\n\n"
    prompt += 'Is the sentence "{text_A}" similar to sentence "{text_B}" ?\nResponse:'
    return prompt


def create_prompt_similarity_5(data, n_shot):
    prompt = "Decide if two sentences are similar. If similar respond with 'similar', if not 'not similar'\n\n"

    for (text_A, text_B), label in data[:n_shot]:
        prompt += f"Sentence 1: {text_A}\nSentence 2: {text_B}\nAre the two sentences similar?\nResponse: {'similar' if label else 'not similar'}\n\n"
    prompt += "Sentence 1: {text_A}\nSentence 2: {text_B}\nAre the two sentences similar?\nResponse:"
    return prompt


def create_prompt_summarisation_nick1(data, n_shot):
    prompt = "Summarise the following text\n\n"

    for text, summary in data[:n_shot]:
        prompt += f"Text: {text}\nSummary: {summary}\n"
    prompt += "Text: {text}\nSummary:"
    return prompt


def create_prompt_summarisation_2(data, n_shot):
    prompt = ""

    for text, summary in data[:n_shot]:
        prompt += f"{text}\n|||\nCreate a summary of the given text: {summary}\n\n"
    prompt += "{text}\n|||\nCreate a summary of the given text:"
    return prompt


def create_prompt_summarisation_3(data, n_shot):
    prompt = (
        "Can you extract in 2 or 3 sentences the hightlights of the news article?\n\n"
    )

    for text, summary in data[:n_shot]:
        prompt += f"Article: {text}\n|||\nHighlights: {summary}\n\n"
    prompt += "Article: {text}\n|||\nHighlights:"
    return prompt


def create_prompt_summarisation_4(data, n_shot):
    prompt = "Extract key highlight points from each article:\n\n"

    for text, summary in data[:n_shot]:
        prompt += f"Article: {text}\n|||\nHighlights: {summary}\n\n"
    prompt += "Article: {text}\n|||\nHighlights:"
    return prompt


def create_prompt_summarisation_5(data, n_shot):
    prompt = "For each article given, extract key highlight points from them, by using the words in the text:\n\n"

    for text, summary in data[:n_shot]:
        prompt += f"Article: {text}\n|||\nHighlights: {summary}\n\n"
    prompt += "Article: {text}\n|||\nHighlights:"
    return prompt


def create_prompt_qa_1(data, n_shot):
    prompt = ""

    for text, answer in data[:n_shot]:
        context, question = text
        prompt += f"Q: {question}\nContext: {context}\nA: {answer}\n"
    prompt += "Q: {question}\nContext: {context}\nA:"
    return prompt


def create_prompt_qa_2(data, n_shot):
    prompt = "Answer the following questions\n\n"

    for text, answer in data[:n_shot]:
        context, question = text
        prompt += f"Question: {question}\nContext: {context}\nAnswer: {answer}\n"
    prompt += "Question: {question}\nContext: {context}\nAnswer:"
    return prompt


def create_prompt_qa_3(data, n_shot):
    prompt = "Answer the following questions based on given context\n\n"

    for text, answer in data[:n_shot]:
        context, question = text
        prompt += f"Question: {question}\nContext: {context}\nAnswer: {answer}\n"
    prompt += "Question: {question}\nContext: {context}\nAnswer:"
    return prompt


def create_prompt_qa_4(data, n_shot):
    prompt = ""

    for text, answer in data[:n_shot]:
        context, question = text
        prompt += f"{context}\n\nQuestion: {question}\nReffering to the passage above, the correct answer to the given question is\n||| {answer}\n\n\n"
    prompt += "{context}\n\nQuestion: {question}\nReffering to the passage above, the correct answer to the given question is\n|||"
    return prompt


def create_prompt_qa_5(data, n_shot):
    prompt = "For each question, extract the answer from the given context:\n\n"

    for text, answer in data[:n_shot]:
        context, question = text
        prompt += f"Question: {question}\nContext: {context}\nAnswer: {answer}\n"
    prompt += "Question: {question}\nContext: {context}\nAnswer:"
    return prompt


CREATE_PROMPT_FUNCTIONS = {
    "classification-1": create_prompt_classification_daniel1,
    "classification-2": create_prompt_classification_daniel2,
    "classification-3": create_prompt_classification_daniel3,
    "classification-4": create_prompt_classification_daniel4,
    "classification-5": create_prompt_classification_daniel5,
    "classification-bloom1": create_prompt_classification_bloom1,
    "classification-bloom2": create_prompt_classification_bloom2,
    "classification-bloom3": create_prompt_classification_bloom3,
    "classification-bloom4": create_prompt_classification_bloom4,
    "classification-bloom5": create_prompt_classification_bloom5,
    "ner-1": create_prompt_ner_1,
    "ner-2": create_prompt_ner_2,
    "ner-3": create_prompt_ner_3,
    "ner-4": create_prompt_ner_4,
    "ner-5": create_prompt_ner_5,
    "similarity-1": create_prompt_similarity_nick1,
    "similarity-2": create_prompt_similarity_2,
    "similarity-3": create_prompt_similarity_3,
    "similarity-4": create_prompt_similarity_4,
    "similarity-5": create_prompt_similarity_5,
    "summarisation-1": create_prompt_summarisation_nick1,
    "summarisation-2": create_prompt_summarisation_2,
    "summarisation-3": create_prompt_summarisation_3,
    "summarisation-4": create_prompt_summarisation_4,
    "summarisation-5": create_prompt_summarisation_5,
    "qa-1": create_prompt_qa_1,
    "qa-2": create_prompt_qa_2,
    "qa-3": create_prompt_qa_3,
    "qa-4": create_prompt_qa_4,
    "qa-5": create_prompt_qa_5,
}


def create_prompt(data, n_shot, task, prompt_name=None):
    prompt_func_name = task
    if prompt_name:
        prompt_func_name += "-" + prompt_name
    prompt_func = CREATE_PROMPT_FUNCTIONS[prompt_func_name]
    return prompt_func(data, n_shot)
