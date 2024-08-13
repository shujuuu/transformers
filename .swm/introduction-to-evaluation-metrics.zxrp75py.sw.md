---
title: Introduction to Evaluation Metrics
---
Data Metrics are used to evaluate the performance of models by comparing predictions against actual values.

They include metrics such as exact match scores and <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="73:16:16" line-data="        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise">`F1`</SwmToken> scores, which measure the accuracy and quality of the predictions.

The exact match score calculates the percentage of predictions that match the ground truth exactly.

The <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="73:16:16" line-data="        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise">`F1`</SwmToken> score is the harmonic mean of precision and recall, providing a balance between the two.

These metrics help in understanding how well the model is performing and where it might need improvements.

<SwmSnippet path="/src/transformers/data/metrics/squad_metrics.py" line="83">

---

# <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="83:2:2" line-data="def get_raw_scores(examples, preds):">`get_raw_scores`</SwmToken> Function

The <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="83:2:2" line-data="def get_raw_scores(examples, preds):">`get_raw_scores`</SwmToken> function computes the exact and <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="85:9:9" line-data="    Computes the exact and f1 scores from the examples and the model predictions">`f1`</SwmToken> scores from the examples and the model predictions. It iterates over the examples, normalizes the answers, and calculates the scores.

```python
def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/metrics/squad_metrics.py" line="37">

---

# <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="37:2:2" line-data="def normalize_answer(s):">`normalize_answer`</SwmToken> Function

The <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="37:2:2" line-data="def normalize_answer(s):">`normalize_answer`</SwmToken> function is used to lower text and remove punctuation, articles, and extra whitespace. This normalization is crucial for accurate comparison of predictions and ground truth.

```python
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/metrics/squad_metrics.py" line="63">

---

# <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="63:2:2" line-data="def compute_exact(a_gold, a_pred):">`compute_exact`</SwmToken> Function

The <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="63:2:2" line-data="def compute_exact(a_gold, a_pred):">`compute_exact`</SwmToken> function calculates the exact match score by comparing the normalized predicted answer with the normalized ground truth answer.

```python
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/metrics/squad_metrics.py" line="67">

---

# <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="67:2:2" line-data="def compute_f1(a_gold, a_pred):">`compute_f1`</SwmToken> Function

The <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="67:2:2" line-data="def compute_f1(a_gold, a_pred):">`compute_f1`</SwmToken> function calculates the <SwmToken path="src/transformers/data/metrics/squad_metrics.py" pos="73:16:16" line-data="        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise">`F1`</SwmToken> score by computing the precision and recall of the predicted tokens against the ground truth tokens.

```python
def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
