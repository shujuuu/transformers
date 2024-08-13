---
title: Overview of Token Classification in Pipelines
---
Token Classification is a pipeline that uses models <SwmToken path="src/transformers/pipelines/token_classification.py" pos="100:25:27" line-data="    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the">`fine-tuned`</SwmToken> for token classification tasks.

It is primarily used for Named Entity Recognition (NER), which involves predicting the classes of tokens in a sequence, such as person, organization, location, or miscellaneous.

The pipeline can be loaded using the task identifier 'ner' and supports models listed on the Hugging Face model hub.

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="53:16:16" line-data="    &quot;&quot;&quot;All the valid aggregation strategies for TokenClassificationPipeline&quot;&quot;&quot;">`TokenClassificationPipeline`</SwmToken> class handles the initialization of the model, tokenizer, and other parameters such as device placement and aggregation strategies.

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="32:3:3" line-data="    def __call__(self, inputs: Union[str, List[str]], **kwargs):">`__call__`</SwmToken> method processes the input texts, classifies each token, and returns the results in a structured format.

The pipeline supports various aggregation strategies to fuse tokens based on model predictions, including 'none', 'simple', 'first', 'average', and 'max'.

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="27:2:2" line-data="class TokenClassificationArgumentHandler(ArgumentHandler):">`TokenClassificationArgumentHandler`</SwmToken> class manages the input arguments and ensures they are correctly formatted for processing.

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="252:3:3" line-data="    def gather_pre_entities(">`gather_pre_entities`</SwmToken> method collects necessary information for each token, which is then used by the <SwmToken path="src/transformers/pipelines/token_classification.py" pos="238:7:7" line-data="            grouped_entities = self.aggregate(pre_entities, self.aggregation_strategy)">`aggregate`</SwmToken> method to group tokens into entities.

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="91">

---

<SwmToken path="src/transformers/pipelines/token_classification.py" pos="91:2:2" line-data="class TokenClassificationPipeline(Pipeline):">`TokenClassificationPipeline`</SwmToken> The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="91:2:2" line-data="class TokenClassificationPipeline(Pipeline):">`TokenClassificationPipeline`</SwmToken> class handles the initialization of the model, tokenizer, and other parameters such as device placement and aggregation strategies.

```python
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="175">

---

Processing Inputs The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="175:3:3" line-data="    def __call__(self, inputs: Union[str, List[str]], **kwargs):">`__call__`</SwmToken> method processes the input texts, classifies each token, and returns the results in a structured format.

```python
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="52">

---

Aggregation Strategies The pipeline supports various aggregation strategies to fuse tokens based on model predictions, including 'none', 'simple', 'first', 'average', and 'max'.

```python
class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="27">

---

Argument Handling The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="27:2:2" line-data="class TokenClassificationArgumentHandler(ArgumentHandler):">`TokenClassificationArgumentHandler`</SwmToken> class manages the input arguments and ensures they are correctly formatted for processing.

```python
class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="252">

---

Gathering Pre-Entities The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="252:3:3" line-data="    def gather_pre_entities(">`gather_pre_entities`</SwmToken> method collects necessary information for each token, which is then used by the <SwmToken path="src/transformers/pipelines/token_classification.py" pos="238:7:7" line-data="            grouped_entities = self.aggregate(pre_entities, self.aggregation_strategy)">`aggregate`</SwmToken> method to group tokens into entities.

```python
    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
```

---

</SwmSnippet>

# Main functions

Main functions

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="91">

---

## <SwmToken path="src/transformers/pipelines/token_classification.py" pos="91:2:2" line-data="class TokenClassificationPipeline(Pipeline):">`TokenClassificationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="91:2:2" line-data="class TokenClassificationPipeline(Pipeline):">`TokenClassificationPipeline`</SwmToken> class is used for Named Entity Recognition (NER) tasks. It initializes the model, tokenizer, and other parameters such as device placement and aggregation strategies.

```python
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/token_classification.py" line="27">

---

## <SwmToken path="src/transformers/pipelines/token_classification.py" pos="27:2:2" line-data="class TokenClassificationArgumentHandler(ArgumentHandler):">`TokenClassificationArgumentHandler`</SwmToken>

The <SwmToken path="src/transformers/pipelines/token_classification.py" pos="27:2:2" line-data="class TokenClassificationArgumentHandler(ArgumentHandler):">`TokenClassificationArgumentHandler`</SwmToken> class manages the input arguments and ensures they are correctly formatted for processing.

```python
class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
