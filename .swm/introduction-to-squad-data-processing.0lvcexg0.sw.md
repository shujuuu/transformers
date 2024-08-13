---
title: Introduction to Squad Data Processing
---
Squad refers to a set of classes and functions designed to process the Stanford Question Answering Dataset (SQuAD).

The <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken> class is a key component that handles the reading and processing of <SwmToken path="src/transformers/data/processors/squad.py" pos="335:17:17" line-data="        examples: list of :class:`~transformers.data.processors.squad.SquadExample`">`squad`</SwmToken> data files.

The <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken> class is extended by <SwmToken path="src/transformers/data/processors/squad.py" pos="544:18:18" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SquadV1Processor`</SwmToken> and <SwmToken path="src/transformers/data/processors/squad.py" pos="544:22:22" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SquadV2Processor`</SwmToken> to handle different versions of the <SwmToken path="src/transformers/data/processors/squad.py" pos="335:17:17" line-data="        examples: list of :class:`~transformers.data.processors.squad.SquadExample`">`squad`</SwmToken> dataset.

The <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken> function converts a list of examples into a list of features that can be directly used as input to a model.

The <SwmToken path="src/transformers/data/processors/squad.py" pos="759:2:2" line-data="class SquadFeatures:">`SquadFeatures`</SwmToken> class represents the features of a single <SwmToken path="src/transformers/data/processors/squad.py" pos="335:17:17" line-data="        examples: list of :class:`~transformers.data.processors.squad.SquadExample`">`squad`</SwmToken> example, including token indices, attention masks, and other relevant information.

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="542">

---

# <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken> class is a key component that handles the reading and processing of <SwmToken path="src/transformers/data/processors/squad.py" pos="544:7:7" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SQuAD`</SwmToken> data files. It is extended by <SwmToken path="src/transformers/data/processors/squad.py" pos="544:18:18" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SquadV1Processor`</SwmToken> and <SwmToken path="src/transformers/data/processors/squad.py" pos="544:22:22" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SquadV2Processor`</SwmToken> to handle different versions of the <SwmToken path="src/transformers/data/processors/squad.py" pos="544:7:7" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SQuAD`</SwmToken> dataset.

```python
class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="318">

---

# <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken> Function

The <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken> function converts a list of examples into a list of features that can be directly used as input to a model. It takes advantage of many of the tokenizer's features to create the model's inputs.

```python
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="759">

---

# <SwmToken path="src/transformers/data/processors/squad.py" pos="759:2:2" line-data="class SquadFeatures:">`SquadFeatures`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/squad.py" pos="759:2:2" line-data="class SquadFeatures:">`SquadFeatures`</SwmToken> class represents the features of a single <SwmToken path="src/transformers/data/processors/squad.py" pos="761:3:3" line-data="    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from">`squad`</SwmToken> example, including token indices, attention masks, and other relevant information.

```python
class SquadFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/data/processors/squad.py" pos="44:2:2" line-data="def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):">`_improve_answer_span`</SwmToken>, <SwmToken path="src/transformers/data/processors/squad.py" pos="105:2:2" line-data="def squad_convert_example_to_features(">`squad_convert_example_to_features`</SwmToken>, and <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/data/processors/squad.py" pos="105:2:2" line-data="def squad_convert_example_to_features(">`squad_convert_example_to_features`</SwmToken> and <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken>.

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="44">

---

## <SwmToken path="src/transformers/data/processors/squad.py" pos="44:2:2" line-data="def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):">`_improve_answer_span`</SwmToken>

The <SwmToken path="src/transformers/data/processors/squad.py" pos="44:2:2" line-data="def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):">`_improve_answer_span`</SwmToken> function returns tokenized answer spans that better match the annotated answer. It iterates through possible new start and end positions to find a span that matches the tokenized answer text.

```python
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="105">

---

## <SwmToken path="src/transformers/data/processors/squad.py" pos="105:2:2" line-data="def squad_convert_example_to_features(">`squad_convert_example_to_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/squad.py" pos="105:2:2" line-data="def squad_convert_example_to_features(">`squad_convert_example_to_features`</SwmToken> function converts a single <SwmToken path="src/transformers/data/processors/squad.py" pos="335:17:17" line-data="        examples: list of :class:`~transformers.data.processors.squad.SquadExample`">`squad`</SwmToken> example into a set of features that can be directly used as input to a model. It handles tokenization, span truncation, and the creation of various masks and indices required for model input.

```python
def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="318">

---

## <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/squad.py" pos="318:2:2" line-data="def squad_convert_examples_to_features(">`squad_convert_examples_to_features`</SwmToken> function converts a list of <SwmToken path="src/transformers/data/processors/squad.py" pos="335:17:17" line-data="        examples: list of :class:`~transformers.data.processors.squad.SquadExample`">`squad`</SwmToken> examples into a list of features. It uses multiprocessing to speed up the conversion process and can return the features in a format suitable for either <SwmToken path="src/transformers/data/processors/squad.py" pos="404:6:6" line-data="            raise RuntimeError(&quot;PyTorch must be installed to return a PyTorch dataset.&quot;)">`PyTorch`</SwmToken> or <SwmToken path="src/transformers/data/processors/squad.py" pos="39:3:3" line-data="    import tensorflow as tf">`tensorflow`</SwmToken>.

```python
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
