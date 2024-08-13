---
title: Getting started with Data Collator in Data Processing
---
A Data Collator is a function that takes a list of samples from a dataset and collates them into a batch, represented as a dictionary of tensors.

It ensures that the data is properly formatted and padded to the same length, which is essential for efficient batch processing in machine learning models.

Data Collators can handle various types of data, including labels and label IDs, and perform special handling for these keys to ensure they are correctly processed.

Different types of Data Collators are available, such as <SwmToken path="src/transformers/data/data_collator.py" pos="87:2:2" line-data="class DataCollatorWithPadding:">`DataCollatorWithPadding`</SwmToken>, <SwmToken path="src/transformers/data/data_collator.py" pos="136:2:2" line-data="class DataCollatorForTokenClassification:">`DataCollatorForTokenClassification`</SwmToken>, and <SwmToken path="src/transformers/data/data_collator.py" pos="233:2:2" line-data="class DataCollatorForSeq2Seq:">`DataCollatorForSeq2Seq`</SwmToken>, each designed to handle specific data processing needs.

These collators dynamically pad the inputs and labels, making them suitable for tasks like token classification, sequence-to-sequence modeling, and language modeling.

<SwmSnippet path="/src/transformers/data/data_collator.py" line="87">

---

The <SwmToken path="src/transformers/data/data_collator.py" pos="87:2:2" line-data="class DataCollatorWithPadding:">`DataCollatorWithPadding`</SwmToken> class dynamically pads the inputs received. It uses a tokenizer to encode the data and provides options for padding strategy, maximum length, and padding to a multiple of a specified value.

```python
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="136">

---

The <SwmToken path="src/transformers/data/data_collator.py" pos="136:2:2" line-data="class DataCollatorForTokenClassification:">`DataCollatorForTokenClassification`</SwmToken> class dynamically pads the inputs and labels received. It handles the padding of sequences and labels to ensure they are of the same length, which is crucial for token classification tasks.

```python
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="233">

---

The <SwmToken path="src/transformers/data/data_collator.py" pos="233:2:2" line-data="class DataCollatorForSeq2Seq:">`DataCollatorForSeq2Seq`</SwmToken> class dynamically pads the inputs and labels received. It is designed for sequence-to-sequence tasks and can prepare decoder input IDs if a model is provided.

```python
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
```

---

</SwmSnippet>

# Main functions

There are several main functions related to data collation. Some of them are <SwmToken path="src/transformers/data/data_collator.py" pos="38:2:2" line-data="def default_data_collator(features: List[InputDataClass]) -&gt; Dict[str, torch.Tensor]:">`default_data_collator`</SwmToken>, <SwmToken path="src/transformers/data/data_collator.py" pos="87:2:2" line-data="class DataCollatorWithPadding:">`DataCollatorWithPadding`</SwmToken>, <SwmToken path="src/transformers/data/data_collator.py" pos="136:2:2" line-data="class DataCollatorForTokenClassification:">`DataCollatorForTokenClassification`</SwmToken>, <SwmToken path="src/transformers/data/data_collator.py" pos="233:2:2" line-data="class DataCollatorForSeq2Seq:">`DataCollatorForSeq2Seq`</SwmToken>, and <SwmToken path="src/transformers/data/data_collator.py" pos="303:2:2" line-data="class DataCollatorForLanguageModeling:">`DataCollatorForLanguageModeling`</SwmToken>. We will dive a little into each of these.

<SwmSnippet path="/src/transformers/data/data_collator.py" line="38">

---

### <SwmToken path="src/transformers/data/data_collator.py" pos="38:2:2" line-data="def default_data_collator(features: List[InputDataClass]) -&gt; Dict[str, torch.Tensor]:">`default_data_collator`</SwmToken>

The <SwmToken path="src/transformers/data/data_collator.py" pos="38:2:2" line-data="def default_data_collator(features: List[InputDataClass]) -&gt; Dict[str, torch.Tensor]:">`default_data_collator`</SwmToken> function collates batches of <SwmToken path="src/transformers/data/data_collator.py" pos="40:19:21" line-data="    Very simple data collator that simply collates batches of dict-like objects and performs special handling for">`dict-like`</SwmToken> objects and performs special handling for keys like <SwmToken path="src/transformers/data/data_collator.py" pos="43:4:4" line-data="        - ``label``: handles a single value (int or float) per object">`label`</SwmToken> and <SwmToken path="src/transformers/data/data_collator.py" pos="44:4:4" line-data="        - ``label_ids``: handles a list of values per object">`label_ids`</SwmToken>. It ensures that the tensor is created with the correct type and handles all other possible keys by stacking or tensorizing them.

```python
def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="87">

---

### <SwmToken path="src/transformers/data/data_collator.py" pos="87:2:2" line-data="class DataCollatorWithPadding:">`DataCollatorWithPadding`</SwmToken>

The <SwmToken path="src/transformers/data/data_collator.py" pos="87:2:2" line-data="class DataCollatorWithPadding:">`DataCollatorWithPadding`</SwmToken> class dynamically pads the inputs received. It uses a tokenizer to pad the sequences according to the specified padding strategy, maximum length, and padding to a multiple of a given value. It also handles labels by renaming them appropriately.

```python
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="136">

---

### <SwmToken path="src/transformers/data/data_collator.py" pos="136:2:2" line-data="class DataCollatorForTokenClassification:">`DataCollatorForTokenClassification`</SwmToken>

The <SwmToken path="src/transformers/data/data_collator.py" pos="136:2:2" line-data="class DataCollatorForTokenClassification:">`DataCollatorForTokenClassification`</SwmToken> class dynamically pads the inputs and labels received. It uses a tokenizer to pad the sequences and handles labels by padding them to the same length as the input sequences. The labels are then converted to tensors.

```python
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="233">

---

### <SwmToken path="src/transformers/data/data_collator.py" pos="233:2:2" line-data="class DataCollatorForSeq2Seq:">`DataCollatorForSeq2Seq`</SwmToken>

The <SwmToken path="src/transformers/data/data_collator.py" pos="233:2:2" line-data="class DataCollatorForSeq2Seq:">`DataCollatorForSeq2Seq`</SwmToken> class dynamically pads the inputs and labels received. It uses a tokenizer to pad the sequences and handles labels by padding them before calling the tokenizer's pad method. It also prepares <SwmToken path="src/transformers/data/data_collator.py" pos="242:6:6" line-data="            prepare the `decoder_input_ids`">`decoder_input_ids`</SwmToken> if the model has the <SwmToken path="src/transformers/data/data_collator.py" pos="241:25:25" line-data="            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to">`prepare_decoder_input_ids_from_labels`</SwmToken> method.

```python
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/data_collator.py" line="303">

---

### <SwmToken path="src/transformers/data/data_collator.py" pos="303:2:2" line-data="class DataCollatorForLanguageModeling:">`DataCollatorForLanguageModeling`</SwmToken>

The <SwmToken path="src/transformers/data/data_collator.py" pos="303:2:2" line-data="class DataCollatorForLanguageModeling:">`DataCollatorForLanguageModeling`</SwmToken> class is used for language modeling. It dynamically pads the inputs to the maximum length of a batch and handles masked language modeling by masking tokens with a specified probability. It also handles special token masks and converts inputs to tensors.

```python
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
