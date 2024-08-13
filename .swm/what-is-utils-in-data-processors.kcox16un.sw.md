---
title: What is Utils in Data Processors
---
Utils in Data Processors provide utility functions and classes to facilitate the handling and processing of data for sequence classification tasks.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class serves as a base class for data converters, defining methods that need to be implemented for specific datasets, such as getting training, development, and test examples, as well as labels.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken> class extends <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> to handle single sentence classification datasets, providing methods to create processors from CSV files and add examples from various sources.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="31:2:2" line-data="class InputExample:">`InputExample`</SwmToken> class represents a single training or test example, including attributes like unique ID, text sequences, and labels.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="56:2:2" line-data="class InputFeatures:">`InputFeatures`</SwmToken> class represents a set of features for data, including input <SwmToken path="src/transformers/data/processors/utils.py" pos="182:1:1" line-data="        ids = []">`ids`</SwmToken>, attention masks, token type <SwmToken path="src/transformers/data/processors/utils.py" pos="182:1:1" line-data="        ids = []">`ids`</SwmToken>, and labels, which are used as inputs to a model.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken> method in <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken> converts examples into a list of <SwmToken path="src/transformers/data/processors/utils.py" pos="56:2:2" line-data="class InputFeatures:">`InputFeatures`</SwmToken>, tokenizing the text and preparing the data for model input.

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="81">

---

# <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class serves as a base class for data converters, defining methods that need to be implemented for specific datasets, such as getting training, development, and test examples, as well as labels.

```python
class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="126">

---

# <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken> class extends <SwmToken path="src/transformers/data/processors/utils.py" pos="126:4:4" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`DataProcessor`</SwmToken> to handle single sentence classification datasets, providing methods to create processors from CSV files and add examples from various sources.

```python
class SingleSentenceClassificationProcessor(DataProcessor):
    """Generic processor for a single sentence classification data set."""

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="31">

---

# <SwmToken path="src/transformers/data/processors/utils.py" pos="31:2:2" line-data="class InputExample:">`InputExample`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/utils.py" pos="31:2:2" line-data="class InputExample:">`InputExample`</SwmToken> class represents a single training or test example, including attributes like unique ID, text sequences, and labels.

```python
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="56">

---

# <SwmToken path="src/transformers/data/processors/utils.py" pos="56:2:2" line-data="class InputFeatures:">`InputFeatures`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/utils.py" pos="56:2:2" line-data="class InputFeatures:">`InputFeatures`</SwmToken> class represents a set of features for data, including input <SwmToken path="src/transformers/data/processors/utils.py" pos="182:1:1" line-data="        ids = []">`ids`</SwmToken>, attention masks, token type <SwmToken path="src/transformers/data/processors/utils.py" pos="182:1:1" line-data="        ids = []">`ids`</SwmToken>, and labels, which are used as inputs to a model.

```python
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="233">

---

# <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken> Method

The <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken> method in <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken> converts examples into a list of <SwmToken path="src/transformers/data/processors/utils.py" pos="243:14:14" line-data="        Convert examples in a list of ``InputFeatures``">`InputFeatures`</SwmToken>, tokenizing the text and preparing the data for model input.

```python
    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/data/processors/utils.py" pos="94:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="98:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="102:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="106:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="144:3:3" line-data="    def create_from_csv(">`create_from_csv`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="148:3:3" line-data="        processor.add_examples_from_csv(">`add_examples_from_csv`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="196:3:3" line-data="    def add_examples(">`add_examples`</SwmToken>, and <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/data/processors/utils.py" pos="94:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="98:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken>, <SwmToken path="src/transformers/data/processors/utils.py" pos="102:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken>, and <SwmToken path="src/transformers/data/processors/utils.py" pos="106:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken>.

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="94">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="94:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="94:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken> function is used to get a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="95:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the train set.&quot;&quot;&quot;">`InputExample`</SwmToken> for the training set.

```python
    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="98">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="98:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="98:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken> function is used to get a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="99:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the dev set.&quot;&quot;&quot;">`InputExample`</SwmToken> for the development set.

```python
    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="102">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="102:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="102:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken> function is used to get a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="103:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the test set.&quot;&quot;&quot;">`InputExample`</SwmToken> for the test set.

```python
    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="106">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="106:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="106:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken> function is used to get the list of labels for the dataset.

```python
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="144">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="144:3:3" line-data="    def create_from_csv(">`create_from_csv`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="144:3:3" line-data="    def create_from_csv(">`create_from_csv`</SwmToken> function creates a processor from a CSV file by calling <SwmToken path="src/transformers/data/processors/utils.py" pos="148:3:3" line-data="        processor.add_examples_from_csv(">`add_examples_from_csv`</SwmToken>.

```python
    def create_from_csv(
        cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="166">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="166:3:3" line-data="    def add_examples_from_csv(">`add_examples_from_csv`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="166:3:3" line-data="    def add_examples_from_csv(">`add_examples_from_csv`</SwmToken> function reads a TSV file and adds examples to the processor.

```python
    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for (i, line) in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="196">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="196:3:3" line-data="    def add_examples(">`add_examples`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="196:3:3" line-data="    def add_examples(">`add_examples`</SwmToken> function adds examples to the processor, updating the examples and labels.

```python
    def add_examples(
        self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        assert labels is None or len(texts_or_text_and_labels) == len(
            labels
        ), f"Text and labels have mismatched lengths {len(texts_or_text_and_labels)} and {len(labels)}"
        assert ids is None or len(texts_or_text_and_labels) == len(
            ids
        ), f"Text and ids have mismatched lengths {len(texts_or_text_and_labels)} and {len(ids)}"
        if ids is None:
            ids = [None] * len(texts_or_text_and_labels)
        if labels is None:
            labels = [None] * len(texts_or_text_and_labels)
        examples = []
        added_labels = set()
        for (text_or_text_and_label, label, guid) in zip(texts_or_text_and_labels, labels, ids):
            if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
                text, label = text_or_text_and_label
            else:
                text = text_or_text_and_label
            added_labels.add(label)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="233">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="233:3:3" line-data="    def get_features(">`get_features`</SwmToken> function converts examples into a list of <SwmToken path="src/transformers/data/processors/utils.py" pos="243:14:14" line-data="        Convert examples in a list of ``InputFeatures``">`InputFeatures`</SwmToken>, tokenizing the text and preparing the data for model input.

```python
    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
