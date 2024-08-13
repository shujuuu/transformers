---
title: Introduction to GLUE Data Processors
---
Glue refers to a set of data processors used for handling various datasets in the General Language Understanding Evaluation (GLUE) benchmark.

These processors are responsible for converting raw data into a format that can be used by machine learning models.

Each processor corresponds to a specific GLUE task, such as <SwmToken path="src/transformers/data/processors/glue.py" pos="621:2:2" line-data="    &quot;cola&quot;: ColaProcessor,">`cola`</SwmToken>, MNLI, MRPC, <SwmToken path="src/transformers/data/processors/glue.py" pos="625:2:4" line-data="    &quot;sst-2&quot;: Sst2Processor,">`sst-2`</SwmToken>, <SwmToken path="src/transformers/data/processors/glue.py" pos="93:16:18" line-data="        label_type = tf.float32 if task == &quot;sts-b&quot; else tf.int64">`sts-b`</SwmToken>, QQP, QNLI, RTE, and WNLI.

The processors handle tasks like reading data from files, tokenizing text, and mapping labels to their respective indices.

For example, the <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> handles the MRPC dataset, converting raw text into <SwmToken path="src/transformers/data/processors/glue.py" pos="43:8:8" line-data="    examples: Union[List[InputExample], &quot;tf.data.Dataset&quot;],">`InputExample`</SwmToken> objects that can be fed into a model.

The <SwmToken path="src/transformers/data/processors/glue.py" pos="90:5:5" line-data="        processor = glue_processors[task]()">`glue_processors`</SwmToken> dictionary maps each GLUE task to its corresponding processor class, facilitating easy access and usage.

The processors also include methods for obtaining training, development, and test examples, as well as the labels for each task.

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="170">

---

<SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> The <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> handles the MRPC dataset, converting raw text into <SwmToken path="src/transformers/data/processors/glue.py" pos="179:3:3" line-data="        return InputExample(">`InputExample`</SwmToken> objects that can be fed into a model.

```python
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="620">

---

<SwmToken path="src/transformers/data/processors/glue.py" pos="620:0:0" line-data="glue_processors = {">`glue_processors`</SwmToken> The <SwmToken path="src/transformers/data/processors/glue.py" pos="620:0:0" line-data="glue_processors = {">`glue_processors`</SwmToken> dictionary maps each GLUE task to its corresponding processor class, facilitating easy access and usage.

```python
glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="186">

---

Processor Methods The processors also include methods for obtaining training, development, and test examples, as well as the labels for each task.

```python
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/data/processors/glue.py" pos="42:2:2" line-data="def glue_convert_examples_to_features(">`glue_convert_examples_to_features`</SwmToken>, <SwmToken path="src/transformers/data/processors/glue.py" pos="79:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken>, and <SwmToken path="src/transformers/data/processors/glue.py" pos="110:2:2" line-data="def _glue_convert_examples_to_features(">`_glue_convert_examples_to_features`</SwmToken>. We will dive a little into these functions.

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="42">

---

### <SwmToken path="src/transformers/data/processors/glue.py" pos="42:2:2" line-data="def glue_convert_examples_to_features(">`glue_convert_examples_to_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/glue.py" pos="42:2:2" line-data="def glue_convert_examples_to_features(">`glue_convert_examples_to_features`</SwmToken> function loads a data file into a list of <SwmToken path="src/transformers/data/processors/glue.py" pos="51:18:18" line-data="    Loads a data file into a list of ``InputFeatures``">`InputFeatures`</SwmToken>. It takes examples, a tokenizer, and optional parameters like <SwmToken path="src/transformers/data/processors/glue.py" pos="45:1:1" line-data="    max_length: Optional[int] = None,">`max_length`</SwmToken>, task, <SwmToken path="src/transformers/data/processors/glue.py" pos="47:1:1" line-data="    label_list=None,">`label_list`</SwmToken>, and <SwmToken path="src/transformers/data/processors/glue.py" pos="48:1:1" line-data="    output_mode=None,">`output_mode`</SwmToken>. It returns a <SwmToken path="src/transformers/data/processors/glue.py" pos="43:13:17" line-data="    examples: Union[List[InputExample], &quot;tf.data.Dataset&quot;],">`tf.data.Dataset`</SwmToken> or a list of <SwmToken path="src/transformers/data/processors/glue.py" pos="51:18:18" line-data="    Loads a data file into a list of ``InputFeatures``">`InputFeatures`</SwmToken> depending on the input type.

```python
def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="79">

---

### <SwmToken path="src/transformers/data/processors/glue.py" pos="79:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/glue.py" pos="79:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken> function converts <SwmToken path="src/transformers/data/processors/glue.py" pos="31:3:3" line-data="    import tensorflow as tf">`tensorflow`</SwmToken> datasets into <SwmToken path="src/transformers/data/processors/glue.py" pos="87:15:17" line-data="            A ``tf.data.Dataset`` containing the task-specific features.">`task-specific`</SwmToken> features. It uses the <SwmToken path="src/transformers/data/processors/glue.py" pos="92:5:5" line-data="        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)">`glue_convert_examples_to_features`</SwmToken> function internally and returns a <SwmToken path="src/transformers/data/processors/glue.py" pos="80:4:8" line-data="        examples: tf.data.Dataset,">`tf.data.Dataset`</SwmToken> containing the features.

```python
    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        task=str,
        max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
        label_type = tf.float32 if task == "sts-b" else tf.int64

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="110">

---

### <SwmToken path="src/transformers/data/processors/glue.py" pos="110:2:2" line-data="def _glue_convert_examples_to_features(">`_glue_convert_examples_to_features`</SwmToken>

The <SwmToken path="src/transformers/data/processors/glue.py" pos="110:2:2" line-data="def _glue_convert_examples_to_features(">`_glue_convert_examples_to_features`</SwmToken> function converts a list of <SwmToken path="src/transformers/data/processors/glue.py" pos="111:6:6" line-data="    examples: List[InputExample],">`InputExample`</SwmToken> objects into <SwmToken path="src/transformers/data/processors/glue.py" pos="51:18:18" line-data="    Loads a data file into a list of ``InputFeatures``">`InputFeatures`</SwmToken>. It handles tasks like tokenizing text, mapping labels, and creating feature objects that can be fed into a model.

```python
def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_map = {label: i for i, label in enumerate(label_list)}
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
