---
title: Exploring Data Processing in Transformers
---
Data processing involves preparing and transforming raw data into a format suitable for training machine learning models.

The <SwmToken path="src/transformers/data/processors/utils.py" pos="80:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class serves as a base class for data converters for sequence classification datasets.

Specific processors like <SwmToken path="src/transformers/data/processors/glue.py" pos="278:2:2" line-data="class ColaProcessor(DataProcessor):">`ColaProcessor`</SwmToken>, <SwmToken path="src/transformers/data/processors/glue.py" pos="216:2:2" line-data="class MnliProcessor(DataProcessor):">`MnliProcessor`</SwmToken>, and <SwmToken path="src/transformers/data/processors/glue.py" pos="417:2:2" line-data="class QqpProcessor(DataProcessor):">`QqpProcessor`</SwmToken> inherit from <SwmToken path="src/transformers/data/processors/utils.py" pos="80:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> and implement methods to read and process data from various sources.

These processors handle tasks such as reading data from files, converting data into examples, and mapping labels to the appropriate format.

The function <SwmToken path="src/transformers/data/processors/glue.py" pos="78:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken> converts raw data examples into features that can be fed into a model for training.

This function utilizes the specific processors to map and format the data correctly, ensuring compatibility with the model's requirements.

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="80">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="80:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> Class

The <SwmToken path="src/transformers/data/processors/utils.py" pos="80:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class serves as a base class for data converters for sequence classification datasets. It defines methods for reading data, converting it into examples, and mapping labels.

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
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="78">

---

## <SwmToken path="src/transformers/data/processors/glue.py" pos="78:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken> Function

The <SwmToken path="src/transformers/data/processors/glue.py" pos="78:3:3" line-data="    def _tf_glue_convert_examples_to_features(">`_tf_glue_convert_examples_to_features`</SwmToken> function converts raw data examples into features that can be fed into a model for training. It utilizes specific processors to map and format the data correctly, ensuring compatibility with the model's requirements.

```python
    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        task=str,
        max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A `tf.data.Dataset` containing the task-specific features.

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

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="109">

---

## <SwmToken path="src/transformers/data/processors/utils.py" pos="109:3:3" line-data="    def tfds_map(self, example):">`tfds_map`</SwmToken> Method

The <SwmToken path="src/transformers/data/processors/utils.py" pos="109:3:3" line-data="    def tfds_map(self, example):">`tfds_map`</SwmToken> method in the <SwmToken path="src/transformers/data/processors/utils.py" pos="80:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class converts examples to the correct format, especially for datasets that are not formatted the same way as the GLUE datasets.

```python
    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="176">

---

## <SwmToken path="src/transformers/data/processors/glue.py" pos="176:3:3" line-data="    def get_example_from_tensor_dict(self, tensor_dict):">`get_example_from_tensor_dict`</SwmToken> Method

The <SwmToken path="src/transformers/data/processors/glue.py" pos="176:3:3" line-data="    def get_example_from_tensor_dict(self, tensor_dict):">`get_example_from_tensor_dict`</SwmToken> method retrieves an example from a dictionary with <SwmToken path="src/transformers/data/processors/utils.py" pos="85:15:15" line-data="        Gets an example from a dict with tensorflow tensors.">`tensorflow`</SwmToken> tensors, ensuring that the keys and values match the corresponding dataset examples.

```python
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
