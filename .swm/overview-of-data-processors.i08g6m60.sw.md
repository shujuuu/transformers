---
title: Overview of Data Processors
---
Data Processors are classes designed to handle the preprocessing of datasets for various NLP tasks.

They convert raw data into a format that can be fed into machine learning models for training and evaluation.

Each Data Processor is tailored to a specific dataset and task, ensuring that the data is correctly formatted and labeled.

For example, the <SwmToken path="src/transformers/data/processors/glue.py" pos="562:2:2" line-data="class WnliProcessor(DataProcessor):">`WnliProcessor`</SwmToken> handles the WNLI dataset, converting raw text and labels into <SwmToken path="src/transformers/data/processors/glue.py" pos="179:3:3" line-data="        return InputExample(">`InputExample`</SwmToken> objects.

Data Processors also provide methods to read data from files, create examples for training, development, and testing, and retrieve the list of labels for the dataset.

They play a crucial role in the data pipeline, ensuring that the data is in the right shape and format before being passed to the model.

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="562">

---

## Example: <SwmToken path="src/transformers/data/processors/glue.py" pos="562:2:2" line-data="class WnliProcessor(DataProcessor):">`WnliProcessor`</SwmToken>

The <SwmToken path="src/transformers/data/processors/glue.py" pos="562:2:2" line-data="class WnliProcessor(DataProcessor):">`WnliProcessor`</SwmToken> handles the WNLI dataset, converting raw text and labels into <SwmToken path="src/transformers/data/processors/glue.py" pos="571:3:3" line-data="        return InputExample(">`InputExample`</SwmToken> objects.

```python
class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

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
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="170">

---

## Example: <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken>

The <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> is another example of a Data Processor, tailored for the MRPC dataset.

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

<SwmSnippet path="/src/transformers/data/processors/squad.py" line="542">

---

## Example: <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken>

The <SwmToken path="src/transformers/data/processors/squad.py" pos="542:2:2" line-data="class SquadProcessor(DataProcessor):">`SquadProcessor`</SwmToken> is used for the <SwmToken path="src/transformers/data/processors/squad.py" pos="544:7:7" line-data="    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and">`SQuAD`</SwmToken> dataset, demonstrating how Data Processors can handle more complex data formats.

```python
class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
