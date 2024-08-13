---
title: The DataProcessor class
---
This document will cover the <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class. We will discuss:

1. What <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> is.
2. The variables and functions defined in <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken>.
3. An example of how to use <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> in <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken>.

```mermaid
graph TD;
  DataProcessor:::currentBaseStyle
DataProcessor --> SquadProcessor
SquadProcessor --> SquadV1Processor
SquadProcessor --> SquadV2Processor
DataProcessor --> MnliProcessor
MnliProcessor --> MnliMismatchedProcessor
DataProcessor --> SingleSentenceClassificationProcessor
DataProcessor --> 1[...]

 classDef currentBaseStyle color:#000000,fill:#7CB9F4

%% Swimm:
%% graph TD;
%%   DataProcessor:::currentBaseStyle
%% <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> --> SquadProcessor
%% SquadProcessor --> SquadV1Processor
%% SquadProcessor --> SquadV2Processor
%% <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> --> <SwmToken path="src/transformers/data/processors/glue.py" pos="217:2:2" line-data="class MnliProcessor(DataProcessor):">`MnliProcessor`</SwmToken>
%% <SwmToken path="src/transformers/data/processors/glue.py" pos="217:2:2" line-data="class MnliProcessor(DataProcessor):">`MnliProcessor`</SwmToken> --> <SwmToken path="src/transformers/data/processors/glue.py" pos="263:2:2" line-data="class MnliMismatchedProcessor(MnliProcessor):">`MnliMismatchedProcessor`</SwmToken>
%% <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> --> <SwmToken path="src/transformers/data/processors/utils.py" pos="126:2:2" line-data="class SingleSentenceClassificationProcessor(DataProcessor):">`SingleSentenceClassificationProcessor`</SwmToken>
%% <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> --> 1[...]
%% 
%%  classDef currentBaseStyle color:#000000,fill:#7CB9F4
```

# What is <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken>

The <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class in <SwmPath>[src/transformers/data/processors/utils.py](src/transformers/data/processors/utils.py)</SwmPath> is a base class for data converters for sequence classification datasets. It provides a standard interface for loading and processing data, which can then be used for training and evaluating machine learning models. The class is designed to be extended by specific dataset processors that implement the required methods.

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="84">

---

# Variables and functions

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="84:3:3" line-data="    def get_example_from_tensor_dict(self, tensor_dict):">`get_example_from_tensor_dict`</SwmToken> is intended to get an example from a dictionary containing <SwmToken path="src/transformers/data/processors/utils.py" pos="86:15:15" line-data="        Gets an example from a dict with tensorflow tensors.">`tensorflow`</SwmToken> tensors. This function needs to be implemented by subclasses.

```python
    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="94">

---

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="94:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken> is intended to return a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="95:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the train set.&quot;&quot;&quot;">`InputExample`</SwmToken> instances for the training set. This function needs to be implemented by subclasses.

```python
    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="98">

---

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="98:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken> is intended to return a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="99:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the dev set.&quot;&quot;&quot;">`InputExample`</SwmToken> instances for the development set. This function needs to be implemented by subclasses.

```python
    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="102">

---

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="102:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken> is intended to return a collection of <SwmToken path="src/transformers/data/processors/utils.py" pos="103:15:15" line-data="        &quot;&quot;&quot;Gets a collection of :class:`InputExample` for the test set.&quot;&quot;&quot;">`InputExample`</SwmToken> instances for the test set. This function needs to be implemented by subclasses.

```python
    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="106">

---

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="106:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken> is intended to return the list of labels for the dataset. This function needs to be implemented by subclasses.

```python
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="110">

---

The function <SwmToken path="src/transformers/data/processors/utils.py" pos="110:3:3" line-data="    def tfds_map(self, example):">`tfds_map`</SwmToken> is used to convert examples from <SwmToken path="src/transformers/data/processors/utils.py" pos="86:15:15" line-data="        Gets an example from a dict with tensorflow tensors.">`tensorflow`</SwmToken> datasets to the correct format expected by the GLUE datasets. It modifies the label of the example if necessary.

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

<SwmSnippet path="/src/transformers/data/processors/utils.py" line="119">

---

The class method <SwmToken path="src/transformers/data/processors/utils.py" pos="120:3:3" line-data="    def _read_tsv(cls, input_file, quotechar=None):">`_read_tsv`</SwmToken> reads a tab-separated value (TSV) file and returns its content as a list of rows.

```python
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
```

---

</SwmSnippet>

# Usage example

The <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> class in <SwmPath>[src/transformers/data/processors/glue.py](src/transformers/data/processors/glue.py)</SwmPath> is an example of how to use the <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> class. It extends <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> and implements the required methods to process the MRPC dataset.

<SwmSnippet path="/src/transformers/data/processors/glue.py" line="90">

---

# Usage example

In the <SwmToken path="src/transformers/data/processors/glue.py" pos="170:2:2" line-data="class MrpcProcessor(DataProcessor):">`MrpcProcessor`</SwmToken> class, the <SwmToken path="src/transformers/data/processors/glue.py" pos="91:8:8" line-data="        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]">`tfds_map`</SwmToken> function from <SwmToken path="src/transformers/data/processors/utils.py" pos="81:2:2" line-data="class DataProcessor:">`DataProcessor`</SwmToken> is used to convert examples from <SwmToken path="src/transformers/data/processors/utils.py" pos="86:15:15" line-data="        Gets an example from a dict with tensorflow tensors.">`tensorflow`</SwmToken> datasets to the correct format.

```python
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
