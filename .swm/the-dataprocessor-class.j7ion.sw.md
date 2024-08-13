---
title: The DataProcessor class
---
This document will cover the <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> class in the <SwmPath>[examples/legacy/multiple_choice/utils_multiple_choice.py](examples/legacy/multiple_choice/utils_multiple_choice.py)</SwmPath> file. We will cover:

1. What <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> is.
2. Variables and functions in <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken>.
3. Usage example of <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> in <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:2:2" line-data="class ArcProcessor(DataProcessor):">`ArcProcessor`</SwmToken>.

```mermaid
graph TD;
  DataProcessor:::currentBaseStyle
DataProcessor --> RaceProcessor
DataProcessor --> SynonymProcessor
DataProcessor --> SwagProcessor
DataProcessor --> ArcProcessor
DataProcessor --> HansProcessor

 classDef currentBaseStyle color:#000000,fill:#7CB9F4

%% Swimm:
%% graph TD;
%%   DataProcessor:::currentBaseStyle
%% <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> --> <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="253:2:2" line-data="class RaceProcessor(DataProcessor):">`RaceProcessor`</SwmToken>
%% <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> --> <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="320:2:2" line-data="class SynonymProcessor(DataProcessor):">`SynonymProcessor`</SwmToken>
%% <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> --> <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="366:2:2" line-data="class SwagProcessor(DataProcessor):">`SwagProcessor`</SwmToken>
%% <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> --> <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:2:2" line-data="class ArcProcessor(DataProcessor):">`ArcProcessor`</SwmToken>
%% <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> --> HansProcessor
%% 
%%  classDef currentBaseStyle color:#000000,fill:#7CB9F4
```

# What is <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken>

The <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> class in <SwmPath>[examples/legacy/multiple_choice/utils_multiple_choice.py](examples/legacy/multiple_choice/utils_multiple_choice.py)</SwmPath> is a base class for data converters for multiple choice data sets. It provides an interface for loading and processing data for training, validation, and testing in multiple choice tasks. The class defines methods that need to be implemented by subclasses to handle specific datasets.

<SwmSnippet path="/examples/legacy/multiple_choice/utils_multiple_choice.py" line="236">

---

# Variables and functions

The function <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="236:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken> is used to get a collection of <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="237:13:13" line-data="        &quot;&quot;&quot;Gets a collection of `InputExample`s for the train set.&quot;&quot;&quot;">`InputExample`</SwmToken>s for the train set. It needs to be implemented by subclasses.

```python
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/examples/legacy/multiple_choice/utils_multiple_choice.py" line="240">

---

The function <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="240:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken> is used to get a collection of <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="241:13:13" line-data="        &quot;&quot;&quot;Gets a collection of `InputExample`s for the dev set.&quot;&quot;&quot;">`InputExample`</SwmToken>s for the dev set. It needs to be implemented by subclasses.

```python
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/examples/legacy/multiple_choice/utils_multiple_choice.py" line="244">

---

The function <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="244:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken> is used to get a collection of <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="245:13:13" line-data="        &quot;&quot;&quot;Gets a collection of `InputExample`s for the test set.&quot;&quot;&quot;">`InputExample`</SwmToken>s for the test set. It needs to be implemented by subclasses.

```python
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

<SwmSnippet path="/examples/legacy/multiple_choice/utils_multiple_choice.py" line="248">

---

The function <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="248:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken> is used to get the list of labels for the dataset. It needs to be implemented by subclasses.

```python
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
```

---

</SwmSnippet>

# Usage example

The <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:2:2" line-data="class ArcProcessor(DataProcessor):">`ArcProcessor`</SwmToken> class is an example of a subclass that implements the <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> interface. Here is how <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> is used in <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:2:2" line-data="class ArcProcessor(DataProcessor):">`ArcProcessor`</SwmToken>.

<SwmSnippet path="/examples/legacy/multiple_choice/utils_multiple_choice.py" line="417">

---

# Usage example

The <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:2:2" line-data="class ArcProcessor(DataProcessor):">`ArcProcessor`</SwmToken> class extends <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="417:4:4" line-data="class ArcProcessor(DataProcessor):">`DataProcessor`</SwmToken> and implements the methods <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="420:3:3" line-data="    def get_train_examples(self, data_dir):">`get_train_examples`</SwmToken>, <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="425:3:3" line-data="    def get_dev_examples(self, data_dir):">`get_dev_examples`</SwmToken>, <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="430:3:3" line-data="    def get_test_examples(self, data_dir):">`get_test_examples`</SwmToken>, and <SwmToken path="examples/legacy/multiple_choice/utils_multiple_choice.py" pos="434:3:3" line-data="    def get_labels(self):">`get_labels`</SwmToken> to handle the ARC dataset.

```python
class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
