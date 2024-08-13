---
title: Getting started with Language Modeling in Data Datasets
---
Language modeling involves creating datasets that are used to train models to predict the next word or sequence of words in a sentence.

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken> class reads a text file, tokenizes the text, and splits it into blocks of a specified size. These blocks are then used as training examples for language models.

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken> class reads a text file line by line, tokenizes each line, and creates training examples from these tokenized lines.

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken> class is similar to <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken>, but it also includes reference information from an additional file, which can be used for tasks like whole word masking.

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken> class is designed for the next sentence prediction task. It reads a text file, tokenizes the text, and creates pairs of sentences to be used as training examples.

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken> class is used for the sentence order prediction task. It prepares sentence pairs from documents to train models to predict the correct order of sentences.

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="41">

---

<SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken> The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken> class reads a text file, tokenizes the text, and splits it into blocks of a specified size. These blocks are then used as training examples for language models.

```python
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="114">

---

<SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken> The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken> class reads a text file line by line, tokenizes each line, and creates training examples from these tokenized lines.

```python
class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="146">

---

<SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken> The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken> class is similar to <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken>, but it also includes reference information from an additional file, which can be used for tasks like whole word masking.

```python
class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        assert os.path.isfile(ref_path), f"Ref file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="335">

---

<SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken> The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken> class is designed for the next sentence prediction task. It reads a text file, tokenizes the text, and creates pairs of sentences to be used as training examples.

```python
class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="188">

---

<SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken> The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken> class is used for the sentence order prediction task. It prepares sentence pairs from documents to train models to predict the correct order of sentences.

```python
class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken>, <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken>, <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken>, <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken>, and <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken>. We will dive a little into each of them.

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="41">

---

### <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken>

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="41:2:2" line-data="class TextDataset(Dataset):">`TextDataset`</SwmToken> class reads a text file, tokenizes the text, and splits it into blocks of a specified size. These blocks are then used as training examples for language models.

```python
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="114">

---

### <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken>

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken> class reads a text file line by line, tokenizes each line, and creates training examples from these tokenized lines.

```python
class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="146">

---

### <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken>

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="146:2:2" line-data="class LineByLineWithRefDataset(Dataset):">`LineByLineWithRefDataset`</SwmToken> class is similar to <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="114:2:2" line-data="class LineByLineTextDataset(Dataset):">`LineByLineTextDataset`</SwmToken>, but it also includes reference information from an additional file, which can be used for tasks like whole word masking.

```python
class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        assert os.path.isfile(ref_path), f"Ref file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="335">

---

### <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken>

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="335:2:2" line-data="class TextDatasetForNextSentencePrediction(Dataset):">`TextDatasetForNextSentencePrediction`</SwmToken> class is designed for the next sentence prediction task. It reads a text file, tokenizes the text, and creates pairs of sentences to be used as training examples.

```python
class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/data/datasets/language_modeling.py" line="188">

---

### <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken>

The <SwmToken path="src/transformers/data/datasets/language_modeling.py" pos="188:2:2" line-data="class LineByLineWithSOPTextDataset(Dataset):">`LineByLineWithSOPTextDataset`</SwmToken> class is used for the sentence order prediction task. It prepares sentence pairs from documents to train models to predict the correct order of sentences.

```python
class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
