---
title: Introduction to Model Management in Commands
---
Model Management involves handling various aspects of machine learning models, including loading, saving, and updating models.

The base classes `PreTrainedModel`, `TFPreTrainedModel`, and `FlaxPreTrainedModel` implement common methods for loading and saving models from local files or directories, as well as from pretrained model configurations provided by the library.

These base classes also support resizing input token embeddings when new tokens are added to the vocabulary and pruning attention heads in the models.

The `add_new_model_like` command facilitates adding new models by automating the process of updating relevant mappings, documentation, and initialization files.

Functions like <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="1045:2:2" line-data="def add_model_to_auto_classes(">`add_model_to_auto_classes`</SwmToken>, <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="983:2:2" line-data="def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):">`insert_tokenizer_in_auto_module`</SwmToken>, and <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="894:2:2" line-data="def add_model_to_main_init(">`add_model_to_main_init`</SwmToken> are used to update the auto module mappings, tokenizer mappings, and main initialization files respectively.

The <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="895:4:4" line-data="    old_model_patterns: ModelPatterns,">`ModelPatterns`</SwmToken> class holds essential information about a new model, such as its name, type, and associated classes, which are used throughout the model management process.

# Base Classes

The base classes `PreTrainedModel`, `TFPreTrainedModel`, and `FlaxPreTrainedModel` implement common methods for loading and saving models from local files or directories, as well as from pretrained model configurations provided by the library.

<SwmSnippet path="/src/transformers/commands/add_new_model_like.py" line="1045">

---

# Adding New Models

The function <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="1045:2:2" line-data="def add_model_to_auto_classes(">`add_model_to_auto_classes`</SwmToken> is used to add a model to the relevant mappings in the auto module.

```python
def add_model_to_auto_classes(
    old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, model_classes: Dict[str, List[str]]
):
    """
    Add a model to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`Dict[str, List[str]]`): A dictionary framework to list of model classes implemented.
    """
    for filename in AUTO_CLASSES_PATTERNS:
        # Extend patterns with all model classes if necessary
        new_patterns = []
        for pattern in AUTO_CLASSES_PATTERNS[filename]:
            if re.search("any_([a-z]*)_class", pattern) is not None:
                framework = re.search("any_([a-z]*)_class", pattern).groups()[0]
                if framework in model_classes:
                    new_patterns.extend(
                        [
                            pattern.replace("{" + f"any_{framework}_class" + "}", cls)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/add_new_model_like.py" line="983">

---

# Tokenizer Mappings

The function <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="983:2:2" line-data="def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):">`insert_tokenizer_in_auto_module`</SwmToken> adds a tokenizer to the relevant mappings in the auto module.

```python
def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Add a tokenizer to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    if old_model_patterns.tokenizer_class is None or new_model_patterns.tokenizer_class is None:
        return

    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    idx = 0
    # First we get to the TOKENIZER_MAPPING_NAMES block.
    while not lines[idx].startswith("    TOKENIZER_MAPPING_NAMES = OrderedDict("):
        idx += 1
    idx += 1

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/add_new_model_like.py" line="894">

---

# Main Initialization

The function <SwmToken path="src/transformers/commands/add_new_model_like.py" pos="894:2:2" line-data="def add_model_to_main_init(">`add_model_to_main_init`</SwmToken> adds a model to the main init of Transformers.

```python
def add_model_to_main_init(
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    frameworks: Optional[List[str]] = None,
    with_processing: bool = True,
):
    """
    Add a model to the main init of Transformers.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        frameworks (`List[str]`, *optional*):
            If specified, only the models implemented in those frameworks will be added.
        with_processsing (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer/feature extractor/processor of the model should also be added to the init or not.
    """
    with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
