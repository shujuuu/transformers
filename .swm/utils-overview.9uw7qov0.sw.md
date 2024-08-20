---
title: Utils Overview
---
Utils refers to a collection of utility functions and constants that assist in various tasks throughout the codebase.

These utilities include functions for checking the availability of specific libraries, handling environment variables, and providing error messages for missing dependencies.

For example, functions like <SwmToken path="src/transformers/utils/import_utils.py" pos="903:2:2" line-data="def is_sklearn_available():">`is_sklearn_available`</SwmToken> and <SwmToken path="src/transformers/utils/import_utils.py" pos="953:2:2" line-data="def is_levenshtein_available():">`is_levenshtein_available`</SwmToken> check if certain libraries are installed in the environment.

Constants such as <SwmToken path="src/transformers/utils/import_utils.py" pos="77:0:0" line-data="USE_TF = os.environ.get(&quot;USE_TF&quot;, &quot;AUTO&quot;).upper()">`USE_TF`</SwmToken> and <SwmToken path="src/transformers/utils/import_utils.py" pos="79:0:0" line-data="USE_JAX = os.environ.get(&quot;USE_FLAX&quot;, &quot;AUTO&quot;).upper()">`USE_JAX`</SwmToken> are used to determine which machine learning frameworks are available and should be used.

Error messages like <SwmToken path="src/transformers/utils/import_utils.py" pos="1168:0:0" line-data="AV_IMPORT_ERROR = &quot;&quot;&quot;">`AV_IMPORT_ERROR`</SwmToken> and <SwmToken path="src/transformers/utils/import_utils.py" pos="1178:0:0" line-data="CV2_IMPORT_ERROR = &quot;&quot;&quot;">`CV2_IMPORT_ERROR`</SwmToken> provide guidance on how to install missing libraries.

## Argument handling

The utils handle various arguments passed to functions, ensuring they are correctly formatted and validated.

## Data format

Utils also manage data formatting, converting data into the required formats for processing.

## Utilities

The utilities include a range of helper functions that streamline common tasks, such as checking library availability and managing environment variables.

<SwmSnippet path="/src/transformers/utils/import_utils.py" line="903">

---

The function <SwmToken path="src/transformers/utils/import_utils.py" pos="903:2:2" line-data="def is_sklearn_available():">`is_sklearn_available`</SwmToken> checks if the <SwmToken path="src/transformers/utils/import_utils.py" pos="162:8:10" line-data="        importlib.metadata.version(&quot;scikit-learn&quot;)">`scikit-learn`</SwmToken> library is installed in the environment.

```python
def is_sklearn_available():
    return _sklearn_available
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/import_utils.py" line="953">

---

The function <SwmToken path="src/transformers/utils/import_utils.py" pos="953:2:2" line-data="def is_levenshtein_available():">`is_levenshtein_available`</SwmToken> checks if the <SwmToken path="src/transformers/utils/import_utils.py" pos="1334:8:10" line-data="{0} requires the python-Levenshtein library but it was not found in your environment. You can install it with pip: `pip">`python-Levenshtein`</SwmToken> library is installed in the environment.

```python
def is_levenshtein_available():
    return _levenshtein_available
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/import_utils.py" line="77">

---

The constant <SwmToken path="src/transformers/utils/import_utils.py" pos="77:0:0" line-data="USE_TF = os.environ.get(&quot;USE_TF&quot;, &quot;AUTO&quot;).upper()">`USE_TF`</SwmToken> is used to determine if <SwmToken path="src/transformers/utils/import_utils.py" pos="197:9:9" line-data="        # Note: _is_package_available(&quot;tensorflow&quot;) fails for tensorflow-cpu. Please test any changes to the line below">`tensorflow`</SwmToken> should be used.

```python
USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/import_utils.py" line="1165">

---

The error message <SwmToken path="src/transformers/utils/import_utils.py" pos="1168:0:0" line-data="AV_IMPORT_ERROR = &quot;&quot;&quot;">`AV_IMPORT_ERROR`</SwmToken> provides guidance on how to install the <SwmToken path="src/transformers/utils/import_utils.py" pos="1169:8:8" line-data="{0} requires the PyAv library but it was not found in your environment. You can install it with:">`PyAv`</SwmToken> library if it is missing.

````python


# docstyle-ignore
AV_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install av
````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/import_utils.py" line="1174">

---

The error message <SwmToken path="src/transformers/utils/import_utils.py" pos="1178:0:0" line-data="CV2_IMPORT_ERROR = &quot;&quot;&quot;">`CV2_IMPORT_ERROR`</SwmToken> provides guidance on how to install the <SwmToken path="src/transformers/utils/import_utils.py" pos="1179:8:8" line-data="{0} requires the OpenCV library but it was not found in your environment. You can install it with:">`OpenCV`</SwmToken> library if it is missing.

````python
"""


# docstyle-ignore
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:
```
````

---

</SwmSnippet>

# Utils Endpoints

Utils Endpoints

<SwmSnippet path="/src/transformers/utils/peft_utils.py" line="29">

---

## <SwmToken path="src/transformers/utils/peft_utils.py" pos="29:2:2" line-data="def find_adapter_config_file(">`find_adapter_config_file`</SwmToken>

The <SwmToken path="src/transformers/utils/peft_utils.py" pos="29:2:2" line-data="def find_adapter_config_file(">`find_adapter_config_file`</SwmToken> function checks if a model stored on the Hub or locally is an adapter model. It returns the path of the adapter config file if it is, or None otherwise. This function takes several parameters such as <SwmToken path="src/transformers/utils/peft_utils.py" pos="30:1:1" line-data="    model_id: str,">`model_id`</SwmToken>, <SwmToken path="src/transformers/utils/peft_utils.py" pos="31:1:1" line-data="    cache_dir: Optional[Union[str, os.PathLike]] = None,">`cache_dir`</SwmToken>, <SwmToken path="src/transformers/utils/peft_utils.py" pos="32:1:1" line-data="    force_download: bool = False,">`force_download`</SwmToken>, and others to locate and verify the adapter configuration file.

```python
def find_adapter_config_file(
    model_id: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    _commit_hash: Optional[str] = None,
) -> Optional[str]:
    r"""
    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path of the adapter
    config file if it is, None otherwise.

    Args:
        model_id (`str`):
            The identifier of the model to look for, can be either a local path or an id to the repository on the Hub.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/peft_utils.py" line="108">

---

## <SwmToken path="src/transformers/utils/peft_utils.py" pos="108:2:2" line-data="def check_peft_version(min_version: str) -&gt; None:">`check_peft_version`</SwmToken>

The <SwmToken path="src/transformers/utils/peft_utils.py" pos="108:2:2" line-data="def check_peft_version(min_version: str) -&gt; None:">`check_peft_version`</SwmToken> function checks if the version of PEFT (Parameter-Efficient Fine-Tuning) is compatible with the specified minimum version. If PEFT is not installed or the version is not compatible, it raises a <SwmToken path="src/transformers/utils/peft_utils.py" pos="117:3:3" line-data="        raise ValueError(&quot;PEFT is not installed. Please install it with `pip install peft`&quot;)">`ValueError`</SwmToken>. This function ensures that the correct version of PEFT is used for model fine-tuning.

```python
def check_peft_version(min_version: str) -> None:
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version)

    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
