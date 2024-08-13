---
title: What is Pipeline Base in Pipelines
---
Pipeline Base refers to the foundational class and components that all pipeline implementations inherit from. It provides the core structure and methods necessary for creating and managing pipelines.

The Pipeline Base class defines the workflow of a pipeline, which includes a sequence of operations such as input handling, tokenization, model inference, <SwmToken path="src/transformers/pipelines/base.py" pos="596:15:17" line-data="        Input -&gt; Tokenization -&gt; Model Inference -&gt; Post-Processing (task dependent) -&gt; Output">`Post-Processing`</SwmToken>, and output generation.

It supports running on both CPU and GPU, allowing for flexible deployment of models depending on the available hardware.

The Pipeline Base class also includes utilities for saving and loading models, handling device placement, and ensuring that tensors are on the correct device.

Additionally, it provides mechanisms for handling different data formats through the <SwmToken path="src/transformers/pipelines/base.py" pos="292:2:2" line-data="class PipelineDataFormat:">`PipelineDataFormat`</SwmToken> class and its subclasses, which support JSON, CSV, and piped input/output formats.

<SwmSnippet path="/src/transformers/pipelines/base.py" line="588">

---

# Pipeline Base Class

The Pipeline Base class is the class from which all pipelines inherit. It defines the workflow of a pipeline, including input handling, tokenization, model inference, <SwmToken path="src/transformers/pipelines/base.py" pos="596:15:17" line-data="        Input -&gt; Tokenization -&gt; Model Inference -&gt; Post-Processing (task dependent) -&gt; Output">`Post-Processing`</SwmToken>, and output generation.

```python
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="598">

---

# Device Support

Pipeline supports running on CPU or GPU through the device argument, allowing for flexible deployment of models depending on the available hardware.

```python
    Pipeline supports running on CPU or GPU through the device argument (see below).

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="292">

---

# Data Format Handling

The <SwmToken path="src/transformers/pipelines/base.py" pos="292:2:2" line-data="class PipelineDataFormat:">`PipelineDataFormat`</SwmToken> class and its subclasses support different data formats such as JSON, CSV, and piped input/output formats.

```python
class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    :obj:`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets
    columns to pipelines keyword arguments through the :obj:`dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/pipelines/base.py" pos="53:2:2" line-data="def infer_framework_load_model(">`infer_framework_load_model`</SwmToken>, <SwmToken path="src/transformers/pipelines/base.py" pos="149:2:2" line-data="def infer_framework_from_model(">`infer_framework_from_model`</SwmToken>, and <SwmToken path="src/transformers/pipelines/base.py" pos="223:2:2" line-data="def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -&gt; str:">`get_default_model`</SwmToken>. We will dive a little into these functions.

<SwmSnippet path="/src/transformers/pipelines/base.py" line="53">

---

## <SwmToken path="src/transformers/pipelines/base.py" pos="53:2:2" line-data="def infer_framework_load_model(">`infer_framework_load_model`</SwmToken>

The <SwmToken path="src/transformers/pipelines/base.py" pos="53:2:2" line-data="def infer_framework_load_model(">`infer_framework_load_model`</SwmToken> function selects the framework <SwmToken path="src/transformers/pipelines/base.py" pos="62:5:6" line-data="    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).">`(TensorFlow`</SwmToken> or <SwmToken path="src/transformers/pipelines/base.py" pos="62:10:10" line-data="    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).">`PyTorch`</SwmToken>) to use from the model passed. It returns a tuple (framework, model). If the model is instantiated, it infers the framework from the model class. Otherwise, it tries to instantiate the model using <SwmToken path="src/transformers/pipelines/base.py" pos="56:1:1" line-data="    model_classes: Optional[Dict[str, Tuple[type]]] = None,">`model_classes`</SwmToken>. If both frameworks are available, <SwmToken path="src/transformers/pipelines/base.py" pos="62:10:10" line-data="    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).">`PyTorch`</SwmToken> is selected by default.

```python
def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs
):
    """
    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).

    If :obj:`model` is instantiated, this function will just infer the framework from the model class. Otherwise
    :obj:`model` is actually a checkpoint name and this method will try to instantiate it using :obj:`model_classes`.
    Since we don't want to instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for :obj:`model`, PyTorch is selected.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            The model to infer the framework from. If :obj:`str`, a checkpoint name. The model to infer the framewrok
            from.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="149">

---

## <SwmToken path="src/transformers/pipelines/base.py" pos="149:2:2" line-data="def infer_framework_from_model(">`infer_framework_from_model`</SwmToken>

The <SwmToken path="src/transformers/pipelines/base.py" pos="149:2:2" line-data="def infer_framework_from_model(">`infer_framework_from_model`</SwmToken> function selects the framework <SwmToken path="src/transformers/pipelines/base.py" pos="157:5:6" line-data="    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).">`(TensorFlow`</SwmToken> or <SwmToken path="src/transformers/pipelines/base.py" pos="157:10:10" line-data="    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).">`PyTorch`</SwmToken>) to use from the model passed. It returns a tuple (framework, model). If the model is instantiated, it infers the framework from the model class. Otherwise, it tries to instantiate the model using <SwmToken path="src/transformers/pipelines/base.py" pos="151:1:1" line-data="    model_classes: Optional[Dict[str, Tuple[type]]] = None,">`model_classes`</SwmToken>. This function calls <SwmToken path="src/transformers/pipelines/base.py" pos="53:2:2" line-data="def infer_framework_load_model(">`infer_framework_load_model`</SwmToken> to perform the actual framework inference.

```python
def infer_framework_from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs
):
    """
    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).

    If :obj:`model` is instantiated, this function will just infer the framework from the model class. Otherwise
    :obj:`model` is actually a checkpoint name and this method will try to instantiate it using :obj:`model_classes`.
    Since we don't want to instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for :obj:`model`, PyTorch is selected.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            The model to infer the framework from. If :obj:`str`, a checkpoint name. The model to infer the framewrok
            from.
        model_classes (dictionary :obj:`str` to :obj:`type`, `optional`):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="223">

---

## <SwmToken path="src/transformers/pipelines/base.py" pos="223:2:2" line-data="def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -&gt; str:">`get_default_model`</SwmToken>

The <SwmToken path="src/transformers/pipelines/base.py" pos="223:2:2" line-data="def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -&gt; str:">`get_default_model`</SwmToken> function selects a default model to use for a given task. It defaults to <SwmToken path="src/transformers/pipelines/base.py" pos="225:26:26" line-data="    Select a default model to use for a given task. Defaults to pytorch if ambiguous.">`pytorch`</SwmToken> if the framework is ambiguous. This function checks the available frameworks and selects the appropriate model based on the task and framework.

```python
def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -> str:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (:obj:`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (:obj:`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (:obj:`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        :obj:`str` The model string representing the default model for this pipeline
    """
    if is_torch_available() and not is_tf_available():
        framework = "pt"
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
