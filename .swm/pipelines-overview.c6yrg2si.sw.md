---
title: Pipelines Overview
---
Pipelines are a high-level abstraction that allows users to perform various Natural Language Processing (NLP) tasks with minimal code.

They encapsulate the entire process of tokenizing input data, passing it through a model, and <SwmToken path="src/transformers/pipelines/base.py" pos="596:15:17" line-data="        Input -&gt; Tokenization -&gt; Model Inference -&gt; Post-Processing (task dependent) -&gt; Output">`Post-Processing`</SwmToken> the model's output.

Pipelines support a wide range of tasks such as text classification, token classification, question answering, text generation, and more.

Each pipeline is associated with a specific task and can be instantiated using the <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="21:4:4" line-data="class Text2TextGenerationPipeline(Pipeline):">`Pipeline`</SwmToken> function, which automatically loads the appropriate model, tokenizer, and configuration.

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="21:4:4" line-data="class Text2TextGenerationPipeline(Pipeline):">`Pipeline`</SwmToken> function simplifies the process of using pretrained models by handling the complexities of model loading and configuration.

Pipelines can be customized by specifying different models, tokenizers, and configurations, allowing for flexibility in their usage.

They are designed to be easy to use, making advanced NLP accessible to a broader audience, including those who may not have deep expertise in machine learning.

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="21">

---

## <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="21:2:2" line-data="class Text2TextGenerationPipeline(Pipeline):">`Text2TextGenerationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="21:2:2" line-data="class Text2TextGenerationPipeline(Pipeline):">`Text2TextGenerationPipeline`</SwmToken> class demonstrates how a pipeline can be used for text-to-text generation tasks. It can be instantiated using the <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="21:4:4" line-data="class Text2TextGenerationPipeline(Pipeline):">`Pipeline`</SwmToken> function with the task identifier <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="26:12:14" line-data="    following task identifier: :obj:`&quot;text2text-generation&quot;`.">`text2text-generation`</SwmToken>.

```python
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    This Text2TextGenerationPipeline pipeline can currently be loaded from :func:`~transformers.pipeline` using the
    following task identifier: :obj:`"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models <https://huggingface.co/models?filter=seq2seq>`__.

    Usage::

        text2text_generator = pipeline("text2text-generation")
        text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="153">

---

## <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="153:2:2" line-data="class SummarizationPipeline(Text2TextGenerationPipeline):">`SummarizationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="153:2:2" line-data="class SummarizationPipeline(Text2TextGenerationPipeline):">`SummarizationPipeline`</SwmToken> class is an example of a pipeline used for summarizing text. It can be instantiated using the <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="157:5:5" line-data="    This summarizing pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task">`pipeline`</SwmToken> function with the task identifier <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="158:8:8" line-data="    identifier: :obj:`&quot;summarization&quot;`.">`summarization`</SwmToken>.

```python
class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'. See the up-to-date
    list of available models on `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__.

    Usage::

        # use bart in pytorch
        summarizer = pipeline("summarization")
        summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

        # use t5 in tf
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="223">

---

## <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="223:2:2" line-data="class TranslationPipeline(Text2TextGenerationPipeline):">`TranslationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="223:2:2" line-data="class TranslationPipeline(Text2TextGenerationPipeline):">`TranslationPipeline`</SwmToken> class shows how a pipeline can be used for translation tasks. It can be instantiated using the <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="227:5:5" line-data="    This translation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task">`pipeline`</SwmToken> function with the task identifier <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="228:8:8" line-data="    identifier: :obj:`&quot;translation_xx_to_yy&quot;`.">`translation_xx_to_yy`</SwmToken>.

```python
class TranslationPipeline(Text2TextGenerationPipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=translation>`__.

    Usage::
        en_fr_translator = pipeline("translation_en_to_fr")
        en_fr_translator("How old are you?")
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="557">

---

## Pipeline Initialization

The <SwmToken path="src/transformers/pipelines/base.py" pos="557:0:0" line-data="PIPELINE_INIT_ARGS = r&quot;&quot;&quot;">`PIPELINE_INIT_ARGS`</SwmToken> constant provides the initialization arguments for pipelines, including the model, tokenizer, framework, and task identifier.

```python
PIPELINE_INIT_ARGS = r"""
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/base.py" line="588">

---

## Pipeline Class

The <SwmToken path="src/transformers/pipelines/base.py" pos="588:2:2" line-data="class Pipeline(_ScikitCompat):">`Pipeline`</SwmToken> class is the base class for all pipelines. It defines the workflow for tokenization, model inference, and <SwmToken path="src/transformers/pipelines/base.py" pos="596:15:17" line-data="        Input -&gt; Tokenization -&gt; Model Inference -&gt; Post-Processing (task dependent) -&gt; Output">`Post-Processing`</SwmToken>.

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

    default_input_names = None

    def __init__(
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
