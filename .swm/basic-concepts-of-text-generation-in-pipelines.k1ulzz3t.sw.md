---
title: Basic concepts of Text Generation in Pipelines
---
Text Generation refers to the process of generating text based on a given input prompt. This is achieved using models that have been trained with an autoregressive language modeling objective.

The <SwmToken path="src/transformers/pipelines/text_generation.py" pos="38:2:2" line-data="class TextGenerationPipeline(Pipeline):">`TextGenerationPipeline`</SwmToken> class is responsible for handling text generation tasks. It predicts the words that will follow a specified text prompt.

When the underlying model is a conversational model, the pipeline can also accept one or more chats and operate in chat mode, continuing the conversation by adding its responses.

The pipeline can be loaded using the task identifier <SwmToken path="src/transformers/pipelines/text_generation.py" pos="73:3:5" line-data="    `&quot;text-generation&quot;`.">`text-generation`</SwmToken> and supports various models trained for text completion and conversational tasks.

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="191:9:9" line-data="        output_ids = self.model.generate(**model_inputs, **generate_kwargs)">`generate`</SwmToken> method for text generation is implemented in the <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="54:24:24" line-data="    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)">`GenerationMixin`</SwmToken> class for each framework: <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="233:9:9" line-data="    # use bart in pytorch">`pytorch`</SwmToken>, <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="10:3:3" line-data="    import tensorflow as tf">`tensorflow`</SwmToken>, and Flax/JAX.

Text Generation Each framework has a generate method for text generation implemented in their respective <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="54:24:24" line-data="    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)">`GenerationMixin`</SwmToken> class: <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="233:9:9" line-data="    # use bart in pytorch">`pytorch`</SwmToken> `~generation.GenerationMixin.generate` is implemented in `~generation.GenerationMixin`, <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="10:3:3" line-data="    import tensorflow as tf">`tensorflow`</SwmToken> `~generation.TFGenerationMixin.generate` is implemented in `~generation.TFGenerationMixin`, and Flax/JAX `~generation.FlaxGenerationMixin.generate` is implemented in `~generation.FlaxGenerationMixin`.

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="26">

---

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="26:2:2" line-data="class Text2TextGenerationPipeline(Pipeline):">`Text2TextGenerationPipeline`</SwmToken> class is an example of a pipeline for text-to-text generation using <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="28:15:15" line-data="    Pipeline for text to text generation using seq2seq models.">`seq2seq`</SwmToken> models. It demonstrates how to use the pipeline for generating text based on a given input.

````python
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
    >>> generator(
    ...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
    ... )
    [{'generated_text': 'question: Who created the RuPERTa-base?'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/text_generation.py" line="38">

---

The <SwmToken path="src/transformers/pipelines/text_generation.py" pos="38:2:2" line-data="class TextGenerationPipeline(Pipeline):">`TextGenerationPipeline`</SwmToken> class is responsible for handling text generation tasks. It predicts the words that will follow a specified text prompt and can operate in chat mode for conversational models.

````python
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Examples:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="openai-community/gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```

    ```python
````

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are preprocess, \_forward, and postprocess. We will dive a little into preprocess and \_forward.

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="176">

---

## preprocess

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="176:3:3" line-data="    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):">`preprocess`</SwmToken> function is responsible for preparing the input data for the model. It tokenizes the input text and applies any necessary transformations before passing it to the model.

```python
    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        return inputs
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="180">

---

## \_forward

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="180:3:3" line-data="    def _forward(self, model_inputs, **generate_kwargs):">`_forward`</SwmToken> function handles the forward pass of the model. It generates the output sequences based on the input data and the specified generation parameters.

```python
    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            in_b, input_length = tf.shape(model_inputs["input_ids"]).numpy()

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        out_b = output_ids.shape[0]
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            output_ids = tf.reshape(output_ids, (in_b, out_b // in_b, *output_ids.shape[1:]))
        return {"output_ids": output_ids}
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/text2text_generation.py" line="199">

---

## postprocess

The <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="199:3:3" line-data="    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):">`postprocess`</SwmToken> function processes the model's output to produce the final generated text. It decodes the token <SwmToken path="src/transformers/pipelines/text2text_generation.py" pos="164:1:1" line-data="              ids of the generated text.">`ids`</SwmToken> into human-readable text and applies any necessary formatting.

```python
    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        return records
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
