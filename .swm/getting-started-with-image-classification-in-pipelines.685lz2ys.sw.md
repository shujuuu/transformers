---
title: Getting started with Image Classification in Pipelines
---
Image classification assigns a label or class to an image. Unlike text or audio classification, the inputs are the pixel values that comprise an image.

There are many applications for image classification, such as detecting damage after a natural disaster, monitoring crop health, or helping screen medical images for signs of disease.

The <SwmToken path="src/transformers/pipelines/image_classification.py" pos="64:2:2" line-data="class ImageClassificationPipeline(Pipeline):">`ImageClassificationPipeline`</SwmToken> class is used to predict the class of an image using any <SwmToken path="src/transformers/pipelines/image_classification.py" pos="66:12:12" line-data="    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an">`AutoModelForImageClassification`</SwmToken>.

The pipeline handles three types of images: a string containing an HTTP link pointing to an image, a string containing a local path to an image, or an image loaded in PIL directly.

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format.

The <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="33:2:2" line-data="class ZeroShotImageClassificationPipeline(Pipeline):">`ZeroShotImageClassificationPipeline`</SwmToken> class predicts the class of an image when you provide an image and a set of <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="36:16:16" line-data="    provide an image and a set of `candidate_labels`.">`candidate_labels`</SwmToken>.

The pipeline uses the <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="35:14:14" line-data="    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you">`CLIPModel`</SwmToken> to estimate the likelihood of each candidate label by using <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="94:1:1" line-data="                logits_per_image">`logits_per_image`</SwmToken>.

## Image Classification Overview

Image classification assigns a label or class to an image. Unlike text or audio classification, the inputs are the pixel values that comprise an image. There are many applications for image classification, such as detecting damage after a natural disaster, monitoring crop health, or helping screen medical images for signs of disease.

<SwmSnippet path="/src/transformers/pipelines/image_classification.py" line="64">

---

## How to Use Image Classification

The <SwmToken path="src/transformers/pipelines/image_classification.py" pos="64:2:2" line-data="class ImageClassificationPipeline(Pipeline):">`ImageClassificationPipeline`</SwmToken> class is used to predict the class of an image using any <SwmToken path="src/transformers/pipelines/image_classification.py" pos="66:12:12" line-data="    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an">`AutoModelForImageClassification`</SwmToken>. This pipeline predicts the class of an image.

````python
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/image_classification.py" line="112">

---

The pipeline handles three types of images: a string containing an HTTP link pointing to an image, a string containing a local path to an image, or an image loaded in PIL directly. The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format.

```python
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/zero_shot_image_classification.py" line="33">

---

The <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="33:2:2" line-data="class ZeroShotImageClassificationPipeline(Pipeline):">`ZeroShotImageClassificationPipeline`</SwmToken> class predicts the class of an image when you provide an image and a set of <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="36:16:16" line-data="    provide an image and a set of `candidate_labels`.">`candidate_labels`</SwmToken>. This pipeline uses the <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="35:14:14" line-data="    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you">`CLIPModel`</SwmToken> to estimate the likelihood of each candidate label by using <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="94:1:1" line-data="                logits_per_image">`logits_per_image`</SwmToken>.

````python
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="google/siglip-so400m-patch14-384")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
````

---

</SwmSnippet>

# Main functions

Main functions

<SwmSnippet path="/src/transformers/pipelines/image_classification.py" line="64">

---

## <SwmToken path="src/transformers/pipelines/image_classification.py" pos="64:2:2" line-data="class ImageClassificationPipeline(Pipeline):">`ImageClassificationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/image_classification.py" pos="64:2:2" line-data="class ImageClassificationPipeline(Pipeline):">`ImageClassificationPipeline`</SwmToken> class is used to predict the class of an image using any <SwmToken path="src/transformers/pipelines/image_classification.py" pos="66:12:12" line-data="    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an">`AutoModelForImageClassification`</SwmToken>. This pipeline handles three types of images: a string containing an HTTP link pointing to an image, a string containing a local path to an image, or an image loaded in PIL directly. It accepts either a single image or a batch of images, which must all be in the same format.

````python
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/zero_shot_image_classification.py" line="33">

---

## <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="33:2:2" line-data="class ZeroShotImageClassificationPipeline(Pipeline):">`ZeroShotImageClassificationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="33:2:2" line-data="class ZeroShotImageClassificationPipeline(Pipeline):">`ZeroShotImageClassificationPipeline`</SwmToken> class predicts the class of an image when you provide an image and a set of <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="36:16:16" line-data="    provide an image and a set of `candidate_labels`.">`candidate_labels`</SwmToken>. This pipeline uses the <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="35:14:14" line-data="    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you">`CLIPModel`</SwmToken> to estimate the likelihood of each candidate label by using <SwmToken path="src/transformers/pipelines/zero_shot_image_classification.py" pos="94:1:1" line-data="                logits_per_image">`logits_per_image`</SwmToken>.

````python
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="google/siglip-so400m-patch14-384")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
````

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
