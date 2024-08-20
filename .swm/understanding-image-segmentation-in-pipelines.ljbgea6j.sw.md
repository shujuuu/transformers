---
title: Understanding Image Segmentation in Pipelines
---
Image segmentation refers to the process of partitioning an image into multiple segments to simplify or change the representation of an image into something more meaningful and easier to analyze.

In pipelines, image segmentation models assign a label to each pixel in an image, effectively separating areas corresponding to different objects or regions of interest.

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="31:2:2" line-data="class ImageSegmentationPipeline(Pipeline):">`ImageSegmentationPipeline`</SwmToken> class is used to perform segmentation tasks using models like <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="33:12:12" line-data="    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and">`AutoModelForXXXSegmentation`</SwmToken>, predicting masks of objects and their classes.

The pipeline can handle various types of images, including <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="105:11:13" line-data="                - A string containing an HTTP(S) link pointing to an image">`HTTP(S`</SwmToken>) links, local paths, and PIL images, and can process both single images and batches of images.

The segmentation task can be specified as semantic, instance, or panoptic, depending on the model's capabilities.

The output of the pipeline includes dictionaries containing the mask, label, and optionally the score of each detected object.

### Image segmentation

[Mask2Former](model_doc/mask2former) is a universal architecture for solving all types of image segmentation tasks. Traditional segmentation models are typically tailored towards a particular subtask of image segmentation, like instance, semantic or panoptic segmentation. Mask2Former frames each of those tasks as a *mask classification* problem. Mask classification groups pixels into *N* segments, and predicts *N* masks and their corresponding class label for a given image. We'll explain how Mask2Former works in this section, and then you can try finetuning SegFormer at the end.

<SwmSnippet path="/src/transformers/pipelines/image_segmentation.py" line="31">

---

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="31:2:2" line-data="class ImageSegmentationPipeline(Pipeline):">`ImageSegmentationPipeline`</SwmToken> class is used to perform segmentation tasks using models like <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="33:12:12" line-data="    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and">`AutoModelForXXXSegmentation`</SwmToken>, predicting masks of objects and their classes.

````python
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/image_segmentation.py" line="97">

---

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="97:3:3" line-data="    def __call__(self, images, **kwargs) -&gt; Union[Predictions, List[Prediction]]:">`__call__`</SwmToken> method in the <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="31:2:2" line-data="class ImageSegmentationPipeline(Pipeline):">`ImageSegmentationPipeline`</SwmToken> class handles various types of images, including <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="105:11:13" line-data="                - A string containing an HTTP(S) link pointing to an image">`HTTP(S`</SwmToken>) links, local paths, and PIL images, and can process both single images and batches of images. The segmentation task can be specified as semantic, instance, or panoptic, depending on the model's capabilities.

```python
    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
```

---

</SwmSnippet>

# Main functions

Main functions

<SwmSnippet path="/src/transformers/pipelines/image_segmentation.py" line="31">

---

## <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="31:2:2" line-data="class ImageSegmentationPipeline(Pipeline):">`ImageSegmentationPipeline`</SwmToken>

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="31:2:2" line-data="class ImageSegmentationPipeline(Pipeline):">`ImageSegmentationPipeline`</SwmToken> class is used to perform segmentation tasks using models like <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="33:12:12" line-data="    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and">`AutoModelForXXXSegmentation`</SwmToken>, predicting masks of objects and their classes.

````python
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/image_segmentation.py" line="97">

---

## **call**

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="97:3:3" line-data="    def __call__(self, images, **kwargs) -&gt; Union[Predictions, List[Prediction]]:">`__call__`</SwmToken> function performs segmentation (detects masks & classes) in the <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="99:19:21" line-data="        Perform segmentation (detect masks &amp; classes) in the image(s) passed as inputs.">`image(s`</SwmToken>) passed as inputs. It handles various types of images and can process both single images and batches of images.

```python
    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/image_segmentation.py" line="171">

---

## postprocess

The <SwmToken path="src/transformers/pipelines/image_segmentation.py" pos="171:3:3" line-data="    def postprocess(">`postprocess`</SwmToken> function processes the model outputs to generate the final segmentation results, including masks, labels, and optionally scores for each detected object.

```python
    def postprocess(
        self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
    ):
        fn = None
        if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            fn = self.image_processor.post_process_panoptic_segmentation
        elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
            fn = self.image_processor.post_process_instance_segmentation

        if fn is not None:
            outputs = fn(
                model_outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_sizes=model_outputs["target_size"],
            )[0]

            annotation = []
            segmentation = outputs["segmentation"]

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
