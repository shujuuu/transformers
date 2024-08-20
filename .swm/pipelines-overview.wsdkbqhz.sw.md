---
title: Pipelines Overview
---
Pipelines are a great and easy way to use models for inference.

These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks.

Tasks supported by pipelines include Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction, and Question Answering.

## Pipeline usage

While each task has an associated <SwmToken path="src/transformers/pipelines/audio_classification.py" pos="67:4:4" line-data="class AudioClassificationPipeline(Pipeline):">`Pipeline`</SwmToken>, it is simpler to use the general <SwmToken path="src/transformers/pipelines/audio_classification.py" pos="67:4:4" line-data="class AudioClassificationPipeline(Pipeline):">`Pipeline`</SwmToken> abstraction which contains all the task-specific pipelines. The <SwmToken path="src/transformers/pipelines/audio_classification.py" pos="67:4:4" line-data="class AudioClassificationPipeline(Pipeline):">`Pipeline`</SwmToken> automatically loads a default model and a preprocessing class capable of inference for your task.

<SwmSnippet path="/src/transformers/pipelines/audio_classification.py" line="67">

---

The <SwmToken path="src/transformers/pipelines/audio_classification.py" pos="67:2:2" line-data="class AudioClassificationPipeline(Pipeline):">`AudioClassificationPipeline`</SwmToken> class is an example of a specific pipeline that uses a model for audio classification tasks. It demonstrates how to create a pipeline for a specific task and use it to classify audio inputs.

````python
class AudioClassificationPipeline(Pipeline):
    """
    Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
    raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
    formats.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
    >>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    [{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-classification"`.
````

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/automatic_speech_recognition.py" line="126">

---

The <SwmToken path="src/transformers/pipelines/automatic_speech_recognition.py" pos="126:2:2" line-data="class AutomaticSpeechRecognitionPipeline(ChunkPipeline):">`AutomaticSpeechRecognitionPipeline`</SwmToken> class is another example of a specific pipeline that transcribes spoken text from audio inputs. It shows how to set up and use a pipeline for automatic speech recognition tasks.

````python
class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Example:

    ```python
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
````

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
