---
title: Overview of Model Mappings
---
Modeling involves defining mappings between model configurations and their corresponding implementations. These mappings are organized in an <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:4:4" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`OrderedDict`</SwmToken> to ensure the order is maintained.

Each mapping associates a configuration class, such as <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="9:3:3" line-data="        (&quot;RoFormerConfig&quot;, &quot;RoFormerForQuestionAnswering&quot;),">`RoFormerConfig`</SwmToken>, with a specific model implementation, like <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="9:8:8" line-data="        (&quot;RoFormerConfig&quot;, &quot;RoFormerForQuestionAnswering&quot;),">`RoFormerForQuestionAnswering`</SwmToken>. This allows for dynamic model instantiation based on configuration.

Different tasks have their own specific mappings. For example, <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken> maps configurations to models designed for question answering tasks, while <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken> maps configurations to models for causal language modeling.

These mappings enable the framework to support a wide variety of models and tasks, making it flexible and extensible for different NLP applications.

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="7">

---

# Question Answering Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken> constant maps various configurations to their respective question-answering model implementations.

```python
MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForQuestionAnswering"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForQuestionAnswering"),
        ("BigBirdConfig", "BigBirdForQuestionAnswering"),
        ("ConvBertConfig", "ConvBertForQuestionAnswering"),
        ("LEDConfig", "LEDForQuestionAnswering"),
        ("DistilBertConfig", "DistilBertForQuestionAnswering"),
        ("AlbertConfig", "AlbertForQuestionAnswering"),
        ("CamembertConfig", "CamembertForQuestionAnswering"),
        ("BartConfig", "BartForQuestionAnswering"),
        ("MBartConfig", "MBartForQuestionAnswering"),
        ("LongformerConfig", "LongformerForQuestionAnswering"),
        ("XLMRobertaConfig", "XLMRobertaForQuestionAnswering"),
        ("RobertaConfig", "RobertaForQuestionAnswering"),
        ("SqueezeBertConfig", "SqueezeBertForQuestionAnswering"),
        ("BertConfig", "BertForQuestionAnswering"),
        ("XLNetConfig", "XLNetForQuestionAnsweringSimple"),
        ("FlaubertConfig", "FlaubertForQuestionAnsweringSimple"),
        ("MegatronBertConfig", "MegatronBertForQuestionAnswering"),
        ("MobileBertConfig", "MobileBertForQuestionAnswering"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="41">

---

# Causal Language Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken> constant maps configurations to models designed for causal language modeling tasks.

```python
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForCausalLM"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForCausalLM"),
        ("GPTNeoConfig", "GPTNeoForCausalLM"),
        ("BigBirdConfig", "BigBirdForCausalLM"),
        ("CamembertConfig", "CamembertForCausalLM"),
        ("XLMRobertaConfig", "XLMRobertaForCausalLM"),
        ("RobertaConfig", "RobertaForCausalLM"),
        ("BertConfig", "BertLMHeadModel"),
        ("OpenAIGPTConfig", "OpenAIGPTLMHeadModel"),
        ("GPT2Config", "GPT2LMHeadModel"),
        ("TransfoXLConfig", "TransfoXLLMHeadModel"),
        ("XLNetConfig", "XLNetLMHeadModel"),
        ("XLMConfig", "XLMWithLMHeadModel"),
        ("CTRLConfig", "CTRLLMHeadModel"),
        ("ReformerConfig", "ReformerModelWithLMHead"),
        ("BertGenerationConfig", "BertGenerationDecoder"),
        ("XLMProphetNetConfig", "XLMProphetNetForCausalLM"),
        ("ProphetNetConfig", "ProphetNetForCausalLM"),
        ("BartConfig", "BartForCausalLM"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="72">

---

# Image Classification Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="72:0:0" line-data="MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for image classification.

```python
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("ViTConfig", "ViTForImageClassification"),
        ("DeiTConfig", "('DeiTForImageClassification', 'DeiTForImageClassificationWithTeacher')"),
    ]
)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="176">

---

# Sequence Classification Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="176:0:0" line-data="MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for sequence classification tasks.

```python
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForSequenceClassification"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForSequenceClassification"),
        ("BigBirdConfig", "BigBirdForSequenceClassification"),
        ("ConvBertConfig", "ConvBertForSequenceClassification"),
        ("LEDConfig", "LEDForSequenceClassification"),
        ("DistilBertConfig", "DistilBertForSequenceClassification"),
        ("AlbertConfig", "AlbertForSequenceClassification"),
        ("CamembertConfig", "CamembertForSequenceClassification"),
        ("XLMRobertaConfig", "XLMRobertaForSequenceClassification"),
        ("MBartConfig", "MBartForSequenceClassification"),
        ("BartConfig", "BartForSequenceClassification"),
        ("LongformerConfig", "LongformerForSequenceClassification"),
        ("RobertaConfig", "RobertaForSequenceClassification"),
        ("SqueezeBertConfig", "SqueezeBertForSequenceClassification"),
        ("LayoutLMConfig", "LayoutLMForSequenceClassification"),
        ("BertConfig", "BertForSequenceClassification"),
        ("XLNetConfig", "XLNetForSequenceClassification"),
        ("MegatronBertConfig", "MegatronBertForSequenceClassification"),
        ("MobileBertConfig", "MobileBertForSequenceClassification"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="139">

---

# Next Sentence Prediction Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="139:0:0" line-data="MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for next sentence prediction.

```python
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("BertConfig", "BertForNextSentencePrediction"),
        ("MegatronBertConfig", "MegatronBertForNextSentencePrediction"),
        ("MobileBertConfig", "MobileBertForNextSentencePrediction"),
    ]
)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="315">

---

# Models with LM Head

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="315:0:0" line-data="MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(">`MODEL_WITH_LM_HEAD_MAPPING_NAMES`</SwmToken> constant maps configurations to models that include a language modeling head.

```python
MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForMaskedLM"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForConditionalGeneration"),
        ("GPTNeoConfig", "GPTNeoForCausalLM"),
        ("BigBirdConfig", "BigBirdForMaskedLM"),
        ("Speech2TextConfig", "Speech2TextForConditionalGeneration"),
        ("Wav2Vec2Config", "Wav2Vec2ForMaskedLM"),
        ("M2M100Config", "M2M100ForConditionalGeneration"),
        ("ConvBertConfig", "ConvBertForMaskedLM"),
        ("LEDConfig", "LEDForConditionalGeneration"),
        ("BlenderbotSmallConfig", "BlenderbotSmallForConditionalGeneration"),
        ("LayoutLMConfig", "LayoutLMForMaskedLM"),
        ("T5Config", "T5ForConditionalGeneration"),
        ("DistilBertConfig", "DistilBertForMaskedLM"),
        ("AlbertConfig", "AlbertForMaskedLM"),
        ("CamembertConfig", "CamembertForMaskedLM"),
        ("XLMRobertaConfig", "XLMRobertaForMaskedLM"),
        ("MarianConfig", "MarianMTModel"),
        ("FSMTConfig", "FSMTForConditionalGeneration"),
        ("BartConfig", "BartForConditionalGeneration"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="148">

---

# Object Detection Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="148:0:0" line-data="MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for object detection tasks.

```python
MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        ("DetrConfig", "DetrForObjectDetection"),
    ]
)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="155">

---

# Seq-to-Seq Causal LM Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="155:0:0" line-data="MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for sequence-to-sequence causal language modeling.

```python
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("BigBirdPegasusConfig", "BigBirdPegasusForConditionalGeneration"),
        ("M2M100Config", "M2M100ForConditionalGeneration"),
        ("LEDConfig", "LEDForConditionalGeneration"),
        ("BlenderbotSmallConfig", "BlenderbotSmallForConditionalGeneration"),
        ("MT5Config", "MT5ForConditionalGeneration"),
        ("T5Config", "T5ForConditionalGeneration"),
        ("PegasusConfig", "PegasusForConditionalGeneration"),
        ("MarianConfig", "MarianMTModel"),
        ("MBartConfig", "MBartForConditionalGeneration"),
        ("BlenderbotConfig", "BlenderbotForConditionalGeneration"),
        ("BartConfig", "BartForConditionalGeneration"),
        ("FSMTConfig", "FSMTForConditionalGeneration"),
        ("EncoderDecoderConfig", "EncoderDecoderModel"),
        ("XLMProphetNetConfig", "XLMProphetNetForConditionalGeneration"),
        ("ProphetNetConfig", "ProphetNetForConditionalGeneration"),
    ]
)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="216">

---

# Table Question Answering Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="216:0:0" line-data="MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for table question answering tasks.

```python
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("TapasConfig", "TapasForQuestionAnswering"),
    ]
)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="223">

---

# Token Classification Models

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="223:0:0" line-data="MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES`</SwmToken> constant maps configurations to models used for token classification tasks.

```python
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForTokenClassification"),
        ("BigBirdConfig", "BigBirdForTokenClassification"),
        ("ConvBertConfig", "ConvBertForTokenClassification"),
        ("LayoutLMConfig", "LayoutLMForTokenClassification"),
        ("DistilBertConfig", "DistilBertForTokenClassification"),
        ("CamembertConfig", "CamembertForTokenClassification"),
        ("FlaubertConfig", "FlaubertForTokenClassification"),
        ("XLMConfig", "XLMForTokenClassification"),
        ("XLMRobertaConfig", "XLMRobertaForTokenClassification"),
        ("LongformerConfig", "LongformerForTokenClassification"),
        ("RobertaConfig", "RobertaForTokenClassification"),
        ("SqueezeBertConfig", "SqueezeBertForTokenClassification"),
        ("BertConfig", "BertForTokenClassification"),
        ("MegatronBertConfig", "MegatronBertForTokenClassification"),
        ("MobileBertConfig", "MobileBertForTokenClassification"),
        ("XLNetConfig", "XLNetForTokenClassification"),
        ("AlbertConfig", "AlbertForTokenClassification"),
        ("ElectraConfig", "ElectraForTokenClassification"),
        ("FunnelConfig", "FunnelForTokenClassification"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="252">

---

# General Model Mappings

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="252:0:0" line-data="MODEL_MAPPING_NAMES = OrderedDict(">`MODEL_MAPPING_NAMES`</SwmToken> constant provides a general mapping of configurations to their respective model implementations.

```python
MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("VisualBertConfig", "VisualBertModel"),
        ("RoFormerConfig", "RoFormerModel"),
        ("CLIPConfig", "CLIPModel"),
        ("BigBirdPegasusConfig", "BigBirdPegasusModel"),
        ("DeiTConfig", "DeiTModel"),
        ("LukeConfig", "LukeModel"),
        ("DetrConfig", "DetrModel"),
        ("GPTNeoConfig", "GPTNeoModel"),
        ("BigBirdConfig", "BigBirdModel"),
        ("Speech2TextConfig", "Speech2TextModel"),
        ("ViTConfig", "ViTModel"),
        ("Wav2Vec2Config", "Wav2Vec2Model"),
        ("HubertConfig", "HubertModel"),
        ("M2M100Config", "M2M100Model"),
        ("ConvBertConfig", "ConvBertModel"),
        ("LEDConfig", "LEDModel"),
        ("BlenderbotSmallConfig", "BlenderbotSmallModel"),
        ("RetriBertConfig", "RetriBertModel"),
        ("MT5Config", "MT5Model"),
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken>, <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken>, <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="72:0:0" line-data="MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES`</SwmToken>, and <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="80:0:0" line-data="MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_MASKED_LM_MAPPING_NAMES`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken> and <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken>.

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="7">

---

## <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken>

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:0:0" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES`</SwmToken> constant is an <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="7:4:4" line-data="MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(">`OrderedDict`</SwmToken> that maps various model configurations to their respective implementations for question answering tasks. This allows for dynamic instantiation of models based on the configuration provided.

```python
MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForQuestionAnswering"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForQuestionAnswering"),
        ("BigBirdConfig", "BigBirdForQuestionAnswering"),
        ("ConvBertConfig", "ConvBertForQuestionAnswering"),
        ("LEDConfig", "LEDForQuestionAnswering"),
        ("DistilBertConfig", "DistilBertForQuestionAnswering"),
        ("AlbertConfig", "AlbertForQuestionAnswering"),
        ("CamembertConfig", "CamembertForQuestionAnswering"),
        ("BartConfig", "BartForQuestionAnswering"),
        ("MBartConfig", "MBartForQuestionAnswering"),
        ("LongformerConfig", "LongformerForQuestionAnswering"),
        ("XLMRobertaConfig", "XLMRobertaForQuestionAnswering"),
        ("RobertaConfig", "RobertaForQuestionAnswering"),
        ("SqueezeBertConfig", "SqueezeBertForQuestionAnswering"),
        ("BertConfig", "BertForQuestionAnswering"),
        ("XLNetConfig", "XLNetForQuestionAnsweringSimple"),
        ("FlaubertConfig", "FlaubertForQuestionAnsweringSimple"),
        ("MegatronBertConfig", "MegatronBertForQuestionAnswering"),
        ("MobileBertConfig", "MobileBertForQuestionAnswering"),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/modeling_auto_mapping.py" line="41">

---

## <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken>

The <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:0:0" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`</SwmToken> constant is an <SwmToken path="src/transformers/utils/modeling_auto_mapping.py" pos="41:4:4" line-data="MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(">`OrderedDict`</SwmToken> that maps various model configurations to their respective implementations for causal language modeling tasks. This enables the framework to dynamically instantiate the appropriate model based on the given configuration.

```python
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("RoFormerConfig", "RoFormerForCausalLM"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForCausalLM"),
        ("GPTNeoConfig", "GPTNeoForCausalLM"),
        ("BigBirdConfig", "BigBirdForCausalLM"),
        ("CamembertConfig", "CamembertForCausalLM"),
        ("XLMRobertaConfig", "XLMRobertaForCausalLM"),
        ("RobertaConfig", "RobertaForCausalLM"),
        ("BertConfig", "BertLMHeadModel"),
        ("OpenAIGPTConfig", "OpenAIGPTLMHeadModel"),
        ("GPT2Config", "GPT2LMHeadModel"),
        ("TransfoXLConfig", "TransfoXLLMHeadModel"),
        ("XLNetConfig", "XLNetLMHeadModel"),
        ("XLMConfig", "XLMWithLMHeadModel"),
        ("CTRLConfig", "CTRLLMHeadModel"),
        ("ReformerConfig", "ReformerModelWithLMHead"),
        ("BertGenerationConfig", "BertGenerationDecoder"),
        ("XLMProphetNetConfig", "XLMProphetNetForCausalLM"),
        ("ProphetNetConfig", "ProphetNetForCausalLM"),
        ("BartConfig", "BartForCausalLM"),
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
