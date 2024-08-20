---
title: Exploring Configuration in Utils
---
Configuration refers to the setup and parameters used to define the behavior and attributes of machine learning models.

The base class `PretrainedConfig` implements common methods for loading and saving configurations from local files or directories, or from pretrained model configurations provided by the library.

Each derived configuration class implements model-specific attributes, such as `hidden_size`, `num_attention_heads`, and `num_hidden_layers`.

Text models further implement attributes like `vocab_size`.

## PretrainedConfig

The base class `PretrainedConfig` implements common methods for loading and saving configurations from local files or directories, or from pretrained model configurations provided by the library.

## Model-Specific Attributes

Each derived configuration class implements model-specific attributes, such as `hidden_size`, `num_attention_heads`, and `num_hidden_layers`. Text models further implement attributes like `vocab_size`.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
