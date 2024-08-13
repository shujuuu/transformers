---
title: Getting started with T5 Model Overview
---
The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model, or <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1068:32:36" line-data="    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer">`Text-to-Text`</SwmToken> Transfer Transformer, is an encoder-decoder transformer model designed for various NLP tasks. It was proposed in the paper 'Exploring the Limits of Transfer Learning with a Unified <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1068:32:36" line-data="    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer">`Text-to-Text`</SwmToken> Transformer'.

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model is <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1070:36:38" line-data="    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It&#39;s an encoder decoder transformer pre-trained in a text-to-text">`pre-trained`</SwmToken> in a <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1068:32:36" line-data="    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer">`Text-to-Text`</SwmToken> denoising generative setting, which means it can convert any text-based task into a <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1068:32:36" line-data="    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer">`Text-to-Text`</SwmToken> format. This allows for a unified approach to different NLP tasks such as translation, summarization, and question answering.

In the transformers library, the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model inherits from the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="43:7:7" line-data="from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer">`PreTrainedModel`</SwmToken> class, which provides generic methods for downloading, saving, and other common operations. It is also a subclass of PyTorch's <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="180:4:8" line-data="# - torch.nn.Module for the layers and">`torch.nn.Module`</SwmToken>, making it compatible with PyTorch's ecosystem.

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model uses a configuration class, <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="306:11:11" line-data="    def __init__(self, config: T5Config, has_relative_attention_bias=False):">`T5Config`</SwmToken>, to store all the parameters required for the model. Initializing the model with a configuration file does not load the weights; the weights need to be loaded separately using the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:9:9" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`from_pretrained`</SwmToken> method.

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model supports model parallelism, allowing it to distribute its attention modules across multiple devices. This is particularly useful for handling large models that cannot fit into the memory of a single device.

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model includes methods for parallelizing and deparallelizing the model, which helps in distributing the model across several devices and moving it back to the CPU, respectively.

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="1238">

---

## <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> Model Initialization

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1238:2:2" line-data="class T5Model(T5PreTrainedModel):">`T5Model`</SwmToken> class is initialized with a configuration object, which sets up the encoder and decoder stacks using the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1255:7:7" line-data="        self.encoder = T5Stack(encoder_config, self.shared)">`T5Stack`</SwmToken> class.

```python
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="1269">

---

## Model Parallelism

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1270:3:3" line-data="    def parallelize(self, device_map=None):">`parallelize`</SwmToken> and <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1282:3:3" line-data="    def deparallelize(self):">`deparallelize`</SwmToken> methods allow the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model to distribute its components across multiple devices and move them back to the CPU, respectively.

```python
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="1338">

---

## Example Usage

An example of how to use the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model for a <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1068:32:36" line-data="    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer">`Text-to-Text`</SwmToken> task, such as summarization or translation.

```python
            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
```

---

</SwmSnippet>

# Main functions

There are several main functions in the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model. Some of them are <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="283:2:2" line-data="class T5LayerFF(nn.Module):">`T5LayerFF`</SwmToken>, <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="305:2:2" line-data="class T5Attention(nn.Module):">`T5Attention`</SwmToken>, and <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1238:2:2" line-data="class T5Model(T5PreTrainedModel):">`T5Model`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="283:2:2" line-data="class T5LayerFF(nn.Module):">`T5LayerFF`</SwmToken> and <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="305:2:2" line-data="class T5Attention(nn.Module):">`T5Attention`</SwmToken>.

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="283">

---

## <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="283:2:2" line-data="class T5LayerFF(nn.Module):">`T5LayerFF`</SwmToken>

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="283:2:2" line-data="class T5LayerFF(nn.Module):">`T5LayerFF`</SwmToken> class implements the feed-forward layer of the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model. It includes a dense layer with <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="286:10:10" line-data="        if config.feed_forward_proj == &quot;relu&quot;:">`relu`</SwmToken> or <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="288:10:12" line-data="        elif config.feed_forward_proj == &quot;gated-gelu&quot;:">`gated-gelu`</SwmToken> activation, followed by layer normalization and dropout. This layer is used to process the hidden states in the model.

```python
class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="305">

---

## <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="305:2:2" line-data="class T5Attention(nn.Module):">`T5Attention`</SwmToken>

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="305:2:2" line-data="class T5Attention(nn.Module):">`T5Attention`</SwmToken> class implements the attention mechanism of the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model. It includes methods for computing attention scores, projecting hidden states, and handling relative position biases. This class is crucial for the model's ability to focus on different parts of the input sequence.

```python
class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/t5/modeling_t5.py" line="1238">

---

## <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1238:2:2" line-data="class T5Model(T5PreTrainedModel):">`T5Model`</SwmToken>

The <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1238:2:2" line-data="class T5Model(T5PreTrainedModel):">`T5Model`</SwmToken> class is the main class for the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model. It initializes the encoder and decoder stacks, manages model parallelism, and provides methods for loading and saving the model. This class is the entry point for using the <SwmToken path="src/transformers/models/t5/modeling_t5.py" pos="1340:12:12" line-data="            &gt;&gt;&gt; tokenizer = T5Tokenizer.from_pretrained(&#39;t5-small&#39;)">`t5`</SwmToken> model in various NLP tasks.

```python
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
