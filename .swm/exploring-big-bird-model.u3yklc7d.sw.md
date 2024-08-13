---
title: Exploring Big Bird Model
---
The Big Bird Model is a transformer-based model designed to handle long sequences efficiently. It achieves this by using a sparse attention mechanism, which reduces the computational complexity compared to traditional transformers.

The model can function as both an encoder and a decoder. When used as a decoder, it includes a layer of <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1918:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> between the <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1917:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> layers.

The Big Bird Model supports various configurations, such as <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="332:3:3" line-data="        self.is_decoder = config.is_decoder">`is_decoder`</SwmToken> and <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1445:3:3" line-data="        self.add_cross_attention = config.add_cross_attention">`add_cross_attention`</SwmToken>, to adapt its behavior for different tasks, including sequence-to-sequence modeling.

The model's architecture includes components like embeddings, encoder, and optionally a pooling layer. It also allows setting different attention types, such as <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1328:12:12" line-data="        if self.config.attention_type == &quot;original_full&quot;:">`original_full`</SwmToken> or <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1330:12:12" line-data="        elif self.config.attention_type == &quot;block_sparse&quot;:">`block_sparse`</SwmToken>.

The Big Bird Model is integrated with <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="15:4:4" line-data="&quot;&quot;&quot; PyTorch BigBird model. &quot;&quot;&quot;">`PyTorch`</SwmToken> and can be used with the <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="56:5:5" line-data="_TOKENIZER_FOR_DOC = &quot;BigBirdTokenizer&quot;">`BigBirdTokenizer`</SwmToken> for tokenizing input sequences. It supports various downstream tasks like sequence classification, token classification, and causal language modeling.

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1914">

---

The Big Bird Model can behave as both an encoder and a decoder, with <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1918:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> layers added when used as a decoder.

```python
class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1928">

---

The model's architecture includes embeddings, encoder, and optionally a pooling layer. It also allows setting different attention types.

```python
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.attention_type = self.config.attention_type
        self.config = config

        self.block_size = self.config.block_size

        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.pooler = None
            self.activation = None

        if self.attention_type != "original_full" and config.add_cross_attention:
            logger.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting `attention_type=original_full`"
            )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1977">

---

The forward method of the Big Bird Model, which handles the input sequences and various configurations.

```python
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/configuration_big_bird.py" line="32">

---

The configuration class for the Big Bird Model, which defines the model architecture and various parameters.

```python
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BigBirdModel`. It is used to
    instantiate an BigBird model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BigBird
    `google/bigbird-roberta-base <https://huggingface.co/google/bigbird-roberta-base>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50358):
            Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BigBirdModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="2356">

---

An example of the Big Bird Model being used for masked language modeling.

```python
class BigBirdForMaskedLM(BigBirdPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BigBirdForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

```

---

</SwmSnippet>

# Main functions

There are several main functions in the Big Bird Model. Some of them are <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="244:2:2" line-data="class BigBirdEmbeddings(nn.Module):">`BigBirdEmbeddings`</SwmToken>, <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="314:2:2" line-data="class BigBirdSelfAttention(nn.Module):">`BigBirdSelfAttention`</SwmToken>, <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1321:2:2" line-data="class BigBirdAttention(nn.Module):">`BigBirdAttention`</SwmToken>, <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1436:2:2" line-data="class BigBirdLayer(nn.Module):">`BigBirdLayer`</SwmToken>, <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1549:2:2" line-data="class BigBirdEncoder(nn.Module):">`BigBirdEncoder`</SwmToken>, and <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1914:2:2" line-data="class BigBirdModel(BigBirdPreTrainedModel):">`BigBirdModel`</SwmToken>. We will dive a little into each of these functions.

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="244">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="244:2:2" line-data="class BigBirdEmbeddings(nn.Module):">`BigBirdEmbeddings`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="244:2:2" line-data="class BigBirdEmbeddings(nn.Module):">`BigBirdEmbeddings`</SwmToken> class constructs the embeddings from word, position, and token type embeddings. It includes methods for initializing and forward propagation.

```python
class BigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="314">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="314:2:2" line-data="class BigBirdSelfAttention(nn.Module):">`BigBirdSelfAttention`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="314:2:2" line-data="class BigBirdSelfAttention(nn.Module):">`BigBirdSelfAttention`</SwmToken> class implements the <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1917:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> mechanism for the Big Bird model. It includes methods for initializing and forward propagation, handling both regular and <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1918:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken>.

```python
class BigBirdSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1321">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1321:2:2" line-data="class BigBirdAttention(nn.Module):">`BigBirdAttention`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1321:2:2" line-data="class BigBirdAttention(nn.Module):">`BigBirdAttention`</SwmToken> class combines <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1917:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> and output layers. It supports different attention types, such as <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1328:12:12" line-data="        if self.config.attention_type == &quot;original_full&quot;:">`original_full`</SwmToken> and <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1330:12:12" line-data="        elif self.config.attention_type == &quot;block_sparse&quot;:">`block_sparse`</SwmToken>, and includes methods for setting the attention type and forward propagation.

```python
class BigBirdAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed

        if self.config.attention_type == "original_full":
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == "block_sparse":
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1436">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1436:2:2" line-data="class BigBirdLayer(nn.Module):">`BigBirdLayer`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1436:2:2" line-data="class BigBirdLayer(nn.Module):">`BigBirdLayer`</SwmToken> class represents a single layer in the Big Bird model. It includes attention, intermediate, and output sub-layers, and supports <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1918:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> when used as a decoder.

```python
class BigBirdLayer(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BigBirdAttention(config)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1549">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1549:2:2" line-data="class BigBirdEncoder(nn.Module):">`BigBirdEncoder`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1549:2:2" line-data="class BigBirdEncoder(nn.Module):">`BigBirdEncoder`</SwmToken> class consists of multiple <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1556:2:2" line-data="            [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]">`BigBirdLayer`</SwmToken> instances. It includes methods for setting the attention type and forward propagation through all layers.

```python
class BigBirdEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type

        self.layer = nn.ModuleList(
            [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/big_bird/modeling_big_bird.py" line="1914">

---

### <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1914:2:2" line-data="class BigBirdModel(BigBirdPreTrainedModel):">`BigBirdModel`</SwmToken>

The <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="1914:2:2" line-data="class BigBirdModel(BigBirdPreTrainedModel):">`BigBirdModel`</SwmToken> class is the main model class that integrates embeddings, encoder, and optionally a pooling layer. It supports different configurations and attention types, and includes methods for initializing weights and forward propagation.

```python
class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.attention_type = self.config.attention_type
        self.config = config

        self.block_size = self.config.block_size

```

---

</SwmSnippet>

&nbsp;

*This is an* <SwmToken path="src/transformers/models/big_bird/modeling_big_bird.py" pos="286:7:9" line-data="        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves">`auto-generated`</SwmToken> *document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
