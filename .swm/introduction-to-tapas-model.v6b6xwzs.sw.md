---
title: Introduction to TAPAS Model
---
The TAPAS model is designed to handle tabular data and perform tasks such as question answering directly on tables.

It extends the BERT architecture by incorporating additional token type <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken> to encode the tabular structure.

The model can function both as an encoder with <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="787:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> and as a decoder with <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="788:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> layers.

TAPAS models are available in various sizes, including large, base, small, and mini, each pretrained for specific tasks.

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:2:2" line-data="class TapasModel(TapasPreTrainedModel):">`TapasModel`</SwmToken> class inherits from <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="691:2:2" line-data="class TapasPreTrainedModel(PreTrainedModel):">`TapasPreTrainedModel`</SwmToken> and initializes components such as embeddings, encoder, and pooler.

The model's forward method processes input <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken>, attention masks, and token type <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken>, among other parameters, to produce outputs.

TAPAS also includes specialized classes like <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:2:2" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasForSequenceClassification`</SwmToken> and <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:2:2" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasForQuestionAnswering`</SwmToken> for specific tasks.

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="794">

---

Initialization The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:2:2" line-data="class TapasModel(TapasPreTrainedModel):">`TapasModel`</SwmToken> class initializes components such as embeddings, encoder, and pooler.

```python
    def __init__(self, config, add_pooling_layer=True):
        requires_backends(self, "scatter")
        super().__init__(config)
        self.config = config

        self.embeddings = TapasEmbeddings(config)
        self.encoder = TapasEncoder(config)

        self.pooler = TapasPooler(config) if add_pooling_layer else None

        self.init_weights()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="822">

---

Forward Method The model's forward method processes input <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken>, attention masks, and token type <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken>, among other parameters, to produce outputs.

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasModel
            >>> import pandas as pd
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="58">

---

Pretrained Models TAPAS models are available in various sizes, including large, base, small, and mini, each pretrained for specific tasks.

```python
TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # large models
    "google/tapas-large",
    "google/tapas-large-finetuned-sqa",
    "google/tapas-large-finetuned-wtq",
    "google/tapas-large-finetuned-wikisql-supervised",
    "google/tapas-large-finetuned-tabfact",
    # base models
    "google/tapas-base",
    "google/tapas-base-finetuned-sqa",
    "google/tapas-base-finetuned-wtq",
    "google/tapas-base-finetuned-wikisql-supervised",
    "google/tapas-base-finetuned-tabfact",
    # small models
    "google/tapas-small",
    "google/tapas-small-finetuned-sqa",
    "google/tapas-small-finetuned-wtq",
    "google/tapas-small-finetuned-wikisql-supervised",
    "google/tapas-small-finetuned-tabfact",
    # mini models
    "google/tapas-mini",
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="1382">

---

Specialized Classes TAPAS includes specialized classes like <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:2:2" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasForSequenceClassification`</SwmToken> for specific tasks.

```python
class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.tapas = TapasModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="1041">

---

Question Answering The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:2:2" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasForQuestionAnswering`</SwmToken> class is designed for question answering tasks on tabular data.

```python
class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config: TapasConfig):
        super().__init__(config)

        # base model
        self.tapas = TapasModel(config)

        # dropout (only used when training)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # cell selection heads
        if config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.output_weights, std=config.initializer_range
            )  # here, a truncated normal is used in the original implementation
```

---

</SwmSnippet>

# Main functions

There are several main functions in the TAPAS model. Some of them are <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="253:2:2" line-data="class TapasEmbeddings(nn.Module):">`TapasEmbeddings`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="333:2:2" line-data="class TapasSelfAttention(nn.Module):">`TapasSelfAttention`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="444:2:2" line-data="class TapasAttention(nn.Module):">`TapasAttention`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="494:2:2" line-data="class TapasIntermediate(nn.Module):">`TapasIntermediate`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="510:2:2" line-data="class TapasOutput(nn.Module):">`TapasOutput`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="525:2:2" line-data="class TapasLayer(nn.Module):">`TapasLayer`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="608:2:2" line-data="class TapasEncoder(nn.Module):">`TapasEncoder`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="676:2:2" line-data="class TapasPooler(nn.Module):">`TapasPooler`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="691:2:2" line-data="class TapasPreTrainedModel(PreTrainedModel):">`TapasPreTrainedModel`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:2:2" line-data="class TapasModel(TapasPreTrainedModel):">`TapasModel`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="933:2:2" line-data="class TapasForMaskedLM(TapasPreTrainedModel):">`TapasForMaskedLM`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:2:2" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasForQuestionAnswering`</SwmToken>, and <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:2:2" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasForSequenceClassification`</SwmToken>. We will dive a little into each of these functions.

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="253">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="253:2:2" line-data="class TapasEmbeddings(nn.Module):">`TapasEmbeddings`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="253:2:2" line-data="class TapasEmbeddings(nn.Module):">`TapasEmbeddings`</SwmToken> class constructs embeddings from word, position, and token type embeddings. It is similar to <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="255:25:25" line-data="    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of">`BertEmbeddings`</SwmToken> but includes additional token type embeddings to encode tabular structure.

```python
class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        super().__init__()
        # we do not include config.disabled_features and config.disable_position_embeddings from the original implementation
        # word embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token type embeddings
        for i, type_vocab_sizes in enumerate(config.type_vocab_sizes):
            name = f"token_type_embeddings_{i}"
            setattr(self, name, nn.Embedding(type_vocab_sizes, config.hidden_size))

        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="333">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="333:2:2" line-data="class TapasSelfAttention(nn.Module):">`TapasSelfAttention`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="333:2:2" line-data="class TapasSelfAttention(nn.Module):">`TapasSelfAttention`</SwmToken> class implements the <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="787:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> mechanism. It includes methods for computing query, key, and value projections and handling <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="788:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> if encoder hidden states are provided.

```python
class TapasSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="443">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="444:2:2" line-data="class TapasAttention(nn.Module):">`TapasAttention`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="444:2:2" line-data="class TapasAttention(nn.Module):">`TapasAttention`</SwmToken> class combines <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="787:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken> and output layers. It includes methods for pruning heads and forwarding hidden states through the attention mechanism.

```python
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Tapas
class TapasAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TapasSelfAttention(config)
        self.output = TapasSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="493">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="494:2:2" line-data="class TapasIntermediate(nn.Module):">`TapasIntermediate`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="494:2:2" line-data="class TapasIntermediate(nn.Module):">`TapasIntermediate`</SwmToken> class implements the intermediate layer of the transformer. It applies a linear transformation followed by an activation function.

```python
# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TapasIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="509">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="510:2:2" line-data="class TapasOutput(nn.Module):">`TapasOutput`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="510:2:2" line-data="class TapasOutput(nn.Module):">`TapasOutput`</SwmToken> class implements the output layer of the transformer. It applies a linear transformation, dropout, and layer normalization.

```python
# Copied from transformers.models.bert.modeling_bert.BertOutput
class TapasOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="524">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="525:2:2" line-data="class TapasLayer(nn.Module):">`TapasLayer`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="525:2:2" line-data="class TapasLayer(nn.Module):">`TapasLayer`</SwmToken> class represents a single layer of the transformer. It includes <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="787:20:22" line-data="    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of">`self-attention`</SwmToken>, <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="788:1:3" line-data="    cross-attention is added between the self-attention layers, following the architecture described in `Attention is">`cross-attention`</SwmToken> (if applicable), intermediate, and output sub-layers.

```python
# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Tapas
class TapasLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TapasAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = TapasAttention(config)
        self.intermediate = TapasIntermediate(config)
        self.output = TapasOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="608">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="608:2:2" line-data="class TapasEncoder(nn.Module):">`TapasEncoder`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="608:2:2" line-data="class TapasEncoder(nn.Module):">`TapasEncoder`</SwmToken> class stacks multiple <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="612:12:12" line-data="        self.layer = nn.ModuleList([TapasLayer(config) for _ in range(config.num_hidden_layers)])">`TapasLayer`</SwmToken> instances to form the encoder. It processes hidden states through each layer and optionally returns all hidden states and attentions.

```python
class TapasEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TapasLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="675">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="676:2:2" line-data="class TapasPooler(nn.Module):">`TapasPooler`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="676:2:2" line-data="class TapasPooler(nn.Module):">`TapasPooler`</SwmToken> class implements the pooling layer. It extracts the hidden state corresponding to the first token and applies a linear transformation followed by a tanh activation.

```python
# Copied from transformers.models.bert.modeling_bert.BertPooler
class TapasPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="691">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="691:2:2" line-data="class TapasPreTrainedModel(PreTrainedModel):">`TapasPreTrainedModel`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="691:2:2" line-data="class TapasPreTrainedModel(PreTrainedModel):">`TapasPreTrainedModel`</SwmToken> class is an abstract class that handles weights initialization and provides a simple interface for downloading and loading pretrained models.

```python
class TapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="782">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:2:2" line-data="class TapasModel(TapasPreTrainedModel):">`TapasModel`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:2:2" line-data="class TapasModel(TapasPreTrainedModel):">`TapasModel`</SwmToken> class extends <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="782:4:4" line-data="class TapasModel(TapasPreTrainedModel):">`TapasPreTrainedModel`</SwmToken> and initializes components such as embeddings, encoder, and pooler. It processes input <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken>, attention masks, and token type <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="785:3:3" line-data="    type ids.">`ids`</SwmToken> to produce outputs.

```python
class TapasModel(TapasPreTrainedModel):
    """
    This class is a small change compared to :class:`~transformers.BertModel`, taking into account the additional token
    type ids.

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    def __init__(self, config, add_pooling_layer=True):
        requires_backends(self, "scatter")
        super().__init__(config)
        self.config = config

        self.embeddings = TapasEmbeddings(config)
        self.encoder = TapasEncoder(config)

        self.pooler = TapasPooler(config) if add_pooling_layer else None
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="933">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="933:2:2" line-data="class TapasForMaskedLM(TapasPreTrainedModel):">`TapasForMaskedLM`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="933:2:2" line-data="class TapasForMaskedLM(TapasPreTrainedModel):">`TapasForMaskedLM`</SwmToken> class extends <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="933:4:4" line-data="class TapasForMaskedLM(TapasPreTrainedModel):">`TapasPreTrainedModel`</SwmToken> and adds a language modeling head on top. It is used for masked language modeling tasks.

```python
class TapasForMaskedLM(TapasPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        self.tapas = TapasModel(config, add_pooling_layer=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, word_embeddings):
        self.lm_head = word_embeddings

    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="1041">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:2:2" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasForQuestionAnswering`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:2:2" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasForQuestionAnswering`</SwmToken> class extends <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1041:4:4" line-data="class TapasForQuestionAnswering(TapasPreTrainedModel):">`TapasPreTrainedModel`</SwmToken> and adds heads for cell selection and aggregation. It is used for question answering tasks on tables.

```python
class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config: TapasConfig):
        super().__init__(config)

        # base model
        self.tapas = TapasModel(config)

        # dropout (only used when training)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # cell selection heads
        if config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.output_weights, std=config.initializer_range
            )  # here, a truncated normal is used in the original implementation
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/tapas/modeling_tapas.py" line="1382">

---

### <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:2:2" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasForSequenceClassification`</SwmToken>

The <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:2:2" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasForSequenceClassification`</SwmToken> class extends <SwmToken path="src/transformers/models/tapas/modeling_tapas.py" pos="1382:4:4" line-data="class TapasForSequenceClassification(TapasPreTrainedModel):">`TapasPreTrainedModel`</SwmToken> and adds a classification head. It is used for sequence classification tasks.

```python
class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.tapas = TapasModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
