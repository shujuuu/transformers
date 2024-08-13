---
title: Getting started with BERT Model Implementation
---
The BERT model is implemented as a class called <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> which inherits from <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:4:4" line-data="class BertModel(BertPreTrainedModel):">`BertPreTrainedModel`</SwmToken>. This class can function both as an encoder and a decoder, depending on the configuration parameters.

The <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> class includes several components such as embeddings, encoder, and an optional pooling layer. These components are initialized in the constructor.

The forward method of <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> handles various inputs like <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="890:1:1" line-data="        input_ids=None,">`input_ids`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="891:1:1" line-data="        attention_mask=None,">`attention_mask`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="892:1:1" line-data="        token_type_ids=None,">`token_type_ids`</SwmToken>, and others. It processes these inputs to produce the model's outputs, which can include hidden states and attention scores.

The BERT model is used in various other classes and modules within the repository, such as <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1580:2:2" line-data="class BertForMultipleChoice(BertPreTrainedModel):">`BertForMultipleChoice`</SwmToken>, and <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1127:2:2" line-data="class BertLMHeadModel(BertPreTrainedModel):">`BertLMHeadModel`</SwmToken>, to perform specific NLP tasks.

<SwmSnippet path="/src/transformers/models/bert/modeling_bert.py" line="842">

---

## Initialization

The <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> class is initialized with a configuration object and optional pooling layer. This sets up the embeddings, encoder, and pooling components.

```python
class BertModel(BertPreTrainedModel):
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
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/bert/modeling_bert.py" line="888">

---

## Forward Method

The <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="888:3:3" line-data="    def forward(">`forward`</SwmToken> method processes inputs like <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="890:1:1" line-data="        input_ids=None,">`input_ids`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="891:1:1" line-data="        attention_mask=None,">`attention_mask`</SwmToken>, and <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="892:1:1" line-data="        token_type_ids=None,">`token_type_ids`</SwmToken> to produce outputs such as hidden states and attention scores.

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

<SwmSnippet path="/src/transformers/models/bert/modeling_bert.py" line="1482">

---

## Usage in Other Classes

The BERT model is used in various other classes like <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken> to perform specific NLP tasks. This class fine-tunes BERT for sequence classification tasks.

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
```

---

</SwmSnippet>

# Main functions

There are several main functions in the BERT model. Some of them are <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1580:2:2" line-data="class BertForMultipleChoice(BertPreTrainedModel):">`BertForMultipleChoice`</SwmToken>, and <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1127:2:2" line-data="class BertLMHeadModel(BertPreTrainedModel):">`BertLMHeadModel`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> and <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken>.

<SwmSnippet path="/src/transformers/models/bert/modeling_bert.py" line="842">

---

## <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken>

The <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="842:2:2" line-data="class BertModel(BertPreTrainedModel):">`BertModel`</SwmToken> class can function both as an encoder and a decoder, depending on the configuration parameters. It includes components such as embeddings, encoder, and an optional pooling layer. The forward method handles various inputs like <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="890:1:1" line-data="        input_ids=None,">`input_ids`</SwmToken>, <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="891:1:1" line-data="        attention_mask=None,">`attention_mask`</SwmToken>, and <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="892:1:1" line-data="        token_type_ids=None,">`token_type_ids`</SwmToken> to produce the model's outputs, which can include hidden states and attention scores.

```python
class BertModel(BertPreTrainedModel):
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
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/models/bert/modeling_bert.py" line="1482">

---

## <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken>

The <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1482:2:2" line-data="class BertForSequenceClassification(BertPreTrainedModel):">`BertForSequenceClassification`</SwmToken> class is used for sequence classification tasks. It includes the <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="1488:7:7" line-data="        self.bert = BertModel(config)">`BertModel`</SwmToken> as its base and adds a dropout layer and a classifier on top. The forward method processes inputs and produces logits for classification.

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
```

---

</SwmSnippet>

&nbsp;

*This is an* <SwmToken path="src/transformers/models/bert/modeling_bert.py" pos="204:7:9" line-data="        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves">`auto-generated`</SwmToken> *document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
