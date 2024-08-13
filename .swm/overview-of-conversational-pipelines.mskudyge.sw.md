---
title: Overview of Conversational Pipelines
---
Conversational refers to the functionality provided by the <SwmToken path="src/transformers/pipelines/conversational.py" pos="22:6:6" line-data="    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the">`ConversationalPipeline`</SwmToken> class, which is designed for handling <SwmToken path="src/transformers/pipelines/conversational.py" pos="166:1:3" line-data="    Multi-turn conversational pipeline.">`Multi-turn`</SwmToken> conversations.

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="22:6:6" line-data="    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the">`ConversationalPipeline`</SwmToken> can be instantiated using the <SwmToken path="src/transformers/pipelines/conversational.py" pos="38:1:1" line-data="            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and">`pipeline`</SwmToken> function with the task identifier <SwmToken path="src/transformers/pipelines/conversational.py" pos="166:5:5" line-data="    Multi-turn conversational pipeline.">`conversational`</SwmToken>.

This pipeline supports models <SwmToken path="src/transformers/pipelines/conversational.py" pos="171:25:27" line-data="    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,">`fine-tuned`</SwmToken> on <SwmToken path="src/transformers/pipelines/conversational.py" pos="166:1:3" line-data="    Multi-turn conversational pipeline.">`Multi-turn`</SwmToken> conversational tasks, such as <SwmToken path="src/transformers/pipelines/conversational.py" pos="172:6:10" line-data="    currently: `&#39;microsoft/DialoGPT-small&#39;`, `&#39;microsoft/DialoGPT-medium&#39;`, `&#39;microsoft/DialoGPT-large&#39;`. See the">`microsoft/DialoGPT-small`</SwmToken>, <SwmToken path="src/transformers/pipelines/conversational.py" pos="172:16:20" line-data="    currently: `&#39;microsoft/DialoGPT-small&#39;`, `&#39;microsoft/DialoGPT-medium&#39;`, `&#39;microsoft/DialoGPT-large&#39;`. See the">`microsoft/DialoGPT-medium`</SwmToken>, and <SwmToken path="src/transformers/pipelines/conversational.py" pos="172:26:30" line-data="    currently: `&#39;microsoft/DialoGPT-small&#39;`, `&#39;microsoft/DialoGPT-medium&#39;`, `&#39;microsoft/DialoGPT-large&#39;`. See the">`microsoft/DialoGPT-large`</SwmToken>.

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="19:2:2" line-data="class Conversation:">`Conversation`</SwmToken> class is used to manage the conversation history and user inputs, which are then processed by the <SwmToken path="src/transformers/pipelines/conversational.py" pos="22:6:6" line-data="    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the">`ConversationalPipeline`</SwmToken>.

A <SwmToken path="src/transformers/pipelines/conversational.py" pos="19:2:2" line-data="class Conversation:">`Conversation`</SwmToken> instance contains methods to add user inputs, mark inputs as processed, and append model-generated responses.

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="201:3:3" line-data="    def __call__(">`__call__`</SwmToken> method of <SwmToken path="src/transformers/pipelines/conversational.py" pos="22:6:6" line-data="    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the">`ConversationalPipeline`</SwmToken> generates responses for the given conversations by tokenizing the inputs, generating responses using the model, and updating the conversation history.

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="164">

---

## <SwmToken path="src/transformers/pipelines/conversational.py" pos="164:2:2" line-data="class ConversationalPipeline(Pipeline):">`ConversationalPipeline`</SwmToken> Class

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="164:2:2" line-data="class ConversationalPipeline(Pipeline):">`ConversationalPipeline`</SwmToken> class handles <SwmToken path="src/transformers/pipelines/conversational.py" pos="166:1:3" line-data="    Multi-turn conversational pipeline.">`Multi-turn`</SwmToken> conversational tasks. It can be instantiated using the <SwmToken path="src/transformers/pipelines/conversational.py" pos="164:4:4" line-data="class ConversationalPipeline(Pipeline):">`Pipeline`</SwmToken> function with the task identifier 'conversational'.

```python
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    This conversational pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"conversational"`.

    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,
    currently: `'microsoft/DialoGPT-small'`, `'microsoft/DialoGPT-medium'`, `'microsoft/DialoGPT-large'`. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=conversational>`__.

    Usage::

        conversational_pipeline = pipeline("conversational")

        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")

        conversational_pipeline([conversation_1, conversation_2])

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="19">

---

## Conversation Class

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="19:2:2" line-data="class Conversation:">`Conversation`</SwmToken> class manages the conversation history and user inputs. It contains methods to add user inputs, mark inputs as processed, and append model-generated responses.

```python
class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the
    addition of new user input and generated model responses. A conversation needs to contain an unprocessed user input
    before being passed to the :class:`~transformers.ConversationalPipeline`. This user input is either created when
    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response("input")` after a
    conversation turn.

    Arguments:
        text (:obj:`str`, `optional`):
            The initial user input to start the conversation. If not provided, a user input needs to be provided
            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can
            begin.
        conversation_id (:obj:`uuid.UUID`, `optional`):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.
        past_user_inputs (:obj:`List[str]`, `optional`):
            Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and
            :obj:`generated_responses` with equal length lists of strings
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="201">

---

## Generating Responses

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="201:3:3" line-data="    def __call__(">`__call__`</SwmToken> method of <SwmToken path="src/transformers/pipelines/conversational.py" pos="22:6:6" line-data="    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the">`ConversationalPipeline`</SwmToken> generates responses for the given conversations by tokenizing the inputs, generating responses using the model, and updating the conversation history.

```python
    def __call__(
        self,
        conversations: Union[Conversation, List[Conversation]],
        clean_up_tokenization_spaces=True,
        **generate_kwargs
    ):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are **init**, **call**, <SwmToken path="src/transformers/pipelines/conversational.py" pos="31:14:14" line-data="            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can">`add_user_input`</SwmToken>, <SwmToken path="src/transformers/pipelines/conversational.py" pos="109:3:3" line-data="    def mark_processed(self):">`mark_processed`</SwmToken>, and <SwmToken path="src/transformers/pipelines/conversational.py" pos="25:21:21" line-data="    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response(&quot;input&quot;)` after a">`append_response`</SwmToken>. We will dive a little into **init**, **call**, <SwmToken path="src/transformers/pipelines/conversational.py" pos="31:14:14" line-data="            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can">`add_user_input`</SwmToken>, <SwmToken path="src/transformers/pipelines/conversational.py" pos="109:3:3" line-data="    def mark_processed(self):">`mark_processed`</SwmToken>, and <SwmToken path="src/transformers/pipelines/conversational.py" pos="25:21:21" line-data="    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response(&quot;input&quot;)` after a">`append_response`</SwmToken>.

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="58">

---

## **init**

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="58:3:3" line-data="    def __init__(">`__init__`</SwmToken> function initializes a <SwmToken path="src/transformers/pipelines/conversational.py" pos="19:2:2" line-data="class Conversation:">`Conversation`</SwmToken> instance. It sets up the conversation ID, past user inputs, generated responses, and the new user input.

```python
    def __init__(
        self, text: str = None, conversation_id: uuid.UUID = None, past_user_inputs=None, generated_responses=None
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        if past_user_inputs is None:
            past_user_inputs = []
        if generated_responses is None:
            generated_responses = []

        self.uuid: uuid.UUID = conversation_id
        self.past_user_inputs: List[str] = past_user_inputs
        self.generated_responses: List[str] = generated_responses
        self.new_user_input: Optional[str] = text

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="201">

---

## **call**

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="201:3:3" line-data="    def __call__(">`__call__`</SwmToken> function generates responses for the given conversations. It tokenizes the inputs, generates responses using the model, and updates the conversation history.

```python
    def __call__(
        self,
        conversations: Union[Conversation, List[Conversation]],
        clean_up_tokenization_spaces=True,
        **generate_kwargs
    ):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="84">

---

## <SwmToken path="src/transformers/pipelines/conversational.py" pos="84:3:3" line-data="    def add_user_input(self, text: str, overwrite: bool = False):">`add_user_input`</SwmToken>

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="84:3:3" line-data="    def add_user_input(self, text: str, overwrite: bool = False):">`add_user_input`</SwmToken> function adds a user input to the conversation for the next round. It populates the internal <SwmToken path="src/transformers/pipelines/conversational.py" pos="86:35:35" line-data="        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`">`new_user_input`</SwmToken> field.

```python
    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
        field.

        Args:
            text (:obj:`str`): The user input for the next conversation round.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not existing and unprocessed user input should be overwritten when this function is called.
        """
        if self.new_user_input:
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" was overwritten '
                    f'with: "{text}".'
                )
                self.new_user_input = text
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="109">

---

## <SwmToken path="src/transformers/pipelines/conversational.py" pos="109:3:3" line-data="    def mark_processed(self):">`mark_processed`</SwmToken>

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="109:3:3" line-data="    def mark_processed(self):">`mark_processed`</SwmToken> function marks the conversation as processed by moving the content of <SwmToken path="src/transformers/pipelines/conversational.py" pos="111:23:23" line-data="        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and">`new_user_input`</SwmToken> to <SwmToken path="src/transformers/pipelines/conversational.py" pos="111:31:31" line-data="        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and">`past_user_inputs`</SwmToken> and emptying the <SwmToken path="src/transformers/pipelines/conversational.py" pos="111:23:23" line-data="        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and">`new_user_input`</SwmToken> field.

```python
    def mark_processed(self):
        """
        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
        empties the :obj:`new_user_input` field.
        """
        if self.new_user_input:
            self.past_user_inputs.append(self.new_user_input)
        self.new_user_input = None

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/pipelines/conversational.py" line="118">

---

## <SwmToken path="src/transformers/pipelines/conversational.py" pos="118:3:3" line-data="    def append_response(self, response: str):">`append_response`</SwmToken>

The <SwmToken path="src/transformers/pipelines/conversational.py" pos="118:3:3" line-data="    def append_response(self, response: str):">`append_response`</SwmToken> function appends a model-generated response to the list of generated responses.

```python
    def append_response(self, response: str):
        """
        Append a response to the list of generated responses.

        Args:
            response (:obj:`str`): The model generated response.
        """
        self.generated_responses.append(response)

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
