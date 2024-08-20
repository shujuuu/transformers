---
title: What is Model Conversion
---
Model Conversion refers to the process of converting machine learning model checkpoints between different frameworks, such as <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> and <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken>.

The <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken> function is used to create a command for converting a <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> <SwmToken path="src/transformers/commands/convert.py" pos="23:17:19" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`1.0`</SwmToken> checkpoint into a <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken> checkpoint.

The <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken> class handles the registration of the conversion command with the argument parser, making it available for the <SwmToken path="src/transformers/commands/convert.py" pos="42:23:25" line-data="        Register this command to argparse so it&#39;s available for the transformer-cli">`transformer-cli`</SwmToken>.

The <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken> class also defines the <SwmToken path="src/transformers/commands/convert.py" pos="49:10:10" line-data="            help=&quot;CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.&quot;,">`run`</SwmToken> method, which performs the actual conversion of the model checkpoint based on the specified model type.

The conversion process involves loading the model from the <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> checkpoint, converting it to the <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken> format, and saving the converted model to the specified output path.

## Convert a model for all frameworks

To ensure your model can be used by someone working with a different framework, we recommend you convert and upload your model with both <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken> and <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> checkpoints. While users are still able to load your model from a different framework if you skip this step, it will be slower because 🤗 Transformers will need to convert the checkpoint on-the-fly.

<SwmSnippet path="/src/transformers/commands/convert.py" line="21">

---

The <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken> function is used to create a command for converting a <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> <SwmToken path="src/transformers/commands/convert.py" pos="23:17:19" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`1.0`</SwmToken> checkpoint into a <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken> checkpoint.

```python
def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ServeCommand
    """
    return ConvertCommand(
        args.model_type, args.tf_checkpoint, args.pytorch_dump_output, args.config, args.finetuning_task_name
    )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/convert.py" line="38">

---

The <SwmToken path="src/transformers/commands/convert.py" pos="38:2:2" line-data="class ConvertCommand(BaseTransformersCLICommand):">`ConvertCommand`</SwmToken> class handles the registration of the conversion command with the argument parser, making it available for the <SwmToken path="src/transformers/commands/convert.py" pos="42:23:25" line-data="        Register this command to argparse so it&#39;s available for the transformer-cli">`transformer-cli`</SwmToken>.

```python
class ConvertCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "convert",
            help="CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.",
        )
        train_parser.add_argument("--model_type", type=str, required=True, help="Model's type.")
        train_parser.add_argument(
            "--tf_checkpoint", type=str, required=True, help="TensorFlow checkpoint path or folder."
        )
        train_parser.add_argument(
            "--pytorch_dump_output", type=str, required=True, help="Path to the PyTorch saved model output."
        )
        train_parser.add_argument("--config", type=str, default="", help="Configuration file path or folder.")
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/convert.py" line="67">

---

The <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken> class also defines the <SwmToken path="src/transformers/commands/convert.py" pos="49:10:10" line-data="            help=&quot;CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.&quot;,">`run`</SwmToken> method, which performs the actual conversion of the model checkpoint based on the specified model type.

```python
    def __init__(
        self,
        model_type: str,
        tf_checkpoint: str,
        pytorch_dump_output: str,
        config: str,
        finetuning_task_name: str,
        *args,
    ):
        self._logger = logging.get_logger("transformers-cli/converting")

        self._logger.info(f"Loading model {model_type}")
        self._model_type = model_type
        self._tf_checkpoint = tf_checkpoint
        self._pytorch_dump_output = pytorch_dump_output
        self._config = config
        self._finetuning_task_name = finetuning_task_name

```

---

</SwmSnippet>

# Main functions

There are several main functions in this folder. Some of them are <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken> and <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken>. We will dive a little into <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken> and <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken>.

<SwmSnippet path="/src/transformers/commands/convert.py" line="21">

---

## <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken>

The <SwmToken path="src/transformers/commands/convert.py" pos="21:2:2" line-data="def convert_command_factory(args: Namespace):">`convert_command_factory`</SwmToken> function is used to create a command for converting a <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> <SwmToken path="src/transformers/commands/convert.py" pos="23:17:19" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`1.0`</SwmToken> checkpoint into a <SwmToken path="src/transformers/commands/convert.py" pos="23:27:27" line-data="    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.">`PyTorch`</SwmToken> checkpoint. It returns a <SwmToken path="src/transformers/commands/convert.py" pos="27:3:3" line-data="    return ConvertCommand(">`ConvertCommand`</SwmToken> instance with the necessary arguments for the conversion process.

```python
def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ServeCommand
    """
    return ConvertCommand(
        args.model_type, args.tf_checkpoint, args.pytorch_dump_output, args.config, args.finetuning_task_name
    )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/convert.py" line="38">

---

## <SwmToken path="src/transformers/commands/convert.py" pos="38:2:2" line-data="class ConvertCommand(BaseTransformersCLICommand):">`ConvertCommand`</SwmToken>

The <SwmToken path="src/transformers/commands/convert.py" pos="38:2:2" line-data="class ConvertCommand(BaseTransformersCLICommand):">`ConvertCommand`</SwmToken> class handles the registration of the conversion command with the argument parser, making it available for the <SwmToken path="src/transformers/commands/convert.py" pos="42:23:25" line-data="        Register this command to argparse so it&#39;s available for the transformer-cli">`transformer-cli`</SwmToken>. It defines the <SwmToken path="src/transformers/commands/convert.py" pos="49:10:10" line-data="            help=&quot;CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.&quot;,">`run`</SwmToken> method, which performs the actual conversion of the model checkpoint based on the specified model type. The conversion process involves loading the model from the <SwmToken path="src/transformers/commands/convert.py" pos="53:20:20" line-data="            &quot;--tf_checkpoint&quot;, type=str, required=True, help=&quot;TensorFlow checkpoint path or folder.&quot;">`TensorFlow`</SwmToken> checkpoint, converting it to the <SwmToken path="src/transformers/commands/convert.py" pos="49:28:28" line-data="            help=&quot;CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.&quot;,">`PyTorch`</SwmToken> format, and saving the converted model to the specified output path.

```python
class ConvertCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "convert",
            help="CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.",
        )
        train_parser.add_argument("--model_type", type=str, required=True, help="Model's type.")
        train_parser.add_argument(
            "--tf_checkpoint", type=str, required=True, help="TensorFlow checkpoint path or folder."
        )
        train_parser.add_argument(
            "--pytorch_dump_output", type=str, required=True, help="Path to the PyTorch saved model output."
        )
        train_parser.add_argument("--config", type=str, default="", help="Configuration file path or folder.")
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
