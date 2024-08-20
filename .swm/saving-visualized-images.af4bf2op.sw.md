---
title: Saving Visualized Images
---
This document provides an overview of the <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="195:3:3" line-data="    def save(self, saveas=None):">`save`</SwmToken> function, which is responsible for saving visualized images to a file. It explains the flow of how the image is processed and saved, including the handling of different file formats and the generation of image buffers.

The <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="195:3:3" line-data="    def save(self, saveas=None):">`save`</SwmToken> function is used to save visualized images to a file. It first checks the file extension to determine the appropriate method for saving the image. If the file is a <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="198:12:13" line-data="        if saveas.lower().endswith(&quot;.jpg&quot;) or saveas.lower().endswith(&quot;.png&quot;):">`.jpg`</SwmToken> or <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="198:28:29" line-data="        if saveas.lower().endswith(&quot;.jpg&quot;) or saveas.lower().endswith(&quot;.png&quot;):">`.png`</SwmToken>, it uses OpenCV to save the image. For other formats, it uses Matplotlib's <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="204:5:5" line-data="            self.fig.savefig(saveas)">`savefig`</SwmToken> method. The function relies on the <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="201:3:3" line-data="                self._get_buffer()[:, :, ::-1],">`_get_buffer`</SwmToken> function to generate the image buffer, which handles different backends and resizes the image if necessary. The buffer combines the RGB and alpha channels to ensure the final image is correctly visualized. This flow ensures that the image is saved in the desired format and quality.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      subgraph examples/research_projects
a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle --> 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle
end

subgraph examples/research_projects
2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle --> 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle
end

subgraph hubconf.py
244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> db236d740ba4d332c6343d4b41cab080719359f6c64a85a915244ab1c02cc99f(model)
end

subgraph examples/research_projects
244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle
end

subgraph examples/legacy/run_swag.py
7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle
end

subgraph examples/research_projects
9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle --> d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle
end

subgraph hubconf.py
d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle --> a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle
end

subgraph src/transformers
a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle --> 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle
end

subgraph src/transformers
12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 2221acf97ae37a25b7480b354c46cde1d906c4bf6b7a078105fe2cc413f863dd(load_pytorch_checkpoint_in_tf2_model)
end

subgraph src/transformers
12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle
end

subgraph src/transformers
1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle
end

subgraph src/transformers
db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle --> 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle
end

subgraph src/transformers
4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle --> 4eb0740a562b1d1a86d862174ad24d68b3321eff314e61987d10f38bf8c857b1(evaluate_ast):::mainFlowStyle
end


      ClassDef mainFlowStyle color:#000000,fill:#7CB9F4
ClassDef rootsStyle color:#000000,fill:#00FFF4
ClassDef Style1 color:#000000,fill:#00FFAA
ClassDef Style2 color:#000000,fill:#FFFF00
ClassDef Style3 color:#000000,fill:#AA7CB9

%% Swimm:
%% graph TD;
%%       subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle --> 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle --> 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[hubconf.py](hubconf.py)</SwmPath>
%% 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> db236d740ba4d332c6343d4b41cab080719359f6c64a85a915244ab1c02cc99f(model)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/legacy/run_swag.py](examples/legacy/run_swag.py)</SwmPath>
%% 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle --> d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[hubconf.py](hubconf.py)</SwmPath>
%% d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle --> a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle --> 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 2221acf97ae37a25b7480b354c46cde1d906c4bf6b7a078105fe2cc413f863dd(load_pytorch_checkpoint_in_tf2_model)
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle --> 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle --> 4eb0740a562b1d1a86d862174ad24d68b3321eff314e61987d10f38bf8c857b1(evaluate_ast):::mainFlowStyle
%% end
%% 
%% 
%%       <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> mainFlowStyle color:#000000,fill:#7CB9F4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> rootsStyle color:#000000,fill:#00FFF4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style1 color:#000000,fill:#00FFAA
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style2 color:#000000,fill:#FFFF00
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style3 color:#000000,fill:#AA7CB9
```

# Flow drill down

First, we'll zoom into this section of the flow:

```mermaid
graph TD;
      subgraph examples/research_projects
a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle --> 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle
end

subgraph examples/research_projects
2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle --> 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle
end

subgraph hubconf.py
244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> db236d740ba4d332c6343d4b41cab080719359f6c64a85a915244ab1c02cc99f(model)
end

subgraph examples/research_projects
244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle
end

subgraph examples/research_projects
7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9hz06(...)
end


      ClassDef mainFlowStyle color:#000000,fill:#7CB9F4
ClassDef rootsStyle color:#000000,fill:#00FFF4
ClassDef Style1 color:#000000,fill:#00FFAA
ClassDef Style2 color:#000000,fill:#FFFF00
ClassDef Style3 color:#000000,fill:#AA7CB9

%% Swimm:
%% graph TD;
%%       subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle --> 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 2690078434f45befc7a2c808a3ed529e33b419db12f4252554df4afb5aba8505(_get_buffer):::mainFlowStyle --> 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[hubconf.py](hubconf.py)</SwmPath>
%% 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> db236d740ba4d332c6343d4b41cab080719359f6c64a85a915244ab1c02cc99f(model)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 244dbd59cdd88624887f1b824e909882cfa60d957bca6af0189f489397ccda0b(evaluate):::mainFlowStyle --> 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9hz06(...)
%% end
%% 
%% 
%%       <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> mainFlowStyle color:#000000,fill:#7CB9F4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> rootsStyle color:#000000,fill:#00FFF4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style1 color:#000000,fill:#00FFAA
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style2 color:#000000,fill:#FFFF00
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/examples/research_projects/lxmert/visualizing_image.py" line="195">

---

## Save Function

The <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="195:3:3" line-data="    def save(self, saveas=None):">`save`</SwmToken> function is responsible for saving the visualized image to a file. It checks the file extension to determine whether to save the image using OpenCV (<SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="199:1:3" line-data="            cv2.imwrite(">`cv2.imwrite`</SwmToken>) for <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="198:12:13" line-data="        if saveas.lower().endswith(&quot;.jpg&quot;) or saveas.lower().endswith(&quot;.png&quot;):">`.jpg`</SwmToken> or <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="198:28:29" line-data="        if saveas.lower().endswith(&quot;.jpg&quot;) or saveas.lower().endswith(&quot;.png&quot;):">`.png`</SwmToken> files or using Matplotlib's <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="204:5:5" line-data="            self.fig.savefig(saveas)">`savefig`</SwmToken> for other formats.

```python
    def save(self, saveas=None):
        if saveas is None:
            saveas = self.saveas
        if saveas.lower().endswith(".jpg") or saveas.lower().endswith(".png"):
            cv2.imwrite(
                saveas,
                self._get_buffer()[:, :, ::-1],
            )
        else:
            self.fig.savefig(saveas)
```

---

</SwmSnippet>

<SwmSnippet path="/examples/research_projects/lxmert/visualizing_image.py" line="231">

---

## <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="231:3:3" line-data="    def _get_buffer(self):">`_get_buffer`</SwmToken> Function

The <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="231:3:3" line-data="    def _get_buffer(self):">`_get_buffer`</SwmToken> function generates the image buffer that is used by the <SwmToken path="examples/research_projects/lxmert/visualizing_image.py" pos="195:3:3" line-data="    def save(self, saveas=None):">`save`</SwmToken> function. It handles different backends and resizes the image if necessary. The function also processes the image to combine the RGB and alpha channels, ensuring the final image is correctly visualized.

```python
    def _get_buffer(self):
        if not self.pynb:
            s, (width, height) = self.canvas.print_to_buffer()
            if (width, height) != (self.width, self.height):
                img = cv2.resize(self.img, (width, height))
            else:
                img = self.img
        else:
            buf = io.BytesIO()  # works for cairo backend
            self.canvas.print_rgba(buf)
            width, height = self.width, self.height
            s = buf.getvalue()
            img = self.img

        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

```

---

</SwmSnippet>

<SwmSnippet path="/examples/research_projects/bert-loses-patience/run_glue_with_pabee.py" line="263">

---

## Evaluate Function

The <SwmToken path="examples/research_projects/bert-loses-patience/run_glue_with_pabee.py" pos="263:2:2" line-data="def evaluate(args, model, tokenizer, prefix=&quot;&quot;, patience=0):">`evaluate`</SwmToken> function is used to evaluate the model's performance on a given dataset. It handles different model types <SwmToken path="src/transformers/modeling_tf_utils.py" pos="1016:24:27" line-data="                    # Retrocompatibility patch: some embeddings are stored with the weights name (e.g. Bart&#39;s">`(e.g`</SwmToken>., ALBERT, BERT) and sets specific parameters like regression thresholds and patience. The function loads the evaluation dataset, sets up the evaluation data loader, and performs the evaluation loop, logging the results and saving them to a file.

```python
def evaluate(args, model, tokenizer, prefix="", patience=0):
    if args.model_type == "albert":
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif args.model_type == "bert":
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
```

---

</SwmSnippet>

<SwmSnippet path="/hubconf.py" line="73">

---

## Model Function

The <SwmToken path="hubconf.py" pos="73:2:2" line-data="def model(*args, **kwargs):">`model`</SwmToken> function uses <SwmToken path="hubconf.py" pos="75:5:7" line-data="            # Using torch.hub !">`torch.hub`</SwmToken> to load a <SwmToken path="src/transformers/modeling_tf_utils.py" pos="2557:19:21" line-data="        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.">`pre-trained`</SwmToken> model from Hugging Face's model hub. It supports loading models from different sources, including local directories and <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> checkpoints, and allows updating the model configuration during loading.

```python
def model(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased')    # Download model and configuration from huggingface.co and cache.
            model = torch.hub.load('huggingface/transformers', 'model', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModel.from_pretrained(*args, **kwargs)
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph examples/legacy/run_swag.py
7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle
end

subgraph examples/research_projects
9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle --> d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle
end

subgraph hubconf.py
d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle --> a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle
end

subgraph src/transformers
a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle --> 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle
end

subgraph src/transformers
12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 2221acf97ae37a25b7480b354c46cde1d906c4bf6b7a078105fe2cc413f863dd(load_pytorch_checkpoint_in_tf2_model)
end

subgraph src/transformers
12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle
end

subgraph src/transformers
1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> qn8gk(...)
end


      ClassDef mainFlowStyle color:#000000,fill:#7CB9F4
ClassDef rootsStyle color:#000000,fill:#00FFF4
ClassDef Style1 color:#000000,fill:#00FFAA
ClassDef Style2 color:#000000,fill:#FFFF00
ClassDef Style3 color:#000000,fill:#AA7CB9

%% Swimm:
%% graph TD;
%%       subgraph <SwmPath>[examples/legacy/run_swag.py](examples/legacy/run_swag.py)</SwmPath>
%% 7204f2d33d31255bfba6e222899c67ec461c4139c84e1060ef5e03f17c5045f9(load_and_cache_examples):::mainFlowStyle --> 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 9bd2a82735aaad77d3e97d05b37b13f0c1c858790f4e210614ce5a86ff0dfda6(convert_examples_to_features):::mainFlowStyle --> d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[hubconf.py](hubconf.py)</SwmPath>
%% d8c8177a87a75ba7c0bf928337cf716c05ff54343c2f415deec971f51ec0dc3a(tokenize):::mainFlowStyle --> a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% a7d86a9f9d1a7388834208576e5bfb18e7efdfdb507a5ffc5ed55288d2749593(tokenizer):::mainFlowStyle --> 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 2221acf97ae37a25b7480b354c46cde1d906c4bf6b7a078105fe2cc413f863dd(load_pytorch_checkpoint_in_tf2_model)
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 12ffe1d7574436f291223198deced7685ac3cf37f14b908fd33ce7dd53cc3ab8(from_pretrained):::mainFlowStyle --> 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> qn8gk(...)
%% end
%% 
%% 
%%       <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> mainFlowStyle color:#000000,fill:#7CB9F4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> rootsStyle color:#000000,fill:#00FFF4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style1 color:#000000,fill:#00FFAA
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style2 color:#000000,fill:#FFFF00
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/examples/research_projects/bert-loses-patience/run_glue_with_pabee.py" line="353">

---

## Loading and Caching Examples

The <SwmToken path="examples/research_projects/bert-loses-patience/run_glue_with_pabee.py" pos="353:2:2" line-data="def load_and_cache_examples(args, task, tokenizer, evaluate=False):">`load_and_cache_examples`</SwmToken> function is responsible for loading dataset examples and caching them for future use. It ensures that only the first process in distributed training processes the dataset, while others use the cached data. It loads data features from a cache or dataset file, processes the examples using the <SwmToken path="examples/legacy/run_swag.py" pos="126:2:2" line-data="def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):">`convert_examples_to_features`</SwmToken> function, and saves the processed features into a cached file. Finally, it converts the features into tensors and builds a dataset.

```python
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
```

---

</SwmSnippet>

<SwmSnippet path="/examples/legacy/run_swag.py" line="126">

---

## Converting Examples to Features

The <SwmToken path="examples/legacy/run_swag.py" pos="126:2:2" line-data="def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):">`convert_examples_to_features`</SwmToken> function converts raw dataset examples into features that can be fed into a model. For each example, it tokenizes the context and choices, truncates them to fit the maximum sequence length, and creates input <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="80:23:23" line-data="    tf_name = tf_name.replace(&quot;:0&quot;, &quot;&quot;)  # device ids">`ids`</SwmToken>, attention masks, and segment <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="80:23:23" line-data="    tf_name = tf_name.replace(&quot;:0&quot;, &quot;&quot;)  # device ids">`ids`</SwmToken>. These features are then used to create <SwmToken path="examples/legacy/run_swag.py" pos="90:2:2" line-data="class InputFeatures(object):">`InputFeatures`</SwmToken> objects, which are returned for model training or evaluation.

```python
def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in tqdm(enumerate(examples)):
```

---

</SwmSnippet>

<SwmSnippet path="/examples/research_projects/codeparrot/scripts/pretokenizing.py" line="10">

---

## Tokenizing

The <SwmToken path="examples/research_projects/codeparrot/scripts/pretokenizing.py" pos="10:2:2" line-data="def tokenize(example):">`tokenize`</SwmToken> function tokenizes the content of an example using a tokenizer. It generates input <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="80:23:23" line-data="    tf_name = tf_name.replace(&quot;:0&quot;, &quot;&quot;)  # device ids">`ids`</SwmToken> and calculates the ratio of characters to tokens. This function is essential for preparing the text data for model input.

```python
def tokenize(example):
    output = {}
    output["input_ids"] = tokenizer(example["content"], truncation=False)["input_ids"]
    output["ratio_char_token"] = len(example["content"]) / len(output["input_ids"])
    return output
```

---

</SwmSnippet>

<SwmSnippet path="/hubconf.py" line="59">

---

## Loading Pretrained Tokenizer

The <SwmToken path="hubconf.py" pos="59:2:2" line-data="def tokenizer(*args, **kwargs):">`tokenizer`</SwmToken> function loads a pretrained tokenizer from Hugging Face's model hub using <SwmToken path="hubconf.py" pos="61:5:7" line-data="        # Using torch.hub !">`torch.hub`</SwmToken>. It can download the tokenizer vocabulary from the hub or load it from a local directory where it was previously saved.

```python
def tokenizer(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'google-bert/bert-base-uncased')    # Download vocabulary from huggingface.co and cache.
        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

    """

    return AutoTokenizer.from_pretrained(*args, **kwargs)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/modeling_tf_utils.py" line="2542">

---

## Loading Pretrained Model

The <SwmToken path="src/transformers/modeling_tf_utils.py" pos="2542:3:3" line-data="    def from_pretrained(">`from_pretrained`</SwmToken> function loads a pretrained <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> model from a specified path or model hub. It handles various configurations and options, such as loading from a <SwmToken path="hubconf.py" pos="82:21:21" line-data="            # Loading from a TF checkpoint file instead of a PyTorch model (slower)">`PyTorch`</SwmToken> state dictionary, using cached files, and updating the model configuration during loading. This function is crucial for initializing models with pretrained weights for <SwmToken path="src/transformers/modeling_tf_utils.py" pos="2560:40:42" line-data="        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning">`fine-tuning`</SwmToken> or inference.

```python
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/modeling_tf_pytorch_utils.py" line="164">

---

## Loading <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="165:2:2" line-data="# PyTorch =&gt; TF 2.0 #">`PyTorch`</SwmToken> Checkpoint in <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> Model

The <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="169:2:2" line-data="def load_pytorch_checkpoint_in_tf2_model(">`load_pytorch_checkpoint_in_tf2_model`</SwmToken> function loads <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="165:2:2" line-data="# PyTorch =&gt; TF 2.0 #">`PyTorch`</SwmToken> checkpoints into a <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="165:8:10" line-data="# PyTorch =&gt; TF 2.0 #">`2.0`</SwmToken> model. It imports the necessary libraries, loads the <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="165:2:2" line-data="# PyTorch =&gt; TF 2.0 #">`PyTorch`</SwmToken> state dictionary, and updates the <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> model with the loaded weights. This function is essential for transferring pretrained weights from <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="165:2:2" line-data="# PyTorch =&gt; TF 2.0 #">`PyTorch`</SwmToken> models to <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> models.

```python
#####################
# PyTorch => TF 2.0 #
#####################


def load_pytorch_checkpoint_in_tf2_model(
    tf_model,
    pytorch_checkpoint_path,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
):
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
        from safetensors.torch import load_file as safe_load_file  # noqa: F401

        from .pytorch_utils import is_torch_greater_or_equal_than_1_13  # noqa: F401
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph src/transformers
1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle
end

subgraph src/transformers
db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle --> 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle
end

subgraph src/transformers
4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle --> 4eb0740a562b1d1a86d862174ad24d68b3321eff314e61987d10f38bf8c857b1(evaluate_ast):::mainFlowStyle
end


      ClassDef mainFlowStyle color:#000000,fill:#7CB9F4
ClassDef rootsStyle color:#000000,fill:#00FFF4
ClassDef Style1 color:#000000,fill:#00FFAA
ClassDef Style2 color:#000000,fill:#FFFF00
ClassDef Style3 color:#000000,fill:#AA7CB9

%% Swimm:
%% graph TD;
%%       subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 1b797d4990995aa418e67ad84f670cb20b7499cd7f90976afa0b8d845f0783c5(load_tf_sharded_weights_from_safetensors):::mainFlowStyle --> db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% db6525cb1bfd84cb906284b6c4686cd5780636d7c486a86424a0aae512d58378(load_tf_weights_from_safetensors):::mainFlowStyle --> 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[src/transformers/](src/transformers/)</SwmPath>
%% 4b33c19785eb13056b91730624d2bf295125a7d38b3a70d3f1c0142127610274(set_value):::mainFlowStyle --> 4eb0740a562b1d1a86d862174ad24d68b3321eff314e61987d10f38bf8c857b1(evaluate_ast):::mainFlowStyle
%% end
%% 
%% 
%%       <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> mainFlowStyle color:#000000,fill:#7CB9F4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> rootsStyle color:#000000,fill:#00FFF4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style1 color:#000000,fill:#00FFAA
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style2 color:#000000,fill:#FFFF00
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/src/transformers/modeling_tf_utils.py" line="875">

---

## Loading <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> Sharded Weights

The function <SwmToken path="src/transformers/modeling_tf_utils.py" pos="875:2:2" line-data="def load_tf_sharded_weights_from_safetensors(">`load_tf_sharded_weights_from_safetensors`</SwmToken> is responsible for loading <SwmToken path="src/transformers/modeling_tf_pytorch_utils.py" pos="180:3:3" line-data="        import tensorflow as tf  # noqa: F401">`tensorflow`</SwmToken> weights from sharded safetensors checkpoints. It iterates over each shard file, loading the weights into the model and handling any missing, unexpected, or mismatched layers. This ensures that the model is correctly populated with the weights from the checkpoint shards.

```python
def load_tf_sharded_weights_from_safetensors(
    model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None
):
    """
    This is the same as `load_tf_weights_from_safetensors` but for a sharded TF-format safetensors checkpoint.
    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and
    shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/modeling_tf_utils.py" line="1057">

---

### Loading Weights from a Single Shard

The function <SwmToken path="src/transformers/modeling_tf_utils.py" pos="1057:2:2" line-data="def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):">`load_tf_weights_from_safetensors`</SwmToken> loads weights from a single safetensors file into the model. It reads the safetensors file, identifies missing and unexpected layers, and assigns the loaded weights to the corresponding model weights. This function is called by <SwmToken path="src/transformers/modeling_tf_utils.py" pos="875:2:2" line-data="def load_tf_sharded_weights_from_safetensors(">`load_tf_sharded_weights_from_safetensors`</SwmToken> for each shard file.

```python
def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # Read the safetensors file
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        mismatched_layers = []
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        # Find the missing layers from the high level list of layers
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))

        for weight in model.weights:
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                # Check if the shape of the current weight and the one from the H5 file are different
                if K.int_shape(weight) != weight_value.shape:
                    # If yes we reshape the weight from the H5 file accordingly to the current weight
                    # If the two shapes are not compatible we raise an issue
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/agents/python_interpreter.py" line="340">

---

## Setting Values in the Model

The function <SwmToken path="src/transformers/agents/python_interpreter.py" pos="340:2:2" line-data="def set_value(target, value, state, static_tools, custom_tools):">`set_value`</SwmToken> assigns a value to a target within the model's state. It handles different types of targets, such as names, tuples, subscripts, and attributes, ensuring that the value is correctly set in the model's state. This function is crucial for updating the model's parameters during the weight loading process.

```python
def set_value(target, value, state, static_tools, custom_tools):
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/agents/python_interpreter.py" line="703">

---

## Evaluating Abstract Syntax Trees

The function <SwmToken path="src/transformers/agents/python_interpreter.py" pos="703:2:2" line-data="def evaluate_ast(">`evaluate_ast`</SwmToken> evaluates an abstract syntax tree (AST) using the current state of the model. It supports various AST node types, such as assignments, function calls, constants, and more. This function is used to interpret and execute the model's configuration and parameter updates during the weight loading process.

```python
def evaluate_ast(
    expression: ast.AST,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str] = LIST_SAFE_MODULES,
):
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        static_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
```

---

</SwmSnippet>

# Where is this flow used?

This flow is used multiple times in the codebase as represented in the following diagram:

(Note - these are only some of the entry points of this flow)

```mermaid
graph TD;
      subgraph examples/research_projects/information-gain-filtration
227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 7b6c630bd7e5cf19d6c334bc1ccaee4408b471a24b7ce7ee3875816402993870(generate_n_pairs)
end

subgraph examples/research_projects/information-gain-filtration
7b6c630bd7e5cf19d6c334bc1ccaee4408b471a24b7ce7ee3875816402993870(generate_n_pairs) --> 73f618e07383fe4bcbdeafd6bd1c2e297dd76f83e7d3fd765489a99ce15130a2(collect_objective_set)
end

subgraph examples/research_projects/information-gain-filtration
73f618e07383fe4bcbdeafd6bd1c2e297dd76f83e7d3fd765489a99ce15130a2(collect_objective_set) --> 7866028ed6572550fdf3ff5db72b0f4654d1f9cc26b15c92b9a4f5c0e4554cdb(compute_perplexity)
end

subgraph examples/research_projects
7866028ed6572550fdf3ff5db72b0f4654d1f9cc26b15c92b9a4f5c0e4554cdb(compute_perplexity) --> 997b8f5212cd2547cb091260b4408763b348ac29986c918c386c7c6a31ddf2d0(train)
end

subgraph examples/research_projects
997b8f5212cd2547cb091260b4408763b348ac29986c918c386c7c6a31ddf2d0(train) --> 6e607b060f7c1df9e5d57b2ad94926566134955ed97b0934bff72d2b25b99233(step)
end

subgraph examples/research_projects
6e607b060f7c1df9e5d57b2ad94926566134955ed97b0934bff72d2b25b99233(step) --> 15596689f074c099319f83dad04385d5b062acd336562aa582ce6f27f587ab09(optimize)
end

subgraph examples/research_projects
15596689f074c099319f83dad04385d5b062acd336562aa582ce6f27f587ab09(optimize) --> ef01ee1bf20092c3bdf6ab5794e44146f134e3cf8e7a0b994f80b9f026184c88(iter)
end

subgraph examples/research_projects
ef01ee1bf20092c3bdf6ab5794e44146f134e3cf8e7a0b994f80b9f026184c88(iter) --> f9554bc016dae8976077f045de21b831a8c869cfde4e649a91058cf8fa1ec23a(save_checkpoint)
end

subgraph examples/research_projects
f9554bc016dae8976077f045de21b831a8c869cfde4e649a91058cf8fa1ec23a(save_checkpoint) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
end

subgraph examples/research_projects
227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 3a41dfb77316db0cb5871fe12777f8ee020710911a71d87f0e6c155ef5634411(train)
end

subgraph examples/research_projects
3a41dfb77316db0cb5871fe12777f8ee020710911a71d87f0e6c155ef5634411(train) --> c71de3ba0fc3c5e16292c8f656c085c4b342da0a3e3a57f83ae03e54ad766afe(evaluate)
end

subgraph examples/research_projects
c71de3ba0fc3c5e16292c8f656c085c4b342da0a3e3a57f83ae03e54ad766afe(evaluate) --> 6960fe772a31ad459f3e4abfc0fdf90c253b14e047d21d8b3294532ee4ea9ae8(load_and_cache_examples)
end

subgraph examples/research_projects
6960fe772a31ad459f3e4abfc0fdf90c253b14e047d21d8b3294532ee4ea9ae8(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
end

227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train)

a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train) --> 3df33a9ed5a0de1c7e6394da6dfe181bb6397eba15c9fcaa9f2e147eba43ddcd(evaluate)

3df33a9ed5a0de1c7e6394da6dfe181bb6397eba15c9fcaa9f2e147eba43ddcd(evaluate) --> 49ac3dcd0154a2851e03b4b66c4cd995adeece0d05555195aa7548b3b525336e(load_and_cache_examples)

subgraph examples/research_projects
49ac3dcd0154a2851e03b4b66c4cd995adeece0d05555195aa7548b3b525336e(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
end

227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train)

subgraph examples/research_projects
227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 49547cd58d7108ed83769f1696c8b699b9c8e8af61958437f08df3177791584e(train)
end

subgraph examples/research_projects
49547cd58d7108ed83769f1696c8b699b9c8e8af61958437f08df3177791584e(train) --> 416a4c9c2dd39248360609ec81cfee930e7a39f15db12723cbe39ee6007fc4b1(evaluate)
end

subgraph examples/research_projects
416a4c9c2dd39248360609ec81cfee930e7a39f15db12723cbe39ee6007fc4b1(evaluate) --> fd8c633f8bf2caf14fa9f16ce8f9477b12ea87936a5214a6c933068ae611496f(load_and_cache_examples)
end

subgraph examples/research_projects
fd8c633f8bf2caf14fa9f16ce8f9477b12ea87936a5214a6c933068ae611496f(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
end


      ClassDef mainFlowStyle color:#000000,fill:#7CB9F4
ClassDef rootsStyle color:#000000,fill:#00FFF4
ClassDef Style1 color:#000000,fill:#00FFAA
ClassDef Style2 color:#000000,fill:#FFFF00
ClassDef Style3 color:#000000,fill:#AA7CB9

%% Swimm:
%% graph TD;
%%       subgraph <SwmPath>[examples/research_projects/information-gain-filtration/](examples/research_projects/information-gain-filtration/)</SwmPath>
%% 227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 7b6c630bd7e5cf19d6c334bc1ccaee4408b471a24b7ce7ee3875816402993870(generate_n_pairs)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/information-gain-filtration/](examples/research_projects/information-gain-filtration/)</SwmPath>
%% 7b6c630bd7e5cf19d6c334bc1ccaee4408b471a24b7ce7ee3875816402993870(generate_n_pairs) --> 73f618e07383fe4bcbdeafd6bd1c2e297dd76f83e7d3fd765489a99ce15130a2(collect_objective_set)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/information-gain-filtration/](examples/research_projects/information-gain-filtration/)</SwmPath>
%% 73f618e07383fe4bcbdeafd6bd1c2e297dd76f83e7d3fd765489a99ce15130a2(collect_objective_set) --> 7866028ed6572550fdf3ff5db72b0f4654d1f9cc26b15c92b9a4f5c0e4554cdb(compute_perplexity)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 7866028ed6572550fdf3ff5db72b0f4654d1f9cc26b15c92b9a4f5c0e4554cdb(compute_perplexity) --> 997b8f5212cd2547cb091260b4408763b348ac29986c918c386c7c6a31ddf2d0(train)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 997b8f5212cd2547cb091260b4408763b348ac29986c918c386c7c6a31ddf2d0(train) --> 6e607b060f7c1df9e5d57b2ad94926566134955ed97b0934bff72d2b25b99233(step)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 6e607b060f7c1df9e5d57b2ad94926566134955ed97b0934bff72d2b25b99233(step) --> 15596689f074c099319f83dad04385d5b062acd336562aa582ce6f27f587ab09(optimize)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 15596689f074c099319f83dad04385d5b062acd336562aa582ce6f27f587ab09(optimize) --> ef01ee1bf20092c3bdf6ab5794e44146f134e3cf8e7a0b994f80b9f026184c88(iter)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% ef01ee1bf20092c3bdf6ab5794e44146f134e3cf8e7a0b994f80b9f026184c88(iter) --> f9554bc016dae8976077f045de21b831a8c869cfde4e649a91058cf8fa1ec23a(save_checkpoint)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% f9554bc016dae8976077f045de21b831a8c869cfde4e649a91058cf8fa1ec23a(save_checkpoint) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 3a41dfb77316db0cb5871fe12777f8ee020710911a71d87f0e6c155ef5634411(train)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 3a41dfb77316db0cb5871fe12777f8ee020710911a71d87f0e6c155ef5634411(train) --> c71de3ba0fc3c5e16292c8f656c085c4b342da0a3e3a57f83ae03e54ad766afe(evaluate)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% c71de3ba0fc3c5e16292c8f656c085c4b342da0a3e3a57f83ae03e54ad766afe(evaluate) --> 6960fe772a31ad459f3e4abfc0fdf90c253b14e047d21d8b3294532ee4ea9ae8(load_and_cache_examples)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 6960fe772a31ad459f3e4abfc0fdf90c253b14e047d21d8b3294532ee4ea9ae8(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
%% end
%% 
%% 227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train)
%% 
%% a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train) --> 3df33a9ed5a0de1c7e6394da6dfe181bb6397eba15c9fcaa9f2e147eba43ddcd(evaluate)
%% 
%% 3df33a9ed5a0de1c7e6394da6dfe181bb6397eba15c9fcaa9f2e147eba43ddcd(evaluate) --> 49ac3dcd0154a2851e03b4b66c4cd995adeece0d05555195aa7548b3b525336e(load_and_cache_examples)
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 49ac3dcd0154a2851e03b4b66c4cd995adeece0d05555195aa7548b3b525336e(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
%% end
%% 
%% 227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> a3eed5832bf5ad0c25c9ea747a5451534b4bf497205c463bc7076b2598c402d9(train)
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> 49547cd58d7108ed83769f1696c8b699b9c8e8af61958437f08df3177791584e(train)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 49547cd58d7108ed83769f1696c8b699b9c8e8af61958437f08df3177791584e(train) --> 416a4c9c2dd39248360609ec81cfee930e7a39f15db12723cbe39ee6007fc4b1(evaluate)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% 416a4c9c2dd39248360609ec81cfee930e7a39f15db12723cbe39ee6007fc4b1(evaluate) --> fd8c633f8bf2caf14fa9f16ce8f9477b12ea87936a5214a6c933068ae611496f(load_and_cache_examples)
%% end
%% 
%% subgraph <SwmPath>[examples/research_projects/](examples/research_projects/)</SwmPath>
%% fd8c633f8bf2caf14fa9f16ce8f9477b12ea87936a5214a6c933068ae611496f(load_and_cache_examples) --> a8b9cae6c7e5f9b45965d9c1867cc814f63d54ff23357ee9316532ae21a7382f(save):::mainFlowStyle
%% end
%% 
%% 
%%       <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> mainFlowStyle color:#000000,fill:#7CB9F4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> rootsStyle color:#000000,fill:#00FFF4
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style1 color:#000000,fill:#00FFAA
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style2 color:#000000,fill:#FFFF00
%% <SwmToken path="src/transformers/agents/python_interpreter.py" pos="828:10:10" line-data="    elif isinstance(expression, ast.ClassDef):">`ClassDef`</SwmToken> Style3 color:#000000,fill:#AA7CB9
```

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="flows"><sup>Powered by [Swimm](/)</sup></SwmMeta>
