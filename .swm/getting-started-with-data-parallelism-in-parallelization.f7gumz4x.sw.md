---
title: Getting started with Data Parallelism in Parallelization
---
Data Parallelism is a technique used to distribute data across multiple devices to perform parallel computations.

It involves splitting the data into smaller chunks and processing these chunks simultaneously on different devices.

This approach helps in speeding up the training process by leveraging the computational power of multiple devices.

In the code, the function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> is used to distribute layers of a model evenly across all available devices.

The function calculates the number of blocks each device should handle and returns a dictionary mapping devices to their respective layers.

<SwmSnippet path="/src/transformers/utils/model_parallel_utils.py" line="48">

---

Distributing Layers Across Devices The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> is used to distribute layers of a model evenly across all available devices. It calculates the number of blocks each device should handle and returns a dictionary mapping devices to their respective layers.

```python
def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
