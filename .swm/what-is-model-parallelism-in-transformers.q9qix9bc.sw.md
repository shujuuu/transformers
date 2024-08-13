---
title: What is Model Parallelism in Transformers
---
Model Parallelism refers to the technique of distributing different parts of a model across multiple devices to leverage their combined computational power.

It is particularly useful for training large models that do not fit into the memory of a single device.

In transformers, model parallelism is implemented by dividing the model's layers across multiple devices.

The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> is used to distribute the layers evenly across the available devices.

This function calculates the number of layers each device should handle and returns a dictionary mapping each device to its assigned layers.

The <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> function ensures that the device map is correctly configured, checking for duplicate, missing, or extra blocks.

<SwmSnippet path="/src/transformers/utils/model_parallel_utils.py" line="48">

---

# <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> function

The <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> function is responsible for distributing the model's layers evenly across the available devices. It calculates the number of layers each device should handle and returns a dictionary mapping each device to its assigned layers.

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

<SwmSnippet path="/src/transformers/utils/model_parallel_utils.py" line="19">

---

# <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> function

The <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> function ensures that the device map is correctly configured. It checks for duplicate, missing, or extra blocks to ensure that each layer is assigned to exactly one device.

```python
def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    assert len(duplicate_blocks) == 0, (
        "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
        "attention blocks were specified more than once: " + str(duplicate_blocks)
    )
    assert len(missing_blocks) == 0, (
        "There are attention blocks for this model that are not specified in the device_map. Add these attention "
        "blocks to a device on the device_map: " + str(missing_blocks)
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
