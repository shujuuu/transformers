---
title: Basic concepts of Parallelization in Utilities
---
Parallelization refers to the distribution of computational tasks across multiple devices to improve efficiency and performance.

In utilities, parallelization is achieved by dividing the model's layers into blocks and distributing these blocks evenly across available devices.

The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> is used to create a dictionary that maps each device to a list of layers it will handle.

The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> ensures that the device map is correctly configured by checking for duplicate, missing, or extra blocks.

<SwmSnippet path="/src/transformers/utils/model_parallel_utils.py" line="48">

---

# <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> Function

The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="48:2:2" line-data="def get_device_map(n_layers, devices):">`get_device_map`</SwmToken> is used to create a dictionary that maps each device to a list of layers it will handle. This function ensures that the layers are evenly distributed across all available devices.

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

# <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> Function

The function <SwmToken path="src/transformers/utils/model_parallel_utils.py" pos="19:2:2" line-data="def assert_device_map(device_map, num_blocks):">`assert_device_map`</SwmToken> ensures that the device map is correctly configured by checking for duplicate, missing, or extra blocks. This validation step is crucial to ensure that the parallelization is set up correctly and efficiently.

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
