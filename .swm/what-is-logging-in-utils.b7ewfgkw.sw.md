---
title: What is Logging in Utils
---
Logging refers to the centralized logging system that allows easy setup of the library's verbosity levels.

The default verbosity level is set to <SwmToken path="src/transformers/utils/logging.py" pos="66:7:7" line-data="            logging.getLogger().warning(">`warning`</SwmToken>, but it can be changed using direct setters like <SwmToken path="src/transformers/utils/logging.py" pos="203:2:4" line-data="def set_verbosity_info():">`set_verbosity_info()`</SwmToken>.

The logging system supports various levels such as <SwmToken path="src/transformers/utils/logging.py" pos="176:11:11" line-data="    - 10: `transformers.logging.DEBUG`">`DEBUG`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="175:11:11" line-data="    - 20: `transformers.logging.INFO`">`INFO`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="66:7:7" line-data="            logging.getLogger().warning(">`warning`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="173:11:11" line-data="    - 40: `transformers.logging.ERROR`">`ERROR`</SwmToken>, and <SwmToken path="src/transformers/utils/logging.py" pos="172:11:11" line-data="    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`">`CRITICAL`</SwmToken>.

Environment variables like <SwmToken path="src/transformers/utils/logging.py" pos="58:3:3" line-data="    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is">`TRANSFORMERS_VERBOSITY`</SwmToken> can also be used to override the default verbosity level.

The logging system includes functions to enable or disable the default handler, add or remove custom handlers, and manage log propagation.

Explicit formatting can be enabled for log messages, which includes details like the log level, filename, line number, and timestamp.

## Centralized Logging System

Transformers has a centralized logging system, so that you can setup the verbosity of the library easily.

## Default Verbosity Level

Currently the default verbosity of the library is <SwmToken path="src/transformers/utils/logging.py" pos="66:7:7" line-data="            logging.getLogger().warning(">`warning`</SwmToken>.

## Changing Verbosity Level

To change the level of verbosity, just use one of the direct setters. For instance, here is how to change the verbosity to the INFO level.

<SwmSnippet path="/src/transformers/utils/logging.py" line="203">

---

The function <SwmToken path="src/transformers/utils/logging.py" pos="203:2:2" line-data="def set_verbosity_info():">`set_verbosity_info`</SwmToken> sets the verbosity to the <SwmToken path="src/transformers/utils/logging.py" pos="204:15:15" line-data="    &quot;&quot;&quot;Set the verbosity to the `INFO` level.&quot;&quot;&quot;">`INFO`</SwmToken> level.

```python
def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)
```

---

</SwmSnippet>

## Using Environment Variables

You can also use the environment variable <SwmToken path="src/transformers/utils/logging.py" pos="58:3:3" line-data="    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is">`TRANSFORMERS_VERBOSITY`</SwmToken> to override the default verbosity level. You can set it to one of the following levels: <SwmToken path="src/transformers/utils/logging.py" pos="176:11:11" line-data="    - 10: `transformers.logging.DEBUG`">`DEBUG`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="175:11:11" line-data="    - 20: `transformers.logging.INFO`">`INFO`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="66:7:7" line-data="            logging.getLogger().warning(">`warning`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="173:11:11" line-data="    - 40: `transformers.logging.ERROR`">`ERROR`</SwmToken>, <SwmToken path="src/transformers/utils/logging.py" pos="172:11:11" line-data="    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`">`CRITICAL`</SwmToken>.

<SwmSnippet path="/src/transformers/utils/logging.py" line="56">

---

The function <SwmToken path="src/transformers/utils/logging.py" pos="56:2:2" line-data="def _get_default_logging_level():">`_get_default_logging_level`</SwmToken> checks the <SwmToken path="src/transformers/utils/logging.py" pos="58:3:3" line-data="    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is">`TRANSFORMERS_VERBOSITY`</SwmToken> environment variable and sets the default logging level accordingly.

```python
def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level
```

---

</SwmSnippet>

## Managing Handlers and Propagation

The logging system includes functions to enable or disable the default handler, add or remove custom handlers, and manage log propagation.

<SwmSnippet path="/src/transformers/utils/logging.py" line="223">

---

The function <SwmToken path="src/transformers/utils/logging.py" pos="223:2:2" line-data="def disable_default_handler() -&gt; None:">`disable_default_handler`</SwmToken> disables the default handler of the <SwmToken path="src/transformers/utils/logging.py" pos="224:16:16" line-data="    &quot;&quot;&quot;Disable the default handler of the HuggingFace Transformers&#39;s root logger.&quot;&quot;&quot;">`HuggingFace`</SwmToken> Transformers's root logger.

```python
def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/logging.py" line="232">

---

The function <SwmToken path="src/transformers/utils/logging.py" pos="232:2:2" line-data="def enable_default_handler() -&gt; None:">`enable_default_handler`</SwmToken> enables the default handler of the <SwmToken path="src/transformers/utils/logging.py" pos="233:16:16" line-data="    &quot;&quot;&quot;Enable the default handler of the HuggingFace Transformers&#39;s root logger.&quot;&quot;&quot;">`HuggingFace`</SwmToken> Transformers's root logger.

```python
def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)

```

---

</SwmSnippet>

## Enabling Explicit Formatting

Explicit formatting can be enabled for log messages, which includes details like the log level, filename, line number, and timestamp.

<SwmSnippet path="/src/transformers/utils/logging.py" line="278">

---

The function <SwmToken path="src/transformers/utils/logging.py" pos="278:2:2" line-data="def enable_explicit_format() -&gt; None:">`enable_explicit_format`</SwmToken> enables explicit formatting for every <SwmToken path="src/transformers/utils/logging.py" pos="280:11:11" line-data="    Enable explicit formatting for every HuggingFace Transformers&#39;s logger. The explicit formatter is as follows:">`HuggingFace`</SwmToken> Transformers's logger.

````python
def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)
````

---

</SwmSnippet>

# Main functions

Main functions

<SwmSnippet path="/src/transformers/utils/logging.py" line="161">

---

## <SwmToken path="src/transformers/utils/logging.py" pos="161:2:2" line-data="def get_verbosity() -&gt; int:">`get_verbosity`</SwmToken>

The <SwmToken path="src/transformers/utils/logging.py" pos="161:2:2" line-data="def get_verbosity() -&gt; int:">`get_verbosity`</SwmToken> function returns the current logging level for the library's root logger as an integer.

```python
def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Transformers has following logging levels:

    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING` or `transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </Tip>"""

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/logging.py" line="184">

---

## <SwmToken path="src/transformers/utils/logging.py" pos="184:2:2" line-data="def set_verbosity(verbosity: int) -&gt; None:">`set_verbosity`</SwmToken>

The <SwmToken path="src/transformers/utils/logging.py" pos="184:2:2" line-data="def set_verbosity(verbosity: int) -&gt; None:">`set_verbosity`</SwmToken> function sets the verbosity level for the library's root logger.

```python
def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING` or `transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/logging.py" line="147">

---

## <SwmToken path="src/transformers/utils/logging.py" pos="147:2:2" line-data="def get_logger(name: Optional[str] = None) -&gt; logging.Logger:">`get_logger`</SwmToken>

The <SwmToken path="src/transformers/utils/logging.py" pos="147:2:2" line-data="def get_logger(name: Optional[str] = None) -&gt; logging.Logger:">`get_logger`</SwmToken> function returns a logger with the specified name. If no name is provided, it returns the library's root logger.

```python
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/logging.py" line="232">

---

## <SwmToken path="src/transformers/utils/logging.py" pos="232:2:2" line-data="def enable_default_handler() -&gt; None:">`enable_default_handler`</SwmToken>

The <SwmToken path="src/transformers/utils/logging.py" pos="232:2:2" line-data="def enable_default_handler() -&gt; None:">`enable_default_handler`</SwmToken> function enables the default handler for the library's root logger.

```python
def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/utils/logging.py" line="223">

---

## <SwmToken path="src/transformers/utils/logging.py" pos="223:2:2" line-data="def disable_default_handler() -&gt; None:">`disable_default_handler`</SwmToken>

The <SwmToken path="src/transformers/utils/logging.py" pos="223:2:2" line-data="def disable_default_handler() -&gt; None:">`disable_default_handler`</SwmToken> function disables the default handler for the library's root logger.

```python
def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
