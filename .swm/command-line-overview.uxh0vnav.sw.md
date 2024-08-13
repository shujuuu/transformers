---
title: Command Line Overview
---
The command line interface (CLI) provides a way to interact with the repository's functionalities through text-based commands.

The <SwmToken path="src/transformers/commands/train.py" pos="33:2:2" line-data="def train_command_factory(args: Namespace):">`train_command_factory`</SwmToken> function is used to instantiate a training command from provided command line arguments.

The <SwmToken path="src/transformers/commands/user.py" pos="266:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> class handles the creation of new repositories, including checking for necessary tools like <SwmToken path="src/transformers/commands/user.py" pos="276:11:11" line-data="            stdout = subprocess.check_output([&quot;git&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git`</SwmToken> and <SwmToken path="src/transformers/commands/user.py" pos="282:11:13" line-data="            stdout = subprocess.check_output([&quot;git-lfs&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git-lfs`</SwmToken>, and prompting the user for confirmation.

The <SwmToken path="src/transformers/commands/user.py" pos="149:2:2" line-data="class LoginCommand(BaseUserCommand):">`LoginCommand`</SwmToken> class manages user authentication by prompting for credentials and saving the authentication token.

These commands are defined in the <SwmPath>[src/transformers/commands/](src/transformers/commands/)</SwmPath> directory and are used to facilitate various tasks such as training models, managing repositories, and user authentication.

<SwmSnippet path="/src/transformers/commands/train.py" line="33">

---

## Training Models

The <SwmToken path="src/transformers/commands/train.py" pos="33:2:2" line-data="def train_command_factory(args: Namespace):">`train_command_factory`</SwmToken> function is used to instantiate a training command from provided command line arguments.

```python
def train_command_factory(args: Namespace):
    """
    Factory function used to instantiate training command from provided command line arguments.

    Returns: TrainCommand
    """
    return TrainCommand(args)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="266">

---

## Creating Repositories

The <SwmToken path="src/transformers/commands/user.py" pos="266:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> class handles the creation of new repositories, including checking for necessary tools like <SwmToken path="src/transformers/commands/user.py" pos="276:11:11" line-data="            stdout = subprocess.check_output([&quot;git&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git`</SwmToken> and <SwmToken path="src/transformers/commands/user.py" pos="282:11:13" line-data="            stdout = subprocess.check_output([&quot;git-lfs&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git-lfs`</SwmToken>, and prompting the user for confirmation.

```python
class RepoCreateCommand(BaseUserCommand):
    def run(self):
        warnings.warn(
            "Managing repositories through transformers-cli is deprecated. Please use `huggingface-cli` instead."
        )
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="149">

---

## User Authentication

The <SwmToken path="src/transformers/commands/user.py" pos="149:2:2" line-data="class LoginCommand(BaseUserCommand):">`LoginCommand`</SwmToken> class manages user authentication by prompting for credentials and saving the authentication token.

```python
class LoginCommand(BaseUserCommand):
    def run(self):
        print(  # docstyle-ignore
            """
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

        """
        )
        username = input("Username: ")
        password = getpass()
        try:
            token = self._api.login(username, password)
        except HTTPError as e:
            # probably invalid credentials, display error message.
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
