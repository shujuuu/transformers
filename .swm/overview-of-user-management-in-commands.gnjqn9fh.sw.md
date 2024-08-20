---
title: Overview of User Management in Commands
---
User Management in Commands involves handling user authentication and repository management tasks.

The <SwmToken path="src/transformers/commands/user.py" pos="25:2:2" line-data="class UserCommands(BaseTransformersCLICommand):">`UserCommands`</SwmToken> class registers subcommands for user-related actions such as login, logout, and checking the current user.

The <SwmToken path="src/transformers/commands/user.py" pos="29:12:12" line-data="        login_parser.set_defaults(func=lambda args: LoginCommand(args))">`LoginCommand`</SwmToken> class is responsible for logging in users, although it currently uses an outdated mechanism.

The <SwmToken path="src/transformers/commands/user.py" pos="31:12:12" line-data="        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))">`WhoamiCommand`</SwmToken> class allows users to check which Hugging Face account they are logged in as, but it is deprecated.

The <SwmToken path="src/transformers/commands/user.py" pos="33:12:12" line-data="        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))">`LogoutCommand`</SwmToken> class handles logging out users, but it also uses an outdated mechanism.

The <SwmToken path="src/transformers/commands/user.py" pos="146:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> class manages the creation of new repositories on Hugging Face, but this functionality is deprecated in favor of <SwmToken path="src/transformers/commands/user.py" pos="105:6:8" line-data="                &quot;ERROR! `huggingface-cli login` uses an outdated login mechanism &quot;">`huggingface-cli`</SwmToken>.

<SwmSnippet path="/src/transformers/commands/user.py" line="25">

---

# <SwmToken path="src/transformers/commands/user.py" pos="25:2:2" line-data="class UserCommands(BaseTransformersCLICommand):">`UserCommands`</SwmToken> Class

The <SwmToken path="src/transformers/commands/user.py" pos="25:2:2" line-data="class UserCommands(BaseTransformersCLICommand):">`UserCommands`</SwmToken> class registers subcommands for user-related actions such as login, logout, and checking the current user.

```python
class UserCommands(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login", help="Log in using the same credentials as on huggingface.co")
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="101">

---

# <SwmToken path="src/transformers/commands/user.py" pos="101:2:2" line-data="class LoginCommand(BaseUserCommand):">`LoginCommand`</SwmToken> Class

The <SwmToken path="src/transformers/commands/user.py" pos="101:2:2" line-data="class LoginCommand(BaseUserCommand):">`LoginCommand`</SwmToken> class is responsible for logging in users, although it currently uses an outdated mechanism.

```python
class LoginCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "ERROR! `huggingface-cli login` uses an outdated login mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli login instead."
            )
        )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="112">

---

# <SwmToken path="src/transformers/commands/user.py" pos="112:2:2" line-data="class WhoamiCommand(BaseUserCommand):">`WhoamiCommand`</SwmToken> Class

The <SwmToken path="src/transformers/commands/user.py" pos="112:2:2" line-data="class WhoamiCommand(BaseUserCommand):">`WhoamiCommand`</SwmToken> class allows users to check which Hugging Face account they are logged in as, but it is deprecated.

```python
class WhoamiCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! `transformers-cli whoami` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli whoami` instead."
            )
        )
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            user, orgs = whoami(token)
            print(user)
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="135">

---

# <SwmToken path="src/transformers/commands/user.py" pos="135:2:2" line-data="class LogoutCommand(BaseUserCommand):">`LogoutCommand`</SwmToken> Class

The <SwmToken path="src/transformers/commands/user.py" pos="135:2:2" line-data="class LogoutCommand(BaseUserCommand):">`LogoutCommand`</SwmToken> class handles logging out users, but it also uses an outdated mechanism.

```python
class LogoutCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "ERROR! `transformers-cli logout` uses an outdated logout mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli logout instead."
            )
        )
```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="146">

---

# <SwmToken path="src/transformers/commands/user.py" pos="146:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> Class

The <SwmToken path="src/transformers/commands/user.py" pos="146:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> class manages the creation of new repositories on Hugging Face, but this functionality is deprecated in favor of <SwmToken path="src/transformers/commands/user.py" pos="151:7:9" line-data="                &quot;Please use `huggingface-cli` instead.&quot;">`huggingface-cli`</SwmToken>.

```python
class RepoCreateCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! Managing repositories through transformers-cli is deprecated. "
                "Please use `huggingface-cli` instead."
            )
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
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
