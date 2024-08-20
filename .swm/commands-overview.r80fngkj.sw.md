---
title: Commands Overview
---
Commands refer to the various operations that can be executed using the CLI tool provided by the transformers library.

These commands are registered and managed through the <SwmToken path="src/transformers/commands/transformers_cli.py" pos="29:2:2" line-data="def main():">`main`</SwmToken> function in <SwmPath>[src/transformers/commands/transformers_cli.py](src/transformers/commands/transformers_cli.py)</SwmPath>, which sets up the argument parser and subparsers for different commands.

Each command is associated with a specific functionality, such as converting models, downloading models, or managing user accounts.

For example, the <SwmToken path="src/transformers/commands/transformers_cli.py" pos="34:1:1" line-data="    ConvertCommand.register_subcommand(commands_parser)">`ConvertCommand`</SwmToken> is used to convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint.

The <SwmToken path="src/transformers/commands/transformers_cli.py" pos="36:1:1" line-data="    EnvironmentCommand.register_subcommand(commands_parser)">`EnvironmentCommand`</SwmToken> provides information about the current environment, including versions of installed libraries and hardware capabilities.

User-related commands like <SwmToken path="src/transformers/commands/user.py" pos="29:12:12" line-data="        login_parser.set_defaults(func=lambda args: LoginCommand(args))">`LoginCommand`</SwmToken> and <SwmToken path="src/transformers/commands/user.py" pos="31:12:12" line-data="        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))">`WhoamiCommand`</SwmToken> handle user authentication and account information.

Commands are registered using the <SwmToken path="src/transformers/commands/transformers_cli.py" pos="34:3:3" line-data="    ConvertCommand.register_subcommand(commands_parser)">`register_subcommand`</SwmToken> method, which adds them to the argument parser so they can be invoked from the command line.

When a command is executed, the corresponding <SwmToken path="src/transformers/commands/transformers_cli.py" pos="51:3:3" line-data="    # Run">`Run`</SwmToken> method is called to perform the specified operation.

<SwmSnippet path="/src/transformers/commands/transformers_cli.py" line="29">

---

## Registering Commands

The <SwmToken path="src/transformers/commands/transformers_cli.py" pos="29:2:2" line-data="def main():">`main`</SwmToken> function sets up the argument parser and registers various commands using the <SwmToken path="src/transformers/commands/transformers_cli.py" pos="34:3:3" line-data="    ConvertCommand.register_subcommand(commands_parser)">`register_subcommand`</SwmToken> method.

```python
def main():
    parser = ArgumentParser("Transformers CLI tool", usage="transformers-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="transformers-cli command helpers")

    # Register commands
    ConvertCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)
    AddNewModelLikeCommand.register_subcommand(commands_parser)
    LfsCommands.register_subcommand(commands_parser)
    PTtoTFCommand.register_subcommand(commands_parser)

```

---

</SwmSnippet>

<SwmSnippet path="/src/transformers/commands/user.py" line="146">

---

## Example of a Command

The <SwmToken path="src/transformers/commands/user.py" pos="146:2:2" line-data="class RepoCreateCommand(BaseUserCommand):">`RepoCreateCommand`</SwmToken> is an example of a command that manages repository creation. It includes checks for user authentication and necessary tools like <SwmToken path="src/transformers/commands/user.py" pos="159:11:11" line-data="            stdout = subprocess.check_output([&quot;git&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git`</SwmToken> and <SwmToken path="src/transformers/commands/user.py" pos="165:11:13" line-data="            stdout = subprocess.check_output([&quot;git-lfs&quot;, &quot;--version&quot;]).decode(&quot;utf-8&quot;)">`git-lfs`</SwmToken>.

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

<SwmSnippet path="/src/transformers/commands/transformers_cli.py" line="45">

---

## Running a Command

When a command is executed, the <SwmToken path="src/transformers/commands/transformers_cli.py" pos="51:3:3" line-data="    # Run">`Run`</SwmToken> method of the corresponding command class is called to perform the specified operation.

```python
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
