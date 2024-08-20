---
title: Using the Makefile
---
# Intro

This document explains how to use the <SwmPath>[Makefile](Makefile)</SwmPath> in the transformers repository. It will go through each target in the <SwmPath>[Makefile](Makefile)</SwmPath> step by step.

<SwmSnippet path="/Makefile" line="1">

---

# Phony Targets

The <SwmToken path="Makefile" pos="1:0:1" line-data=".PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples benchmark">`.PHONY`</SwmToken> target declares a list of phony targets to avoid conflicts with files of the same name.

```
.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples benchmark
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="4">

---

# Setting PYTHONPATH

The <SwmToken path="Makefile" pos="4:2:2" line-data="export PYTHONPATH = src">`PYTHONPATH`</SwmToken> is set to <SwmToken path="Makefile" pos="4:6:6" line-data="export PYTHONPATH = src">`src`</SwmToken> to ensure that the local checkout is tested in scripts instead of the <SwmToken path="Makefile" pos="3:26:28" line-data="# make sure to test the local checkout in scripts and not the pre-installed one (don&#39;t use quotes!)">`pre-installed`</SwmToken> one.

```
export PYTHONPATH = src
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="6">

---

# Directories to Check

The <SwmToken path="Makefile" pos="6:0:0" line-data="check_dirs := examples tests src utils">`check_dirs`</SwmToken> variable lists the directories to be checked: <SwmToken path="Makefile" pos="6:4:4" line-data="check_dirs := examples tests src utils">`examples`</SwmToken>, <SwmToken path="Makefile" pos="6:6:6" line-data="check_dirs := examples tests src utils">`tests`</SwmToken>, <SwmToken path="Makefile" pos="6:8:8" line-data="check_dirs := examples tests src utils">`src`</SwmToken>, and <SwmToken path="Makefile" pos="6:10:10" line-data="check_dirs := examples tests src utils">`utils`</SwmToken>.

```
check_dirs := examples tests src utils
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="8">

---

# Exclude Folders

The <SwmToken path="Makefile" pos="8:0:0" line-data="exclude_folders :=  &quot;&quot;">`exclude_folders`</SwmToken> variable is initialized as an empty string to specify folders to be excluded from checks.

```
exclude_folders :=  ""
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="10">

---

# Modified Only Fixup

The <SwmToken path="Makefile" pos="10:0:0" line-data="modified_only_fixup:">`modified_only_fixup`</SwmToken> target checks and fixes only the modified Python files using <SwmToken path="Makefile" pos="14:1:1" line-data="		ruff check $(modified_py_files) --fix --exclude $(exclude_folders); \">`ruff`</SwmToken>.

```
modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		ruff check $(modified_py_files) --fix --exclude $(exclude_folders); \
		ruff format $(modified_py_files) --exclude $(exclude_folders);\
	else \
		echo "No library .py files were modified"; \
	fi
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="22">

---

# Dependency Table Update

The <SwmToken path="Makefile" pos="22:0:0" line-data="deps_table_update:">`deps_table_update`</SwmToken> target updates the <SwmPath>[src/transformers/dependency_versions_table.py](src/transformers/dependency_versions_table.py)</SwmPath> file by running <SwmToken path="Makefile" pos="23:2:8" line-data="	@python setup.py deps_table_update">`python setup.py deps_table_update`</SwmToken>.

```
deps_table_update:
	@python setup.py deps_table_update
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="25">

---

# Dependency Table Check Updated

The <SwmToken path="Makefile" pos="25:0:0" line-data="deps_table_check_updated:">`deps_table_check_updated`</SwmToken> target checks if the dependency table is up-to-date and prompts the user to run <SwmToken path="Makefile" pos="28:41:43" line-data="	@md5sum -c --quiet md5sum.saved || (printf &quot;\nError: the version dependency table is outdated.\nPlease run &#39;make fixup&#39; or &#39;make style&#39; and commit the changes.\n\n&quot; &amp;&amp; exit 1)">`make fixup`</SwmToken> or <SwmToken path="Makefile" pos="28:49:51" line-data="	@md5sum -c --quiet md5sum.saved || (printf &quot;\nError: the version dependency table is outdated.\nPlease run &#39;make fixup&#39; or &#39;make style&#39; and commit the changes.\n\n&quot; &amp;&amp; exit 1)">`make style`</SwmToken> if it is outdated.

```
deps_table_check_updated:
	@md5sum src/transformers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="33">

---

# Autogenerate Code

The <SwmToken path="Makefile" pos="33:0:0" line-data="autogenerate_code: deps_table_update">`autogenerate_code`</SwmToken> target updates the dependency table by calling the <SwmToken path="Makefile" pos="33:3:3" line-data="autogenerate_code: deps_table_update">`deps_table_update`</SwmToken> target.

```
autogenerate_code: deps_table_update
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="37">

---

# Repo Consistency

The <SwmToken path="Makefile" pos="37:0:2" line-data="repo-consistency:">`repo-consistency`</SwmToken> target runs various scripts to check the consistency of the repository, such as <SwmPath>[utils/check_copies.py](utils/check_copies.py)</SwmPath>, <SwmPath>[utils/check_table.py](utils/check_table.py)</SwmPath>, and <SwmPath>[utils/check_repo.py](utils/check_repo.py)</SwmPath>.

```
repo-consistency:
	python utils/check_copies.py
	python utils/check_table.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/check_inits.py
	python utils/check_config_docstrings.py
	python utils/check_config_attributes.py
	python utils/check_doctest_list.py
	python utils/update_metadata.py --check-only
	python utils/check_docstrings.py
	python utils/check_support_list.py
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="52">

---

# Quality Check

The <SwmToken path="Makefile" pos="52:0:0" line-data="quality:">`quality`</SwmToken> target runs checks on all files to ensure the repository is in a good state. It includes import checks, <SwmToken path="Makefile" pos="54:1:1" line-data="	ruff check $(check_dirs) setup.py conftest.py">`ruff`</SwmToken> checks, and custom sorting scripts.

```
quality:
	@python -c "from transformers import *" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)
	ruff check $(check_dirs) setup.py conftest.py
	ruff format --check $(check_dirs) setup.py conftest.py
	python utils/custom_init_isort.py --check_only
	python utils/sort_auto_mappings.py --check_only
	python utils/check_doc_toc.py
	python utils/check_docstrings.py --check_all
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="64">

---

# Extra Style Checks

The <SwmToken path="Makefile" pos="64:0:0" line-data="extra_style_checks:">`extra_style_checks`</SwmToken> target runs additional style checks and fixes using custom scripts.

```
extra_style_checks:
	python utils/custom_init_isort.py
	python utils/sort_auto_mappings.py
	python utils/check_doc_toc.py --fix_and_overwrite
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="71">

---

# Style Check and Fix

The <SwmToken path="Makefile" pos="71:0:0" line-data="style:">`style`</SwmToken> target runs <SwmToken path="Makefile" pos="72:1:1" line-data="	ruff check $(check_dirs) setup.py conftest.py --fix --exclude $(exclude_folders)">`ruff`</SwmToken> checks and formatting on all files and calls the <SwmToken path="Makefile" pos="74:6:6" line-data="	${MAKE} autogenerate_code">`autogenerate_code`</SwmToken> and <SwmToken path="Makefile" pos="75:6:6" line-data="	${MAKE} extra_style_checks">`extra_style_checks`</SwmToken> targets.

```
style:
	ruff check $(check_dirs) setup.py conftest.py --fix --exclude $(exclude_folders)
	ruff format $(check_dirs) setup.py conftest.py --exclude $(exclude_folders)
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="79">

---

# Fixup

The <SwmToken path="Makefile" pos="79:0:0" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency">`fixup`</SwmToken> target is a fast fix and check target that works on modified files since the branch was created. It calls <SwmToken path="Makefile" pos="79:3:3" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency">`modified_only_fixup`</SwmToken>, <SwmToken path="Makefile" pos="79:5:5" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency">`extra_style_checks`</SwmToken>, <SwmToken path="Makefile" pos="79:7:7" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency">`autogenerate_code`</SwmToken>, and <SwmToken path="Makefile" pos="79:9:11" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency">`repo-consistency`</SwmToken>.

```
fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="83">

---

# Fix Copies

The <SwmToken path="Makefile" pos="83:0:2" line-data="fix-copies:">`fix-copies`</SwmToken> target makes marked copies of code snippets conform to the original by running various scripts with the <SwmToken path="Makefile" pos="84:9:10" line-data="	python utils/check_copies.py --fix_and_overwrite">`--fix_and_overwrite`</SwmToken> option.

```
fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_table.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite
	python utils/check_doctest_list.py --fix_and_overwrite
	python utils/check_docstrings.py --fix_and_overwrite
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="92">

---

# Run Tests

The <SwmToken path="Makefile" pos="92:0:0" line-data="test:">`test`</SwmToken> target runs tests for the library using <SwmToken path="Makefile" pos="93:6:6" line-data="	python -m pytest -n auto --dist=loadfile -s -v ./tests/">`pytest`</SwmToken>.

```
test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="97">

---

# Run Example Tests

The <SwmToken path="Makefile" pos="97:0:2" line-data="test-examples:">`test-examples`</SwmToken> target runs tests for examples in the <SwmPath>[examples/pytorch/](examples/pytorch/)</SwmPath> directory using <SwmToken path="Makefile" pos="98:6:6" line-data="	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/">`pytest`</SwmToken>.

```
test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="102">

---

# Run Benchmark

The <SwmToken path="Makefile" pos="102:0:0" line-data="benchmark:">`benchmark`</SwmToken> target runs benchmarks using the <SwmPath>[benchmark/benchmark.py](benchmark/benchmark.py)</SwmPath> script with specified configurations.

```
benchmark:
	python3 benchmark/benchmark.py --config-dir benchmark/config --config-name generation --commit=diff backend.model=google/gemma-2b backend.cache_implementation=null,static backend.torch_compile=false,true --multirun
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="107">

---

# Run <SwmToken path="Makefile" pos="107:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> Tests

The <SwmToken path="Makefile" pos="107:0:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`test-sagemaker`</SwmToken> target runs tests for <SwmToken path="Makefile" pos="107:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> DLC release using <SwmToken path="Makefile" pos="108:10:10" line-data="	TEST_SAGEMAKER=True python -m pytest -n auto  -s -v ./tests/sagemaker">`pytest`</SwmToken>. <SwmToken path="Makefile" pos="107:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> dependencies should be installed in advance.

```
test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]
	TEST_SAGEMAKER=True python -m pytest -n auto  -s -v ./tests/sagemaker
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="113">

---

# Release Preparation

The <SwmToken path="Makefile" pos="113:0:2" line-data="pre-release:">`pre-release`</SwmToken>, <SwmToken path="Makefile" pos="116:0:2" line-data="pre-patch:">`pre-patch`</SwmToken>, <SwmToken path="Makefile" pos="119:0:2" line-data="post-release:">`post-release`</SwmToken>, and <SwmToken path="Makefile" pos="122:0:2" line-data="post-patch:">`post-patch`</SwmToken> targets run the <SwmPath>[utils/release.py](utils/release.py)</SwmPath> script with different options to prepare for and finalize releases.

```
pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="125">

---

# Build Release

The <SwmToken path="Makefile" pos="125:0:2" line-data="build-release:">`build-release`</SwmToken> target builds the release by creating distribution files using <SwmPath>[setup.py](setup.py)</SwmPath> and checking the build with <SwmPath>[utils/check_build.py](utils/check_build.py)</SwmPath>.

```
build-release:
	rm -rf dist
	rm -rf build
	python setup.py bdist_wheel
	python setup.py sdist
	python utils/check_build.py
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="general-build-tool"><sup>Powered by [Swimm](/)</sup></SwmMeta>
