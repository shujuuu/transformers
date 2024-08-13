---
title: Using the Makefile
---
# Intro

This document explains how to use the <SwmPath>[Makefile](Makefile)</SwmPath> in the root directory of the repository. It will cover the various targets defined in the <SwmPath>[Makefile](Makefile)</SwmPath> and their purposes.

<SwmSnippet path="/Makefile" line="1">

---

# Phony Targets

The <SwmToken path="Makefile" pos="1:0:1" line-data=".PHONY: deps_table_update modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs">`.PHONY`</SwmToken> directive declares a list of phony targets. These targets do not represent files and are always executed when invoked.

```
.PHONY: deps_table_update modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="4">

---

# Setting PYTHONPATH

The <SwmToken path="Makefile" pos="4:2:2" line-data="export PYTHONPATH = src">`PYTHONPATH`</SwmToken> environment variable is set to <SwmToken path="Makefile" pos="4:6:6" line-data="export PYTHONPATH = src">`src`</SwmToken>, ensuring that the local checkout is tested in scripts instead of the <SwmToken path="Makefile" pos="3:26:28" line-data="# make sure to test the local checkout in scripts and not the pre-installed one (don&#39;t use quotes!)">`pre-installed`</SwmToken> one.

```
export PYTHONPATH = src
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="6">

---

# Directories to Check

The <SwmToken path="Makefile" pos="6:0:0" line-data="check_dirs := examples tests src utils">`check_dirs`</SwmToken> variable lists the directories (<SwmToken path="Makefile" pos="6:4:4" line-data="check_dirs := examples tests src utils">`examples`</SwmToken>, <SwmToken path="Makefile" pos="6:6:6" line-data="check_dirs := examples tests src utils">`tests`</SwmToken>, <SwmToken path="Makefile" pos="6:8:8" line-data="check_dirs := examples tests src utils">`src`</SwmToken>, <SwmToken path="Makefile" pos="6:10:10" line-data="check_dirs := examples tests src utils">`utils`</SwmToken>) that are checked by various targets.

```
check_dirs := examples tests src utils
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="8">

---

# Modified Only Fixup

The <SwmToken path="Makefile" pos="8:0:0" line-data="modified_only_fixup:">`modified_only_fixup`</SwmToken> target identifies modified Python files and applies formatting and linting tools (<SwmToken path="Makefile" pos="12:1:1" line-data="		black $(modified_py_files); \">`black`</SwmToken>, <SwmToken path="Makefile" pos="13:1:1" line-data="		isort $(modified_py_files); \">`isort`</SwmToken>, <SwmToken path="Makefile" pos="14:1:1" line-data="		flake8 $(modified_py_files); \">`flake8`</SwmToken>) to them.

```
modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="21">

---

# Dependency Table Update

The <SwmToken path="Makefile" pos="21:0:0" line-data="deps_table_update:">`deps_table_update`</SwmToken> target updates the dependency versions table by running <SwmToken path="Makefile" pos="22:2:8" line-data="	@python setup.py deps_table_update">`python setup.py deps_table_update`</SwmToken>.

```
deps_table_update:
	@python setup.py deps_table_update
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="24">

---

# Dependency Table Check Updated

The <SwmToken path="Makefile" pos="24:0:0" line-data="deps_table_check_updated:">`deps_table_check_updated`</SwmToken> target checks if the dependency versions table is up-to-date. If not, it prompts the user to run <SwmToken path="Makefile" pos="27:41:43" line-data="	@md5sum -c --quiet md5sum.saved || (printf &quot;\nError: the version dependency table is outdated.\nPlease run &#39;make fixup&#39; or &#39;make style&#39; and commit the changes.\n\n&quot; &amp;&amp; exit 1)">`make fixup`</SwmToken> or <SwmToken path="Makefile" pos="27:49:51" line-data="	@md5sum -c --quiet md5sum.saved || (printf &quot;\nError: the version dependency table is outdated.\nPlease run &#39;make fixup&#39; or &#39;make style&#39; and commit the changes.\n\n&quot; &amp;&amp; exit 1)">`make style`</SwmToken>.

```
deps_table_check_updated:
	@md5sum src/transformers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="32">

---

# Autogenerate Code

The <SwmToken path="Makefile" pos="32:0:0" line-data="autogenerate_code: deps_table_update">`autogenerate_code`</SwmToken> target updates the dependency versions table and runs <SwmPath>[utils/class_mapping_update.py](utils/class_mapping_update.py)</SwmPath> to autogenerate code.

```
autogenerate_code: deps_table_update
	python utils/class_mapping_update.py
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="37">

---

# Extra Quality Checks

The <SwmToken path="Makefile" pos="37:0:0" line-data="extra_quality_checks:">`extra_quality_checks`</SwmToken> target runs additional quality checks using various scripts in the <SwmToken path="Makefile" pos="38:3:3" line-data="	python utils/check_copies.py">`utils`</SwmToken> directory.

```
extra_quality_checks:
	python utils/check_copies.py
	python utils/check_table.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/check_inits.py
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="45">

---

# Quality Check

The <SwmToken path="Makefile" pos="45:0:0" line-data="quality:">`quality`</SwmToken> target runs formatting and linting tools (<SwmToken path="Makefile" pos="46:1:1" line-data="	black --check $(check_dirs)">`black`</SwmToken>, <SwmToken path="Makefile" pos="47:1:1" line-data="	isort --check-only $(check_dirs)">`isort`</SwmToken>, <SwmToken path="Makefile" pos="49:1:1" line-data="	flake8 $(check_dirs)">`flake8`</SwmToken>) on all files in the <SwmToken path="Makefile" pos="46:8:8" line-data="	black --check $(check_dirs)">`check_dirs`</SwmToken> directories and performs extra quality checks.

```
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	python utils/custom_init_isort.py --check_only
	flake8 $(check_dirs)
	${MAKE} extra_quality_checks
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="54">

---

# Extra Style Checks

The <SwmToken path="Makefile" pos="54:0:0" line-data="extra_style_checks:">`extra_style_checks`</SwmToken> target runs additional style checks using scripts in the <SwmToken path="Makefile" pos="55:3:3" line-data="	python utils/custom_init_isort.py">`utils`</SwmToken> directory.

```
extra_style_checks:
	python utils/custom_init_isort.py
	python utils/style_doc.py src/transformers docs/source --max_len 119
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="59">

---

# Style Check

The <SwmToken path="Makefile" pos="59:0:0" line-data="style:">`style`</SwmToken> target formats the code using <SwmToken path="Makefile" pos="60:1:1" line-data="	black $(check_dirs)">`black`</SwmToken> and <SwmToken path="Makefile" pos="61:1:1" line-data="	isort $(check_dirs)">`isort`</SwmToken>, autogenerates code, and performs extra style checks.

```
style:
	black $(check_dirs)
	isort $(check_dirs)
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="67">

---

# Fixup

The <SwmToken path="Makefile" pos="67:0:0" line-data="fixup: modified_only_fixup extra_style_checks autogenerate_code extra_quality_checks">`fixup`</SwmToken> target performs a quick fix and check on modified files, runs extra style checks, autogenerates code, and performs extra quality checks.

```
fixup: modified_only_fixup extra_style_checks autogenerate_code extra_quality_checks
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="71">

---

# Fix Copies

The <SwmToken path="Makefile" pos="71:0:2" line-data="fix-copies:">`fix-copies`</SwmToken> target ensures that marked copies of code snippets conform to the original by running various scripts in the <SwmToken path="Makefile" pos="72:3:3" line-data="	python utils/check_copies.py --fix_and_overwrite">`utils`</SwmToken> directory.

```
fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_table.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="78">

---

# Run Tests

The <SwmToken path="Makefile" pos="78:0:0" line-data="test:">`test`</SwmToken> target runs the test suite using <SwmToken path="Makefile" pos="84:6:6" line-data="	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/">`pytest`</SwmToken> with parallel execution enabled.

```
test:
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="83">

---

# Run Example Tests

The <SwmToken path="Makefile" pos="83:0:2" line-data="test-examples:">`test-examples`</SwmToken> target runs tests for the examples in the <SwmPath>[examples/pytorch/](examples/pytorch/)</SwmPath> directory using <SwmToken path="Makefile" pos="84:6:6" line-data="	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/">`pytest`</SwmToken> with parallel execution enabled.

```
test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="88">

---

# Run <SwmToken path="Makefile" pos="88:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> Tests

The <SwmToken path="Makefile" pos="88:0:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`test-sagemaker`</SwmToken> target runs tests for the <SwmToken path="Makefile" pos="88:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> DLC release. <SwmToken path="Makefile" pos="88:2:2" line-data="test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]">`sagemaker`</SwmToken> dependencies should be installed in advance.

```
test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]
	TEST_SAGEMAKER=True python -m pytest -n auto  -s -v ./tests/sagemaker
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="94">

---

# Build Documentation

The <SwmToken path="Makefile" pos="94:0:0" line-data="docs:">`docs`</SwmToken> target builds the documentation using Sphinx with the <SwmToken path="Makefile" pos="95:14:15" line-data="	cd docs &amp;&amp; make html SPHINXOPTS=&quot;-W -j 4&quot;">`-W`</SwmToken> and <SwmToken path="Makefile" pos="95:17:20" line-data="	cd docs &amp;&amp; make html SPHINXOPTS=&quot;-W -j 4&quot;">`-j 4`</SwmToken> options.

```
docs:
	cd docs && make html SPHINXOPTS="-W -j 4"
```

---

</SwmSnippet>

<SwmSnippet path="/Makefile" line="99">

---

# Release Targets

The <SwmToken path="Makefile" pos="99:0:2" line-data="pre-release:">`pre-release`</SwmToken>, <SwmToken path="Makefile" pos="102:0:2" line-data="pre-patch:">`pre-patch`</SwmToken>, <SwmToken path="Makefile" pos="105:0:2" line-data="post-release:">`post-release`</SwmToken>, and <SwmToken path="Makefile" pos="108:0:2" line-data="post-patch:">`post-patch`</SwmToken> targets handle various stages of the release process by running <SwmPath>[utils/release.py](utils/release.py)</SwmPath> with appropriate options.

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

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers"><sup>Powered by [Swimm](/)</sup></SwmMeta>
