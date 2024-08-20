---
title: Data Overview
---
Data refers to the input and output files used for training, validation, and testing machine learning models.

The input files are typically named with a `.source` extension, while the output files are named with a `.target` extension.

For training, the files are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/train.source](examples/legacy/seq2seq/test_data/wmt_en_ro/train.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/train.target](examples/legacy/seq2seq/test_data/wmt_en_ro/train.target)</SwmPath>; for validation, they are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/val.source](examples/legacy/seq2seq/test_data/wmt_en_ro/val.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/val.target](examples/legacy/seq2seq/test_data/wmt_en_ro/val.target)</SwmPath>; and for testing, they are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/test.source](examples/legacy/seq2seq/test_data/wmt_en_ro/test.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/test.target](examples/legacy/seq2seq/test_data/wmt_en_ro/test.target)</SwmPath>.

Data collators are objects that form batches from lists of dataset elements, potentially applying preprocessing steps like padding.

Some data collators, such as `DataCollatorForLanguageModeling`, may also apply random data augmentation techniques like random masking.

The `DefaultDataCollator` is a simple data collator that handles batches of dict-like objects, performing special handling for keys named `label` and `label_ids`.

The `DataCollatorForSeq2Seq` dynamically pads the inputs and labels, and can use a model to prepare `decoder_input_ids` if label smoothing is applied.

## Data Organization

The input files are typically named with a `.source` extension, while the output files are named with a `.target` extension. For training, the files are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/train.source](examples/legacy/seq2seq/test_data/wmt_en_ro/train.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/train.target](examples/legacy/seq2seq/test_data/wmt_en_ro/train.target)</SwmPath>; for validation, they are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/val.source](examples/legacy/seq2seq/test_data/wmt_en_ro/val.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/val.target](examples/legacy/seq2seq/test_data/wmt_en_ro/val.target)</SwmPath>; and for testing, they are named <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/test.source](examples/legacy/seq2seq/test_data/wmt_en_ro/test.source)</SwmPath> and <SwmPath>[examples/legacy/seq2seq/test_data/wmt_en_ro/test.target](examples/legacy/seq2seq/test_data/wmt_en_ro/test.target)</SwmPath>.

## Data Collators

Data collators are objects that form batches from lists of dataset elements, potentially applying preprocessing steps like padding. Some data collators, such as `DataCollatorForLanguageModeling`, may also apply random data augmentation techniques like random masking.

## Default Data Collator

The `DefaultDataCollator` is a simple data collator that handles batches of dict-like objects, performing special handling for keys named `label` and `label_ids`.

## Data Collator for Seq2Seq

The `DataCollatorForSeq2Seq` dynamically pads the inputs and labels, and can use a model to prepare `decoder_input_ids` if label smoothing is applied.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBdHJhbnNmb3JtZXJzJTNBJTNBc2h1anV1dQ==" repo-name="transformers" doc-type="overview"><sup>Powered by [Swimm](/)</sup></SwmMeta>
