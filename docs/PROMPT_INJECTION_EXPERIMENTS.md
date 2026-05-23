# Prompt Injection Guard Experiments

The project evaluated multiple prompt-injection datasets and models.

## Datasets

- `neuralchemy/Prompt-injection-dataset`
- `deepset/prompt-injections`
- `geekyrakshit/prompt-injection-dataset`

## Key Findings

A model can perform very well on its own dataset while failing to generalize to another dataset. This was observed during cross-dataset evaluation between Geekyrakshit and Neuralchemy.

## Neuralchemy DistilBERT Result

- Accuracy: 0.9586
- Precision unsafe: 0.9605
- Recall unsafe: 0.9692
- F1 unsafe: 0.9648
- Confusion matrix: `[[368, 22], [17, 535]]`

## Deepset DistilBERT Result

- Accuracy: 0.9482
- Precision unsafe: 1.0000
- Recall unsafe: 0.9000
- F1 unsafe: 0.9473
- Confusion matrix: `[[56, 0], [6, 54]]`

## Geekyrakshit DistilBERT Result

- Accuracy: 0.9993
- Precision unsafe: 0.9997
- Recall unsafe: 0.9988
- F1 unsafe: 0.9992
- Confusion matrix: `[[133939, 33], [151, 129791]]`

## Cross-Dataset Lesson

High in-distribution performance is not enough. Security models must also be evaluated on out-of-distribution datasets and manual hard cases.
