# Web Attack Guard Experiments

This module detects SQL Injection, XSS, LFI/path traversal, command injection, encoded payloads, and other web attack patterns.

## Dataset

The first Web Attack Guard was trained on:

- `truongp/web-attack-detection`

Dataset scale:

- Total rows: 625,904
- Benign: 331,129
- Attack: 294,775

## Web Attack Guard v1

Base model:

- `distilbert-base-uncased`

Full 1-epoch training result:

- Accuracy: 0.9927
- Precision attack: 0.9957
- Recall attack: 0.9887
- F1 attack: 0.9922
- Confusion matrix: `[[32964, 126], [333, 29168]]`

## Manual Attack Tests

The v1 model successfully detected:

- login bypass SQLi,
- UNION-based SQLi,
- error-based SQLi,
- boolean-based blind SQLi,
- time-based blind SQLi,
- obfuscated UNION SQLi,
- URL-encoded SQLi,
- XSS,
- LFI/path traversal.

Example attack scores:

- Login bypass SQLi: 0.9919
- UNION SQLi: 0.9991
- Error-based SQLi: 1.0000
- Boolean blind SQLi: 0.9982
- Obfuscated UNION SQLi: 0.99
- URL encoded SQLi: 0.996
- Conditional time-based SQLi: 0.98

## False Positive Problem

Web Attack Guard v1 incorrectly flagged some benign SQL fragments and normal SQL queries, such as:

- `WHERE`
- `SELECT`
- `AND`
- `OR`
- `SELECT * FROM users;`
- `SELECT id, name FROM users WHERE id = 5;`

## SQLShield Experiment

A separate SQLShield model was trained on `salmane11/SQLShield`.

SQLShield DistilBERT result:

- Accuracy: 0.9994
- Precision sql_risk: 0.9988
- Recall sql_risk: 1.0000
- F1 sql_risk: 0.9994
- Confusion matrix: `[[963, 1], [0, 836]]`

However, manual testing showed that SQLShield also flagged benign SQL fragments such as `WHERE`, `SELECT`, `AND`, `OR`, and simple normal SQL queries as SQL risk. Therefore, SQLShield was not selected as the final solution.

## Web Attack Guard v2

The final approach was to fine-tune Web Attack Guard v1 with SQL hard-negative examples.

The v2 training dataset included:

- original benign examples,
- original attack examples,
- benign SQL fragments and normal SQL queries,
- normal INSERT/UPDATE/DELETE statements,
- hard attack payloads including SQLi, XSS, LFI, encoded SQLi, stored SQLi, and obfuscated SQLi.

Web Attack Guard v2 result:

- Accuracy: 0.9949
- Precision attack: 0.9957
- Recall attack: 0.9933
- F1 attack: 0.9945
- Confusion matrix: `[[7076, 26], [41, 6086]]`

## v2 Manual Validation

Web Attack Guard v2 correctly classified:

- `WHERE` as benign,
- `SELECT` as benign,
- `SELECT * FROM users;` as benign,
- normal INSERT/UPDATE/DELETE as benign,
- long benign SQL queries as benign,
- SQL Injection payloads as web attacks,
- XSS as web attack,
- LFI/path traversal as web attack.

One remaining known limitation is that some normal HTTP request headers can still be classified as attacks. This is marked as a future hard-negative fine-tuning task.
