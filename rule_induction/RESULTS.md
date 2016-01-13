Feature Extraction: to deck

Simplest: One vs. Rest classifier using Linear SVM

Accuracy: 49.31%

```
Classification accuracy on training data: 0.493174
             precision    recall  f1-score   support

          0       0.50      0.99      0.66      8705
          1       0.00      0.00      0.00      7447
          2       0.00      0.00      0.00       842
          3       0.00      0.00      0.00       370
          4       0.00      0.00      0.00        68
          5       0.00      0.00      0.00        40
          6       0.00      0.00      0.00        23
          7       0.00      0.00      0.00         5
          8       0.00      0.00      0.00         3
          9       0.00      0.00      0.00         4

avg / total       0.25      0.49      0.33     17507
```

Simplest 2: One vs. One classifier using Linear SVM

Accuracy: 49.72%

```
Classification accuracy on training data: 0.497230
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       0.50      1.00      0.66      8705
          1       0.00      0.00      0.00      7447
          2       0.00      0.00      0.00       842
          3       0.00      0.00      0.00       370
          4       0.00      0.00      0.00        68
          5       0.00      0.00      0.00        40
          6       0.00      0.00      0.00        23
          7       0.00      0.00      0.00         5
          8       0.00      0.00      0.00         3
          9       0.00      0.00      0.00         4

avg / total       0.25      0.50      0.33     17507
```