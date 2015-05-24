All using 1K data-points.

Simplest: One vs. Rest classifier using Linear SVM

Accuracy: 50.15%

```
Classification accuracy on training data: 0.501429
             precision    recall  f1-score   support

          0       0.52      0.94      0.67       359
          1       0.44      0.04      0.08       289
          2       0.00      0.00      0.00        28
          3       0.00      0.00      0.00        19
          4       0.09      0.33      0.14         3
          5       0.00      0.00      0.00         1
          6       0.00      0.00      0.00         1

avg / total       0.45      0.50      0.37       700
```

Simplest 2: One vs. One classifier using Linear SVM

Accuracy: 51.71%

```
Classification accuracy on training data: 0.517143
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       0.52      0.99      0.68       359
          1       0.83      0.02      0.03       289
          2       0.00      0.00      0.00        28
          3       0.00      0.00      0.00        19
          4       0.08      0.33      0.13         3
          5       0.00      0.00      0.00         1
          6       0.00      0.00      0.00         1

avg / total       0.61      0.52      0.37       700
```