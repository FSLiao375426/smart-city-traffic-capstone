# Supervised model results

Test set: 2018 (chronological split).

## Regression: traffic_volume

| model                  |     MAE |     RMSE |    R2 |   train_R2 |   fit_seconds |
|:-----------------------|--------:|---------:|------:|-----------:|--------------:|
| linear_regression      | 805.695 | 1016.77  | 0.735 |      0.732 |         0.084 |
| random_forest          | 221.778 |  375.546 | 0.964 |      0.982 |         3.514 |
| hist_gradient_boosting | 216.599 |  361.843 | 0.966 |      0.973 |         1.327 |

## Classification: high_risk (proxy label)

| model                  |   accuracy |   precision |   recall |    F1 |   ROC_AUC |
|:-----------------------|-----------:|------------:|---------:|------:|----------:|
| logistic_regression    |      0.959 |       0.691 |    0.992 | 0.814 |     0.994 |
| random_forest          |      0.983 |       0.848 |    0.997 | 0.916 |     0.997 |
| hist_gradient_boosting |      0.985 |       0.867 |    0.992 | 0.925 |     0.998 |
