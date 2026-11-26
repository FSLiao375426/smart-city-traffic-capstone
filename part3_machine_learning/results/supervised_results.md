# Supervised model results

Training 2012-2016, validation 2017 (model choice), test Jan-Sep 2018 (scored once, models refitted on 2012-2017). The selection rule and the choice are in model_selection.json.

## Regression: traffic_volume

| model                 | selected   |   valid_MAE |   valid_R2 |   test_MAE |   test_RMSE |   test_R2 |   fit_seconds |
|:----------------------|:-----------|------------:|-----------:|-----------:|------------:|----------:|--------------:|
| Dummy (training mean) |            |    1744.24  |     -0.004 |   1728.15  |    1974.81  |    -0     |         0.002 |
| Linear regression     |            |     801.583 |      0.73  |    806.103 |    1017.25  |     0.735 |         0.077 |
| Random forest         |            |     239.458 |      0.963 |    220.437 |     372.223 |     0.964 |        10.474 |
| Gradient boosting     | yes        |     235.891 |      0.965 |    213.226 |     350.082 |     0.969 |         3.804 |

## Classification: high_risk (proxy label)

| model                          | selected   |   valid_F1 |   test_accuracy |   test_precision |   test_recall |   test_F1 |   test_ROC_AUC |
|:-------------------------------|:-----------|-----------:|----------------:|-----------------:|--------------:|----------:|---------------:|
| Dummy (always 'not high risk') |            |      0     |           0.889 |            0     |         0     |     0     |          0.5   |
| Logistic regression            |            |      0.8   |           0.919 |            0.581 |         0.979 |     0.729 |          0.982 |
| Random forest                  |            |      0.954 |           0.982 |            0.866 |         0.996 |     0.926 |          0.997 |
| Gradient boosting              | yes        |      0.957 |           0.984 |            0.884 |         0.983 |     0.931 |          0.998 |
