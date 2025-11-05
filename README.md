# Classification of KL grades of Knee Osteoarthritis
Deep Learning project focused in classifying different KL grades of Osteoarthritis in the knee region. 

## Results
Results were achieved using transfer learning on Resnet50.

### Validation
Using k-fold cross validation the following aggregated metrics were collected: (mean ± std)
```
  Loss: 1.1931 ± 0.1437
  Accuracy: 0.6165 ± 0.0420
  Precision (macro): 0.6453 ± 0.0447
  Recall (macro): 0.6465 ± 0.0275
  F1 Score (macro): 0.6392 ± 0.0359
  Cohen's Kappa: 0.4663 ± 0.0525
  ROC-AUC (macro): 0.8519 ± 0.0200
  
  Per-class F1: ['0.711', '0.297', '0.570', '0.754', '0.865'] ± ['0.044', '0.045', '0.052', '0.049', '0.049']
  Per-class Precision: ['0.708', '0.340', '0.550', '0.793', '0.835'] ± ['0.040', '0.064', '0.068', '0.058', '0.097']
  Per-class Recall: ['0.730', '0.270', '0.605', '0.721', '0.907'] ± ['0.114', '0.047', '0.082', '0.060', '0.064']
```
### Evaluation
Soft voting of k-fold ensemble was utilized for evaluation on the held-out test set. 
Metrics of ensemble evaluation:
```
  Loss: 0.8003
  Accuracy: 0.6921
  Precision: 0.6992
  Recall: 0.7055
  Macro F1 Score: 0.6972
  Cohen's Kappa: 0.5649
  ROC-AUC Macro: 0.9082
  Per-class F1: ['0.780', '0.290', '0.667', '0.852', '0.897']
  Per-class Precision: ['0.717', '0.385', '0.675', '0.852', '0.867']
  Per-class Recall: ['0.856', '0.233', '0.659', '0.852', '0.929']
```
#### Confusion Matrix of ensemble evaluation:
![Confusion Matrix of Ensemble Predictions](results/confusion_matrix_ensemble.png)
