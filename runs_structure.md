```
runs
	0.baseline
		0.1.3D CNN followed by logistic regression
			0.1.1.baseline-CustomCNN-3D
				0.1.1.1.baseline-CustomCNN-3D-fold1
					log.txt
					tensorboard-logs
					weights.pt
					predictions.csv
				0.1.1.2.baseline-CustomCNN-3D-fold2
				...
				performance.csv
				summary.json
			0.1.2.baseline-DeepSymNet
			...
	1.siamese
	...
  overall_performance.csv


```

Fold `performance.csv`

```
fold;model_name;best_epoch;set;accuracy;precision;recall;f1_score;auc_score
```



`predictions.csv`

```
patiend_id;proba;y_pred
```



`overall_performance.csv`

```
model_name;type;type_id;exp_id;variation_id;accuracy_avg;accuracy_stddev;precision_avg;precision_stddev;recall_avg;recall_stddev;f1_score_avg;f1_score_stddev;auc_score_avg;auc_score_stddev
```



