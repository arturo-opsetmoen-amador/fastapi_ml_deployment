# Model Card
Model card for model reporting. This report follows the 
standard introduced in the Model Card paper: 
https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Developed by Arturo Opsetmoen Amador at Alv AS, 2022, v1.
* XGBoost model.
* Trained in the Census Income Dataset (more details below). 
* Bayesian search was used for hyperparameter tuning. 
* Stratified K-fold cross validation was used to evaluate
model performance.

## Intended Use
* This model is part of a web application deployment. The model
will be deployed to Heroku and its predictions will be served 
via FastAPI. 
* The model is used to predict whether the salary of a subject
exceeds the $50K/yr threshold based on the input from the census
data. 

## Training Data
* The data used to train the XGBoost model is the Census Income Dataset
from the University of California, Irvine (UCI).
* This is a multivariate dataset with 48842 instances and 14 
attributes. 
* For more details about the dataset please visit https://archive.ics.uci.edu/ml/datasets/census+income 

## Evaluation Data
* The data used to evaluate the model comes from the same
Census Income Dataset. 
* We used a stratified data splitting to 
hold out 20% of the dataset for evaluation. 
* The remaining 80% of the data was passed through a Bayesian
hyperparameter tuning to find XGBoost's best parameters.
* Stratified K-fold validation was used in the selection of the
best parameters. 
* After finding the best parameters, the XGBoost model was
trained with the full (80% of the dataset) training set. 

## Metrics
Stratified k-fold x-validation with a K = 10 used to (Bayes) search 
the space of the best parameters for the XGBoost model. The metrics
obtained on the hold-out set were as followed: 

* Test xgb_Precision: 0.71847739888977
* Test xgb_Recall: 0.6031957390146472
* Test xgb_FBeta: 0.6558089033659066

The best paremeter dictionary after Bayes search is recorded in the 
```best_params.pkl``` file under models/.

## Ethical Considerations

* This model should be used for didactic purposes only as the
dataset used for training was collected in 1994 and for a particular
geographical region. 
* Furthermore, the data includes features such as sex, race, and
native country which could introduce biases against minorities.

## Caveats and Recommendations
* Due to the ethical considerations mentioned above, this data and
the model trained, should not be used in any real world setting. 
* To get better performance other models could be considered as part
of the "parameter" tunning. For example CatBoost, or LightGMB. 