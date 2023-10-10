# Hyper Param Search Visualization 
 Visualization of Hyperparameter Tuning via OOP paradigm

## Files

**Grid_Search_Visual.py:** the py file containing the `gridsearch_visual` class 

**Gridsearch.ipynb:** scratch work for `gridsearch_visual`, contains some test code and work in progress code

**test_gridsearch_visual.py:** unit test for `Grid_Search_Visual.py`


# gridsearch_visual

`gridsearch_visual` is a Python class that simplifies hyperparameter tuning and visualization of results using scikit-learn's GridSearchCV.

## Overview

The `gridsearch_visual` class streamlines the process of hyperparameter tuning by integrating with scikit-learn's `GridSearchCV`. It not only automates the search over specified hyperparameter values but also provides visualizations to understand the impact of hyperparameter changes on model performance and training times.

## Features

- **Ease of Integration**: Seamlessly integrates with scikit-learn models.
- **Automated Grid Search**: Conducts grid search across user-defined hyperparameter values.
- **Performance Visualization**: Generates visualizations of performance metrics and training times.
- **Metric Customization**: Allows customization of evaluation metrics.
- **Task Compatibility**: Works with classification and regression class objects in scikit-learn, xgboost, and even pytorch(as long as it is a class)


### Features to come:

- Support Grid Search, random search, and bayesian search
- Support not only classification but also regression models
- Computes feature importance
- In addition to visualization of hyperparameters, also include AUC for threshold selection in visualization process(final stage)
- Multi-dimension visualization 

## Getting Started

### Prerequisites

- Python 3.x
- scikit-learn
- seaborne

### Installation

Not yet officially deployed 




## Demo

Here's a basic example of how to use the gridsearch_visual class:


suppose you have a classifier, say, a random forest classifier and you have a list of hyper_parameters you would like to tune in `param_grd`:



``` python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X = iris.data
y = iris.target
param_grid = {'n_estimators': [10, 20, 30, 40, 50], 'max_depth': [5, 10, 15, 20, 25]}
clf_base = RandomForestClassifier(random_state=9)  #test_size=0.2, random_state=9, cv = 4, best_param_metric = "accuracy" as default parameters
demo_iris = gridsearch_visual(clf_base, X, y, param_grid)
```

Now that you have created a `gridsearch_visual` object, simply reference 
``` python
demo_iris.hyperparameter_tune_grid_search()
```

woud yield:

![image](https://github.com/somajay-jefferson/Hyper-Param-Search-Visualization-/assets/98189101/dc45e9db-cbf2-4126-9f8d-eeb8c4ece6bd)




With this information, you would likely choose `n_estimators = 20` and  `max_depth = 10` to maximize validation(I know... and trust me it was just miss labeled) and minimize run time

## Acknowledgements

- This project is inspired by this post by Daniel J. Toth(https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d)
- Also by tired of figuring out hyperparameters when training an xgboost tree
