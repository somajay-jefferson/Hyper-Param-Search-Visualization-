# Hyper Param Search Visualization 
 Visualization of Hyperparameter Tuning via OOP paradigm



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

## Getting Started

### Prerequisites

- Python 3.x
- scikit-learn
- seaborne

### Installation

Not yet officially deployed 


### Features to come:

- Support Grid Search, random search, and bayesian search
- Support not only classification but also regression models
- In addition to visualization of hyperparameters, also include AUC for threshold selection in visualization process(final stage)
- Multi-dimension visualization 
