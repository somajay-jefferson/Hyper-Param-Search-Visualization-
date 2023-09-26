from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import graphviz
import os
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


class gridsearch_visual():
    '''
    #specify default model to use, e.g.:
    xgb_model_base = xgb.XGBClassifier(
        tree_method="hist",
        random_state = 9,
        objective='binary:logistic' )
    '''
    def __init__(self, classifer, X, y, param_grid, test_size=0.2, random_state=9, cv = 4, best_param_metric = "accuracy"):
        '''
        Initialize the gridsearch_visual object.

        Args:
            classifier (object): The classifier to use.. e.g. logistic reg, xgboost trees
            X (array-like): Input features.
            y (array-like): Target values.
            param_grid (dict): Hyperparameter grid.
            test_size (float, optional): Test set size. Defaults to 0.2.
            random_state (int, optional): Random state. Defaults to 9.
            cv (int, optional): Number of cross-validation folds. Defaults to 4.
            best_param_metric (str, optional): The best metric to use for evaluation. Defaults to accuracy.
        '''
        
        
        
        self.X = X
        self.y = y
        self.clf_base = classifer
        self.cv = cv
        self.grid_key = np.NAN
        self.grid_value = np.NAN

        self.best_param_metric = best_param_metric
        self.default_param = classifer.get_params()#self.param_to_list(classifer.get_params())    #to later throw into gridsearch
        self.default_param_0 = deepcopy(self.default_param)
        self.param_grid = param_grid   #specify in global 


        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state) # Default 80-20
        
        
        self.clf = GridSearchCV(estimator=self.clf_base, 
                                param_grid={},   #no hyper param grid
                                scoring=['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'f1', 'roc_auc', 'recall'], 
                                return_train_score=True, 
                                verbose=3, 
                                cv=self.cv, 
                                n_jobs= -1, 
                                refit = False).fit(self.X_train, self.y_train)  


        #self.metric_cache = {}    #self.train_and_evaluate()  
        self.results_dict = {'clf0' : self.clf.cv_results_} #initialize cache dict that would take in the metric number for later grid searcb
        self.best_hyper_param = {key: np.NaN for key in self.param_grid.keys()}



    def __repr__():


        return f"gridsearch_visual(classifier={self.clf_base.__class__.__name__}, cv={self.cv}, best_param_metric={self.best_param_metric})"




    #def param_to_list(self, param_dict):
    #    #default parameters have to be wrapped in lists - even single values - so GridSearchCV can take them as inputs
    #    param_dict_cleaned = {}
    #    for key in param_dict.keys():
    #        param_dict_cleaned[key] = [param_dict[key]]
    #    return param_dict_cleaned
    


    def train_and_evaluate(self, num, search_param, best_param_metric ):
        #self.clf_base.fit(self.X_train, self.y_train)  #train the classifier



        # update self.clf with param_grid 
        self.clf = GridSearchCV(estimator=self.clf_base, 
                                param_grid=search_param, 
                                scoring=['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'f1', 'roc_auc', 'recall'], 
                                return_train_score=True, 
                                verbose=3, 
                                cv=self.cv, 
                                n_jobs= -1, 
                                refit = best_param_metric)
        

        #self.y_train_pred = self.clf.predict(self.X_train)
        #self.y_val_pred = self.clf.predict(self.X_val)
        #print(self.clf_base)
        #print(self.clf)
        #print(search_param)
        #print(self.clf.best_params_)
        self.clf.fit(self.X_train, self.y_train)
        self.results_dict[f'clf{num+1}'] = self.clf.cv_results_
        self.best_hyper_param = {key: self.clf.best_params_[key] if key in self.clf.best_params_ else value for key, value in self.best_hyper_param.items()}


        # update default param to use the best hyperparam tuned 
        self.update_classifier()



    def update_classifier(self):
        # Update the classifier with new parameters
        #for key, value_default in self.default_param.items():
        #    for key_best, value_best in self.best_hyper_param.items():
        #        
        #        if key_best == key and not  math.isnan(self.best_hyper_param[key_best]):
        #            self.default_param[key] = self.best_hyper_param[key_best]
        #update the clf_base with updated default parameters to include the best tuned hyperparameter from self.best_hyper_param
        self.default_param[self.grid_key] = self.best_hyper_param[self.grid_key]
        #print(self.default_param)
        self.clf_base.set_params(**self.default_param)
        #self.clf_base = self.clf_base.fit(self.X_train, self.y_train)

  



    def hyperparameter_tune_grid_search(self, visualize = True):
        #this should populate self.results_dict and self.best_hyper_param


        for index, (grid_key, values) in enumerate(self.param_grid.items()):
         
            self.grid_key = grid_key
            self.grid_value = values
            print(self.grid_key)
            self.train_and_evaluate(index, {grid_key: values}, self.best_param_metric)

        if visualize == True:
            self.visualize_h_param()
    
    
    
    
    def visualize_h_param(self, metric = None):
        if metric is None:
            metric = self.best_param_metric

        results_dict = self.results_dict
        nrows = len(results_dict.keys())



        fig, ax = plt.subplots(nrows,2,figsize=(27,8*nrows))

        for row in range(nrows):

            clf_number = list(results_dict.keys())[row] 


            if clf_number != 'clf0':
                hyper_param_name = [key for item in results_dict[clf_number]['params'] for key in item.keys()][0]
                data_range = param_grid[hyper_param_name]
        
            #print(clf_number, hyper_param_name)
        
        
        #metric
                y1 = results_dict[clf_number]['mean_train_accuracy']#.loc[:,'mean_train_score']
                y2 = results_dict[clf_number]['mean_test_accuracy']#.loc[:,'mean_test_score']
                x = np.arange(len(data_range))


                ax[row, 0].plot(x, y1, label='train scores', color='blue')
                ax[row, 0].plot(x, y2, label='test scores', color='red')
                ax[row, 0].set_title(f'Iteration #{row+1} metric')


                ax[row, 0].set_xticks(x)
                ax[row, 0].set_xticklabels([str(value) for value in data_range])


                ax[row, 0].grid('major')
                ax[row, 0].legend()
                ax[row, 0].set_xlabel(hyper_param_name)
                ax[row, 0].set_ylabel('mean score')



                #run time
                y3 = results_dict[clf_number]['mean_fit_time']#.loc[:,'mean_train_score']
                ax[row, 1].plot(x, y3, label='run time', color='red')
                #ax[row, 1].set_title(f'Iteration #{i+1} results')
                ax[row, 1].set_title(f'Iteration #{row+1} runtime')
        
                ax[row, 1].set_xticks(x)
                ax[row, 1].set_xticklabels([str(value) for value in data_range])

                ax[row, 1].grid('major')
                #ax[row, 1].legend()
                ax[row, 1].set_xlabel(hyper_param_name)
                ax[row, 1].set_ylabel('seconds')

            else:
                ax[row, 0].axis('off')
                ax[row, 0].text(x=0.5, y=0.5, s='No iteration has been performed; accuracy: ' + str(np.round(results_dict['clf0']['mean_test_accuracy'][0], decimals=2)), fontsize=16, va='center', ha='center')


                #run time
                ax[row, 1].axis('off')
                ax[row, 1].text(x=0.5, y=0.5, s="run time: " + str(np.round(results_dict['clf0']['mean_fit_time'][0], decimals=2))+" seconds", fontsize=16, va='center', ha='center')

            plt.show()

        

    

    


