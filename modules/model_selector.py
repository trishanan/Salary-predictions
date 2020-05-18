import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class model_container:
    def __init__(self,models = []):
        self.models = models
        self.best_model = None
        self.mean_mse = {}
    
    ## Adding a number of models to models list
    def add_model (self,model):
        self.models.append(model)
    
    
    ## Function to create cross-validation score for a number of models
    def cross_validate_score(self,data,k=3):
        features = data.train_df[data.feature_cols]
        target = data.train_df[data.target_col]
        for model in self.models:
            mse = cross_val_score(model, features, target, cv=k, n_jobs = 4, scoring = 'neg_mean_squared_error')
            self.mean_mse[model] = -1.0 * np.mean(mse)
    
    
    ## Function to select the best model
    def select_best_model(self):
        self.best_model = min(self.mean_mse, key = self.mean_mse.get)
        
    
    ## Function to fit the best model
    def fit_best_model(self, features,target):
        self.best_model.fit(features,target)
    
    
    ## Function to predict using the best model
    def predict_using_best_model (self, features):
        self.best_model.predict(features)
    
    
    ## Function to get the feature importances of each model
    def get_feature_importance(self,model,columns):
        if hasattr(model,'feature_importances_'):
            important_features = model.feature_importances_
            important_features_df = pd.DataFrame({'feature' : columns, 'importance': important_features})
            important_features_df.sort_values(by = 'importance', ascending = False, inplace = True)
            important_features_df.set_index('feature',inplace = True, drop = True)
            return important_features_df
        else:
            return 'Feature importance is not an attribute of this model'
        
        
    ## Function to print the summary of the results of the best model and plot the feature importances
    def print_summary(self,data):
        '''prints summary of models, best model, and feature importance'''
        print('\nModel Summaries:\n')
        for model in self.mean_mse:
            print('\n', model, '- MSE:', self.mean_mse[model])
        print('\nBest Model:\n', self.best_model)
        print('\nMSE of Best Model\n', self.mean_mse[self.best_model])
        print('\nFeature Importances\n', self.get_feature_importance(self.best_model, data.feature_cols))

        feature_importances = self.get_feature_importance(self.best_model, data.feature_cols)
        feature_importances.plot.bar()
     