import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

class Data:
    def __init__(self,train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, id_col):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.feature_cols = cat_cols + num_cols
        self.target_col = target_col
        self.id_col = id_col
        self.label_encoders = {}
        self.train_df = self._create_train_df(train_feature_file, train_target_file)
        self.test_df = self._create_test_df(test_file)
        self.target_col = target_col
        
        
    ## Function to create a train dataframe that will load the data, clean and preprocess the data    
    def _create_train_df(self,train_feature_df,train_target_df,preprocess = True, label_encode = True):
        train_features = self._load_data(train_feature_df)
        train_salaries = self._load_data(train_target_df)
        train_df = self._join_data(train_features,train_salaries)
        if preprocess:
            train_df = self._preprocess_data(train_df)
            train_df = self._shuffle_data(train_df)
        if label_encode:
            self.label_encode_df(train_df, self.cat_cols)
        return train_df
    
    
    ## Function to create a test dataframe that will load the data, clean and preprocess the data
    def _create_test_df(self,test_feature_df, label_encode = True):
        test_features = self._load_data(test_feature_df)
        if label_encode:
            self.label_encode_df(test_features, self.cat_cols)
        return test_features 
        
    ## Function to load data usin.l,g pandas    
    def _load_data(self,file):
        return pd.read_csv(file)
    
    
    ## Function to join datasets using pd.merge
    def _join_data (self, train_features, train_salaries):
        train_df = pd.merge(train_features, train_salaries,on = 'jobId')
        return train_df
    
    
    ## Function to preprocess data by dropping duplicates and removing zero salaries entries
    def _preprocess_data(self, df):
        cleaned_train_df = df.drop_duplicates(subset='jobId')
        cleaned_train_df = cleaned_train_df[cleaned_train_df.salary > 0]
        return cleaned_train_df
    
    
    ## Shuffle the dataframe
    def _shuffle_data(self, df):
        return shuffle(df).reset_index()
    
    
    ## Function created for label encoding the variables
    def label_encode_df(self, df, cols):
        '''creates one label encoder for each column in the data object instance'''
        for col in cols:
            if col in self.label_encoders:
                #if label encoder already exits for col, use it
                self._label_encode(df, col, self.label_encoders[col])
            else:
                self._label_encode(df, col)
                
    
    ## Function created for label encoding the variables
    def _label_encode(self, df, col, le=None):
        '''label encodes data'''
        if le:
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            self.label_encoders[col] = le
    
    
    
    