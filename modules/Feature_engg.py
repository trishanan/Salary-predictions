import pandas as pd
import numpy as np


class Feature_generator:
    def __init__(self,data):
        ## Generating new features for different categorical variables
        self.data =  data
        self.cat_cols = data.cat_cols
        self.groups = data.train_df.groupby(self.cat_cols) ## Creating a groupby object
    
    
    ## Create 
    def add_group_stats(self):
        group_statistics = self._get_group_stats()
        group_statistics.reset_index(inplace = True)
        
        self.data.train_df = self._merge_dfs(self.data.train_df,group_statistics,self.cat_cols,fillna = True)
        self.data.test_df = self._merge_dfs(self.data.test_df,group_statistics,self.cat_cols,fillna = True)
        self.data.train_df = self.data.train_df.sample(n = 20000,replace = True, random_state = 2)
        self.data.test_df = self.data.test_df.sample(n = 20000,replace = True, random_state = 2)
        group_stats_cols = ['group_mean', 'group_max', 'group_min', 'group_std', 'group_median']
        self._extend_col_lists(self.data, cat_cols=group_stats_cols)  
        
    ## Create a dataframe that has various statistics for different categories
    def _get_group_stats(self):
        target = self.data.target_col
        group_stats_df = pd.DataFrame({"group_mean": self.groups[target].mean()})
        group_stats_df['group_max'] = self.groups[target].max()
        group_stats_df['group_min'] = self.groups[target].min()
        group_stats_df['group_std'] = self.groups[target].std()
        group_stats_df['group_median'] = self.groups[target].median()
        return group_stats_df
    
    
    ## Merge dataframes based on different categorical groups
    def _merge_dfs (self, df, new_cols_df,keys,fillna = False):
        result_df = pd.merge(df,new_cols_df,on = keys, how = "left")
        if fillna:
            result_df.fillna(0,inplace = True)
        return result_df
    
    
    def _extend_col_lists(self, data, cat_cols=[], num_cols=[]):
        '''addes engineered feature cols to data col lists'''
        data.num_cols.extend(num_cols)
        data.cat_cols.extend(cat_cols)
        data.feature_cols.extend(num_cols + cat_cols)