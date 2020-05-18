import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def preprocess_data (train_features, train_salaries):
    train_df = pd.merge(train_features, train_salaries,on = 'jobId')
    train_df = train_df[train_df.salary > 8.5]
    return train_df



def plot (df,col):
    plt.figure(figsize = (14,6))
    plt.subplot(1,2,1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        mean = df.groupby(col)['salary'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xlabel(col)
    plt.ylabel('counts')
    plt.xticks(rotation = 45)
    
    plt.subplot(1,2,2)
    if df[col].dtype == "int64" or  col == 'companyId':
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(mean.index)), mean.values + std.values, mean.values - std.values, alpha = 0.1)
    else:
        sns.boxplot(x = col, y = 'salary', data = df)
    plt.xticks(rotation = 45)
    plt.ylabel('salaries')

def label_encode(train_df):     ## Convert to categorical data and label encode the data
    for col in train_df.columns:
        if train_df[col].dtype == "O" and col != 'jobId':
             train_df[col] = train_df[col].astype('category')
             encode_label(train_df, col)
    return train_df 
    
def encode_label(df, col):    #encode the categories using average salary for each category to replace label
    cat_dict ={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat]['salary'].mean()   
    df[col] = df[col].map(cat_dict)


def plot_heatmap(train_df):   ##Plotting heatmap using transformed variables
    fig = plt.figure(figsize = (12,10))
    features = ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
    sns.heatmap(train_df[features + ['salary']].corr(), cmap='Blues', annot=True)
    plt.xticks(rotation = 45)
    plt.show()