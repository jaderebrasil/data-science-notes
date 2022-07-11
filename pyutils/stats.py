import pandas as pd
from termcolor import colored

def describe(df, target=None, verbose=True):
    '''
    Return a DataFrame where the rows are columns of the original
    DataFrame (df) with informations like type, count, unique, nulls and
    others.

    The columns Skewness and Kurtosis are also present, they're very useful
    to identify columns that will need aditional care, like a treatment for
    outliers.

    If a target name is provided (target), we will have a column "corr" that shows
    the correlation beetween the row variable and the target.

    If verbose is True, the function will also print other informations like
    the shape and column names.
    '''
    size = df.shape[0]
    numerical_cols = [col for col in df.columns if
                      df[col].dtype in ['int64', 'float64']]

    # Series
    types = df.dtypes.rename('types')
    counts = df.apply(lambda x: x.count()).rename('counts')
    uniques = df.apply(lambda x: [x.unique()]).transpose()[0].rename('unique')
    nulls = df.apply(lambda x: x.isnull().sum()).rename('nulls')
    distincts = df.apply(lambda x: x.unique().shape[0]).rename('distincts')
    missing_ration = ((df.isnull().sum()/ size) * 100).rename('missing_ration')
    skewness = df[numerical_cols].skew().rename('skewness')
    kurtosis = df[numerical_cols].kurt().rename('kurtosis')

    if target is None:
        result = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)
    else:
        corr = df.corr()[target].rename('corr')
        result = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)

    if verbose:
        print('___________________________')
        print('Data shape:', df.shape)
        print(colored(f"Train data columns", color = 'blue', attrs= ['dark', 'bold']))
        print(colored(df.columns, color = 'green'))
        print(colored('%d columns' % df.columns.size, color = 'red', attrs= ['dark', 'bold']))
        print(colored(df.dtypes.value_counts(), color = 'red'))
        print('___________________________')

    return result
