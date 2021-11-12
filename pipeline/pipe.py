import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## variables transforming functions
def wap(df):
    total_price = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    total_size = df['bid_size1'] + df['ask_size1']
    return total_price / total_size

def log_return(seq):
    return np.log(seq).diff()

def realized_volatility(seq):
    return np.sqrt(np.sum(seq**2))

def skewness(seq):
    seq_mean = seq.mean()
    seq_median = seq.median()
    seq_std = seq.std()
    if seq_std:
        return 3*(seq_mean-seq_median)/seq_std
    else:
        return 0

def kurtosis(seq):
    seq_mean = seq.mean()
    seq_mean_diff = seq - seq_mean
    squared_4 = (seq_mean_diff**4).mean()
    squared_2 = (seq_mean_diff**2).mean()
    if squared_2:
        return squared_4 / squared_2**2 - 3
    else:
        return -3

def right_skew(seq): # only for seconds_in_bucket column.
    centered_300 = seq[seq>=300]
    centered_300_sum = (centered_300-300).sum()
    bigger_300_sum = (seq>=300).sum()
    return centered_300_sum / bigger_300_sum





## loading target train, test, sample_submission files
def load_target_data():
    root_dir = 'C:/Users/frank/data_analysis/kaggle-optiver-volatility/data/'
    train = pd.read_csv(root_dir+'train.csv')
    test = pd.read_csv(root_dir+'test.csv')
    sample_submission = pd.read_csv(root_dir+'sample_submission.csv')
    return train, test, sample_submission



## transform book, trade data into 1 merged dataframe
def generate_agg_func_dict(book_true): # if book_true==1, book_data_reform. or Else.

    if book_true:
        agg_func_dict = {
            'log_return':           [realized_volatility],
            'seconds_in_bucket':    ['count', right_skew, skewness, kurtosis],
            'wap':                  [skewness, kurtosis],
            'bid_size1':            [skewness, 'max', 'mean'],
            'ask_size1':            [skewness, 'max', 'mean'],
            'bid_size2':            [skewness],
            'ask_size2':            [skewness]
        }

    else:
        agg_func_dict = {
            'size':         ['max','sum'],
            'order_count':  ['max']
        }

    return agg_func_dict
        


def generate_filename(book_true, stock_id):

    root_dir = 'C:/Users/frank/data_analysis/kaggle-optiver-volatility/data/'
    if book_true:
        filename = root_dir+f'book_train.parquet/stock_id={stock_id}'

    else:
        filename = root_dir+f'trade_train.parquet/stock_id={stock_id}'

    return filename



def generate_tmp_data(book_true, filename):

    if book_true:
        tmp = pd.read_parquet(filename)
        tmp['wap'] = wap(tmp)
        tmp['log_return'] = tmp.groupby(['time_id'])['wap'].apply(log_return)
    
    else :
        tmp = pd.read_parquet(filename)
        tmp['log_return'] = tmp.groupby(['time_id'])['price'].apply(log_return)
    
    return tmp



def generate_cols(book_true):

    if book_true:
        cols = ['realized_volatility',
                'seconds_count',
                'seconds_right_skew',
                'seconds_skewness',
                'seconds_kurtosis',
                'wap_skewness',
                'wap_kurtosis',
                'bid_size1_skewness',
                'bid_size1_max',
                'bid_size1_mean',
                'ask_size1_skewness',
                'ask_size1_max',
                'ask_size1_mean',
                'bid_size2_skewness',
                'ask_size2_skewness',
                'row_id']

    else :
        cols = ['size_max',
                'size_sum',
                'order_count_max',
                'row_id']
    
    return cols



# reform data function
def reform_data(stock_id, book_true):

    agg_func_dict = generate_agg_func_dict(book_true)
    filename = generate_filename(book_true, stock_id)
    tmp = generate_tmp_data(book_true, filename)
    cols = generate_cols(book_true)

    res = tmp.groupby(['time_id']).agg(agg_func_dict)
    res.reset_index(inplace = True)
    res['row_id'] = res['time_id'].apply(lambda x: str(f'{stock_id}-{x}'))
    res.drop(['time_id'], axis = 1, inplace = True)
    res.columns = cols

    return res



## generating new dataframes with derived variables
def regenerate_data(stock_id_list):

    # generate empty dataframe. 
    frame = pd.DataFrame(dtype = 'float64')

    # Generating new dataframe with derived features
    for stock_id in stock_id_list:
        print("current stock : "+str(stock_id))
    
        # 1. book data
        book_transformed = reform_data(stock_id, 1)

        # 2. trade data
        trade_transformed = reform_data(stock_id, 0)

        transformed = pd.merge(book_transformed, trade_transformed, how='outer', on='row_id')
        frame = pd.concat([frame, transformed], ignore_index = True)

    root_dir = 'C:/Users/frank/data_analysis/kaggle-optiver-volatility/data/'
    frame.to_csv(root_dir+'frame.csv')


if __name__ == '__main__':
    train, _, _ = load_target_data()
    stock_id_list = train.stock_id.value_counts().index
    stock_id_list = sorted(list(stock_id_list))
    regenerate_data(stock_id_list)



## Joblib..