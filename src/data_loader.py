# data_loader.py

from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_newsgroups(categories, remove_headers=True, random_state=42):
    remove = ('headers', 'footers', 'quotes') if remove_headers else ()
    
    train_data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=remove
    )
    
    test_data = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=random_state,
        remove=remove
    )
    
    df_train = pd.DataFrame({
        'text': train_data.data,
        'target': train_data.target,
        'category': [train_data.target_names[i] for i in train_data.target]
    })
    
    df_test = pd.DataFrame({
        'text': test_data.data,
        'target': test_data.target,
        'category': [test_data.target_names[i] for i in test_data.target]
    })
    
    return df_train, df_test, train_data.target_names
