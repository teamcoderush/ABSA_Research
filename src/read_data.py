# Load libraries
import re
import pandas as pd

# Custom imports

# Load dataset
url = "../data/csv/ABSA16_Restaurants_Train_SB1_v2.csv" # relative dataset URL
dataset = pd.read_csv(url, encoding = 'latin1') # reads dataset with headers

train = dataset
        .groupby('text', as_index=False)['category']
        .agg({'categories':(lambda x: list(x))})

# Clean the dataset
def clean(text):
    return re.sub("[^a-zA-Z]",      # The pattern to search for
                  " ",              # The pattern to replace it with
                  train['text'].get_text())    # The text to search


# 