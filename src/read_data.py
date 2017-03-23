# Load libraries
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords # Import the stop word
from sklearn.feature_extraction.text import CountVectorizer
# Custom imports
import models
# Transforms review text to words
def review_to_words( text ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(text, "lxml").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

if __name__ == "__main__":
    # Load dataset
    url = "../data/csv/ABSA16_Restaurants_Train_SB1_v2.csv" # relative dataset URL
    dataset = pd.read_csv(url, encoding = 'latin1') # reads dataset with headers
    #
    train = dataset.groupby('text', as_index=False)['category'].agg({'categories': (lambda x: list(x))})
    #
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["text"].size
    #
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    #
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list 
    print("Cleaning and parsing the training set review sentences...")
    for i in range( 0, num_reviews ):
        if( (i + 1) % 250 == 0 ):
            print("\tReview sentence %d of %d" % ( i + 1, num_reviews ))
        # Call our function for each one, and add the result to the list of 
        # clean reviews
        clean_train_reviews.append( review_to_words( train["text"][i] ) )
    #
    print("\nCreating the bag of words...")
    #
    # Initialize the "CountVectorizer"  
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = 5000)
    # Scikit-learn's bag of words tool
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    #
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    # Define Categories
    categories = []
    categories.append(('restaurantGeneral', 'RESTAURANT#GENERAL'))
    categories.append(('restaurantPrice', 'RESTAURANT#PRICES'))
    categories.append(('restaurantMiscellaneous', 'RESTAURANT#MISCELLANEOUS'))
    
    categories.append(('foodPrice', 'FOOD#PRICES'))
    categories.append(('foodQuality', 'FOOD#QUALITY'))
    categories.append(('foodStyleOptions', 'FOOD#STYLE_OPTIONS'))
    
    categories.append(('drinksPrice', 'DRINKS#PRICES'))
    categories.append(('drinksQuality', 'DRINKS#QUALITY'))
    categories.append(('drinksStyleOptions', 'DRINKS#STYLE_OPTIONS'))
    
    categories.append(('ambienceGeneral', 'AMBIENCE#GENERAL'))
    categories.append(('serviceGeneral', 'SERVICE#GENERAL'))
    categories.append(('locationGeneral', 'LOCATION#GENERAL'))
    
    for name, category in categories:
        #
        # For each, print the vocabulary word and the number of times it 
        # appears in the training set
        # for tag, count in zip(vocab, dist):
        #   print(count, tag)
        #
        train[name] = train['categories'].apply(lambda x: category in x)
        #
        print("Training the SVM...")
        # Initialize a Random Forest classifier with 100 trees
        # forest = RandomForestClassifier(n_estimators = 100)
        # Fit the forest to the training set, using the bag of words as 
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        # forest = forest.fit( train_data_features, train["resturentGeneral"] )
        features = train_data_features
        labels = train[name]
        models.cross_val(features, labels);