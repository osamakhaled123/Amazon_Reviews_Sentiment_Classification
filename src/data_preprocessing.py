import kagglehub
arhamrumi_amazon_product_reviews_path = kagglehub.dataset_download('arhamrumi/amazon-product-reviews')

print('Data source import complete.')

import pandas as pd
import numpy as np
import spacy
import contractions
import torch
import os
import gc
import glob

if spacy.prefer_gpu():
    print("Using GPU")
else:
    print("Using CPU")

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

data = pd.read_csv('data/amazon-product-reviews/Reviews.csv')

target = data['Score']
data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Time', 'Summary'], axis=1, inplace=True)

lengths = [len(x) for x in data.Text]

def split_scores(data):
    unique, counts = np.unique(np.array(data['Score']), return_counts=True)
    minimum = counts.min() // 10

    df = data.copy().iloc[0:0]
    for val in unique:
        proportion = data[data['Score'] == val]
        proportion = proportion[:minimum]
        data = data[~data['Text'].isin(proportion['Text'])]
        df = pd.concat([df, proportion], ignore_index=True)

    return data, df

training_data, val_data = split_scores(data.copy())

def cleaning(dataset):
    """ unifying the container type
        and cleaning the text reviews
    """
    #Unifying the container type to Series
    #This enables flexibility as function can handle different container types
    if not isinstance(dataset, pd.Series):
        series = pd.Series(dataset)
    else:
        series = dataset.copy()

    #Lowercase and remove unnecessary characters from texts
    series = series.str.lower()
    series = series.str.replace('<[^>]+>', '', regex=True)
    series = series.str.replace(r'[^\w\s]', '', regex=True)
    series = series.str.replace(r'\s{2,}', '', regex=True)

    #Expaning contractions
    series = series.apply(contractions.fix)

    # Process with appropriate backend
    if spacy.prefer_gpu():
        print("Using GPU acceleration")
        doc = nlp.pipe(series, batch_size=4096, n_process=1) # Larger batches for GPU
    else:
        print("Using CPU")
        doc = nlp.pipe(series, batch_size=64, n_process=-1)
    # removing stop words and stemming
    cleaned_text = []

    for texts in spacy.util.minibatch(doc, size=2048):
        for text in texts:
            tokens = [token.lemma_ for token in text if not token.is_stop]
            cleaned_text.append(" ".join(tokens))

        torch.cuda.empty_cache()

    del series, doc
    gc.collect()

    return cleaned_text

cleaned_data = 'data'
os.makedirs(cleaned_data, exist_ok=True)
cleaned_data_directory = 'data/cleaned_data_reviews'
os.makedirs(cleaned_data_directory, exist_ok=True)
cleaned_val_directory = 'data/cleaned_val_reviews'
os.makedirs(cleaned_val_directory, exist_ok=True)

def cleaning_batches(dataset, batch_size, directory_path):
    data = dataset.copy()
    batch_size = batch_size

    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        outfile = os.path.join(directory_path, f"cleaned_data_review_{batch_start//batch_size + 1}.csv")

        if os.path.exists(outfile):
            print(f"batch {batch_start//batch_size + 1} is already processed\n")
            continue

        cleaned_batch = cleaning(data['Text'][batch_start:batch_end])
        print(f"batch {batch_start//batch_size + 1} processing completed\n")

        pd.DataFrame({'cleaned':cleaned_batch}).to_csv(outfile, index=False)

        del cleaned_batch
        gc.collect()


    files = sorted(glob.glob(directory_path+'/'+'cleaned_data_review_*.csv'))
    cleaned_reviews = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

    return cleaned_reviews


cleaned_training = cleaning_batches(training_data, 2**15, cleaned_data_directory)

cleaned_validation = cleaning_batches(val_data, 2000, cleaned_val_directory)

cleaned_training['score'] = pd.Series(training_data['Score'].values)
cleaned_validation['score'] = pd.Series(val_data['Score'].values)

cleaned_training.to_csv('data/cleaned_data_reviews/cleaned_training_reviews.csv', index=False)
cleaned_validation.to_csv('data/cleaned_val_reviews/cleaned_validating_reviews.csv', index=False)