


def TF_IDF(training_data, validating_data, max_features=10000, min_df=15, k=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, min_df=min_df)
    X_train = vectorizer.fit_transform(training_data['cleaned'].astype(str))
    X_val = vectorizer.transform(validating_data['cleaned'].astype(str))

    selector = SelectKBest(chi2, k=k)
    X_train = selector.fit_transform(X_train, training_data['score'])
    X_val = selector.transform(X_val)

    unique, counts = np.unique(training_data['score'].values, return_counts=True)
    class_weights = 1 / counts

    return X_train, X_val, class_weights