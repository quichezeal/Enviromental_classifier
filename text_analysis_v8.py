def R3_classify(R3_file,output_self):
    import pandas as pd
    

    df = pd.read_csv(R3_file, encoding = "ISO-8859-1")

    #######Text Processing
    import re
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    #nltk.download('stopwords')
    user_stopword=open("sys_stopwords.txt","r")
    stopword_data=user_stopword.read()
    user_stopword_list = stopword_data.split(" ") 
    user_stopword.close()
    stop_words = set(stopwords.words('english')+user_stopword_list)

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text

    df['abstract'] = df['abstract'].fillna('')
    df['keywords'] = df['keyword'].fillna('')
    df.columns.values[0] = "title"
    df['processed_text'] = df['title'] + ' ' + df['abstract'] + ' ' + df['keywords']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    #########Feature Extraction
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text']).toarray()

    #########Subcategory Creation
    def map_text_to_subcategories(text):
        text = text.lower()
        if 'climate' in text or 'flood' in text or 'storm' in text or 'coastal' in text or 'weather' in text or 'emissions' in text:
            return 'climate and environmental change'
        elif 'species' in text or 'native' in text or 'invasive' in text or 'exotic' in text or 'plant' in text or 'diversity' in text:
            return 'biodiversity and ecosystems'
        elif 'urban' in text or 'housing' in text or 'city' in text or 'bridges' in text or 'infrastructure' in text or 'transport' in text:
            return 'urbanization and infrastructure'
        elif 'health' in text or 'asthma' in text or 'cancer' in text or 'exposure' in text or 'pollution' in text or 'disease' in text:
            return 'health and environmental risks'
        elif 'energy' in text or 'gas' in text or 'oil' in text or 'biochar' in text or 'renewable' in text or 'power' in text or 'carbon' in text:
            return 'energy and sustainability'
        elif 'water' in text or 'soil' in text or 'sediment' in text or 'erosion' in text or 'contamination' in text or 'river' in text:
            return 'water and soil systems'
        elif 'social' in text or 'policy' in text or 'political' in text or 'economic' in text or 'cultural' in text or 'states' in text:
            return 'social and political dimensions'
        elif 'houston' in text or 'texas' in text or 'american' in text or 'local' in text or 'regional' in text:
            return 'houston and regional studies'
        else:
            return 'unknown'

    df['subcategories'] = df['processed_text'].apply(map_text_to_subcategories)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['subcategories'])

    #########Train/Test Split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #########Train Classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score


    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)

    import joblib

    # Save model and vectorizer
    joblib.dump(classifier, "classifier_simple.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer_simple.pkl")
    joblib.dump(label_encoder, 'label_encoder_simple.joblib')

    train_title = []
    train_abstract = []
    train_keyword = []
    train_labels = []
    train_preds = []
    train_uri = []
    train_id = []
    train_collection = []

    for i in range(len(df['processed_text'])):
        new_text = [df['processed_text'][i]]
        new_text_vectorized = tfidf.transform(new_text)
        pred=label_encoder.inverse_transform(classifier.predict(new_text_vectorized))
     
        train_title.extend([df['title'][i]])
        train_abstract.extend([df['abstract'][i]])
        train_keyword.extend([df['keywords'][i]])
        train_labels.extend([df['subcategories'][i]])
        train_preds.extend(pred)
        train_uri.extend([df['uri'][i]])
        train_id.extend([df['id'][i]])
        train_collection.extend([df['collection'][i]])

    results_df = pd.DataFrame({
        'title': train_title,
        'abstract': train_abstract,
        'keyword': train_keyword,
        'actual label': train_labels,
        'predicted label': train_preds,
        'uri': train_uri,
        'id': train_id,
        'collection': train_collection
    })

    df_sorted = results_df.sort_values(by="predicted label")
    
    df_sorted.to_csv(output_self, index=False)
    return(output_self)


############Training on an external source of data
def External_classify(external_file,output_self):
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    
    df = pd.read_csv(external_file, encoding = "ISO-8859-1")
    df['title'] = df['Item Title']
    df['abstract'] = df['Abstract'].fillna('')

    user_stopword=open("sys_stopwords.txt","r")
    stopword_data=user_stopword.read()
    user_stopword_list = stopword_data.split(" ") 
    user_stopword.close()
    stop_words = set(stopwords.words('english')+user_stopword_list)

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text

    df['processed_text'] = df['title'] + ' ' + df['abstract']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    #########Feature Extraction
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text']).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Subcategories'])

    #########Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #########Train Classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(classifier, "classifier_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, 'label_encoder.joblib')

    # Load and use
    loaded_classifier = joblib.load("classifier_model.pkl")
    loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")

    df = pd.read_csv(output_self, encoding = "ISO-8859-1")

    df['title'] = df['title']
    df['abstract'] = df['abstract'].fillna('')
    df['keywords'] = df['keyword'].fillna('')

    df['processed_text'] = df['title'] + ' ' + df['abstract'] + ' ' + df['keywords']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    X = loaded_tfidf.transform(df['processed_text']).toarray()

    import numpy as np

    probabilities = loaded_classifier.predict_proba(X)

    # Get the maximum probability and the corresponding class
    max_prob = np.max(probabilities, axis=1)
    predicted_classes = loaded_classifier.classes_[np.argmax(probabilities, axis=1)]

    # Set a threshold for "unknown"
    threshold = 0.2
    final_predictions_prob = np.where(max_prob >= threshold, predicted_classes, "unknown")

    df['Predicted Class'] = [None] * len(df)

    for i in range(len(df['processed_text'])):
        if final_predictions_prob[i] != "unknown":
            df.loc[i, 'Predicted Class'] = label_encoder.inverse_transform([int(final_predictions_prob[i])])
        else:
            df.loc[i, 'Predicted Class'] = final_predictions_prob[i]
            
    df=df[['title', 'abstract', 'keyword', 'actual label', 'predicted label', 'Predicted Class',
           'uri', 'id', 'collection']]
    output_external = input('Please enter how would you like to name the result trained on external data, including the .csv\n')
    df.to_csv(output_external, index=False)
    filtered_df = df[
        (df['predicted label'] == df['Predicted Class']) &
        (df['predicted label'] == df['actual label'])
    ]
    filtered_df = filtered_df[filtered_df['Predicted Class'] != "unknown"]
    filtered_df = filtered_df.sort_values(by="Predicted Class")
    output_external_cleaned = input('Please enter how would you like to name the cleaned-up result trained on external data, including the .csv\n')
    filtered_df.to_csv(output_external_cleaned, index=False)
    return output_external_cleaned


def External_only(external_file):
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    
    df = pd.read_csv(external_file, encoding = "ISO-8859-1")
    df['title'] = df['Item Title']
    df['abstract'] = df['Abstract'].fillna('')

    user_stopword=open("sys_stopwords.txt","r")
    stopword_data=user_stopword.read()
    user_stopword_list = stopword_data.split(" ") 
    user_stopword.close()
    stop_words = set(stopwords.words('english')+user_stopword_list)

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text

    df['processed_text'] = df['title'] + ' ' + df['abstract']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    #########Feature Extraction
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text']).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Subcategories'])

    #########Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #########Train Classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(classifier, "classifier_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, 'label_encoder.joblib')