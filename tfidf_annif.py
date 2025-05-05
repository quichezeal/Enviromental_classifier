def Annif_classify(meatdata_file=None, external_file=None):
    import requests
    from importlib import metadata

    API_BASE = 'https://api.annif.org/v1/'

    from annif_client import AnnifClient

    annif = AnnifClient()

    import pandas as pd
    if meatdata_file:
        df = pd.read_csv(meatdata_file, encoding = "ISO-8859-1")
    else:
        meatdata_file = input('Please enter the metadata file name you wish to apply Annif to, including the .csv\n')
        df = pd.read_csv(meatdata_file, encoding = "ISO-8859-1")

    import re
    from nltk.corpus import stopwords

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

    df.columns.values[0] = "title"
    df['abstract'] = df['abstract'].fillna('')
    df['keyword'] = df['keyword'].fillna('')

    df['processed_text'] = df['title'] + ' ' + df['abstract']+ ' ' + df['keyword']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)
        
    import joblib
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    loaded_classifier = joblib.load("classifier_model.pkl")
    loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load('label_encoder.joblib')

    df['Annif Assigned'] = [None] * len(df)
    for i in range(0,len(df)):
        result = annif.suggest(project_id='yso-en', text=df.iloc[i]['processed_text'])
        #labels_list.append(' '.join(item['label'] for item in result))
        df.loc[i, 'Annif Assigned'] = ' '.join(item['label'] for item in result)
        if i % 10 == 0:  # Check if the iteration number is divisible by 200
            print(f"The classifier is now finished processing {i} entries from the server")
        
    if external_file:
        X = loaded_tfidf.transform(df['Annif Assigned']).toarray()
        pred_R3 = loaded_classifier.predict(X)

        df['Annif Class']=label_encoder.inverse_transform(pred_R3)
        df = df.sort_values(by='Annif Class')
    
    if 'processed_text' in df.columns:
        df.drop('processed_text', axis=1, inplace=True)
    Annif_file = input('Please enter how would you like to name the Annif result with subcategory prediction, including the .csv\n')
    df.to_csv(Annif_file, index=False)
