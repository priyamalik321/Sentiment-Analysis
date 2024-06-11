from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


class Train:
    def __init__(self, configs):
        self.config = configs
        self.filename = self.config.get('filename')
        self.features_inputs = self.config.get('features_input')
        self.target_feature = self.config.get('target_feature')
        self.text_column = self.config.get('text_column')
        self.sentiment_column = self.config.get('sentiment_column')
        self.label_encoder = LabelEncoder()  # Initialize LabelEncoder

    def data_frame(self, df, features):
        df = df[df[self.sentiment_column] != 'neutral']
        return pd.DataFrame(df[features])

    def split_df(self, df):
        x = df[self.text_column]
        y = df[self.sentiment_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        return pd.DataFrame(x_train, columns=[self.text_column]), pd.DataFrame(x_test, columns=[
            self.text_column]), y_train, y_test

    def remove_tags(self, text):
        if not isinstance(text, str):
            return text
        pattern = re.compile(r"[^a-zA-Z\s]")
        return pattern.sub(r'', text)

    def clean_text(self, data):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        cleaned_texts = []
        for text in data:
            if isinstance(text, str):
                words = text.split()
                words = [word.lower() for word in words]
                words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                cleaned_text = ' '.join(words)
                cleaned_texts.append(cleaned_text)
            else:
                cleaned_texts.append('')
        return cleaned_texts

    def create_pipeline(self):
        text_pipeline = Pipeline(steps=[
            ('count_vectorizer', CountVectorizer(ngram_range=(1, 3)))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_pipeline, self.text_column)
            ]
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        return model_pipeline

    def train_model(self):
        df = pd.read_csv(self.filename)
        df = self.data_frame(df, self.features_inputs)
        x_train, x_test, y_train, y_test = self.split_df(df)

        x_train[self.text_column] = x_train[self.text_column].apply(self.remove_tags)
        x_train[self.text_column] = self.clean_text(x_train[self.text_column])

        x_test[self.text_column] = x_test[self.text_column].apply(self.remove_tags)
        x_test[self.text_column] = self.clean_text(x_test[self.text_column])

        model_pipeline = self.create_pipeline()
        y_train = self.label_encoder.fit_transform(y_train)
        model_pipeline.fit(x_train, y_train)

        y_test = self.label_encoder.transform(y_test)
        y_pred = model_pipeline.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print("Accuracy:", score)

        return model_pipeline, x_train, x_test, y_train, y_test

    def save_test_dataset(self, x_test):
        test_data = pd.concat([x_test], axis=1)
        test_data.to_csv('test_dataset.csv', index=False)
