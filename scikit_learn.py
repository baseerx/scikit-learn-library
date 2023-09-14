import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class scikit:
    def __init__(self, music_file=None):
        self.music_file = music_file

    def model_function(self):
        music_data = pd.read_csv("music.csv")
        cleaned_music_data = music_data.dropna()
     
        self.music_file = cleaned_music_data

        X = cleaned_music_data.drop(columns=['genre'])
        y = cleaned_music_data['genre']
        model = DecisionTreeClassifier()
        model.fit(X, y)

        # Set feature names directly on the model
        model.feature_names_out_ = X.columns.tolist()

        predictions = model.predict([[21, 1], [22, 0]])

    def check_accuracy(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.music_file.drop(columns=['genre']),
            self.music_file['genre'],
            test_size=0.2,  # You can adjust the test size
            random_state=42  # Optional: Set a random seed for reproducibility
        )

        # Create a model and fit it
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Set feature names directly on the model
        model.feature_names_out_ = X_train.columns.tolist()

        # Make predictions
        predictions = model.predict(X_test)

        # Check the accuracy
        score = model.score(X_test, y_test)
        print(score)
