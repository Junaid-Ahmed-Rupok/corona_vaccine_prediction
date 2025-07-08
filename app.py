import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import OrdinalEncoder

model = joblib.load('model.pkl')  # loading the best model
encoder = joblib.load('encoder.pkl')  # loading the encoder

model_features = model.feature_names_in_.tolist()  # features of the best model
model_cats = [feature for feature in model_features if
              feature in encoder.feature_names_in_]  # categorical features of the model

categorical_class = {
    encoder.feature_names_in_[idx]: encoder.categories_[idx].tolist()
    for idx in range(len(encoder.feature_names_in_))
}
feature_info = {
    feature: categorical_class[feature] if feature in model_cats else 'number'
    for feature in model_features
}

categories = [feature_info[feature] for feature in model_cats]  # categories of the model_cats


def create_modified_encoder(encoder):
    if not model_cats:
        return None
    modified_encoder = OrdinalEncoder(categories=categories)

    dummy_values = [[categories[i][0] for i in range(len(model_cats))]]  # dummy values creation
    modified_encoder.fit(dummy_values)  # training the modified encoder
    modified_encoder.feature_names_in_ = np.array(model_cats)  # feature names of the modified encoder

    return modified_encoder


modified_encoder = create_modified_encoder(encoder)  # modified encoder creation

app = Flask(__name__)  # defining the flask


@app.route('/')
def home():
    return render_template('index.html', features=feature_info)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()  # taking user inputs as dictionary
    df = pd.DataFrame([data])  # converting the dictionary into a dataframe

    # Convert numeric columns from strings to floats
    for col in df.columns:
        if col in ['Age', 'Comorbidity_Count']:  # Skip categorical column
            df[col] = pd.to_numeric(df[col])

    if modified_encoder != None:
        df[model_cats] = modified_encoder.transform(df[model_cats])  # encoding the categorical inputs

    predictions = model.predict(df)[0]  # prediction using the best model

    # ['AstraZeneca' 'Sinopharm' 'Sinovac' 'Pfizer' 'Moderna']
    #     0 if x == 'AstraZeneca' else 1 if x == 'Sinopharm' else 2 if x == 'Sinovac'
    #         else 3 if x == 'Pfizer' else 4

    x = 'AstraZeneca' if predictions == 0 else 'Sinopharm' if predictions == 1 else 'Sinovac' if predictions == 2 else 'Pfizer' if predictions == 3 else 'Moderna'

    return render_template('index.html', features=feature_info, predictions=x)


if __name__ == "__main__":
    app.run(debug=True)  # Force IPv4