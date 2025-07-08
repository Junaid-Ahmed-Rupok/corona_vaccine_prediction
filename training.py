import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif


df = joblib.load('cleaned_df.pkl') # loading the dataframe
encoder = joblib.load('encoder.pkl') # loading the encoder

x = df.drop(columns=['Vaccine_Type']) # features
y = df['Vaccine_Type'] # label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # splitting the data


# feature selection

# adaboost
ad_demo_model = AdaBoostClassifier(n_estimators=200, learning_rate=0.2, random_state=42)  # defining the model
ad_demo_model.fit(x_train, y_train)
ad_feature_importance = pd.Series(ad_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
ad_features = ad_feature_importance.index.tolist()  # features for the adaboost model

# xgboost
xg_demo_model = XGBClassifier(n_estimators=200, learning_rate=0.2, random_state=42)  # defining the model
xg_demo_model.fit(x_train, y_train)
xg_feature_importance = pd.Series(xg_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
xg_features = xg_feature_importance.index.tolist()  # features for the xgboost model

# random forest
rf_demo_model = RandomForestClassifier(random_state=42)  # defining the model
rf_demo_model.fit(x_train, y_train)
rf_feature_importance = pd.Series(rf_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
rf_features = rf_feature_importance.index.tolist()  # features for the random forest model

# decision tree
dt_demo_model = DecisionTreeClassifier(random_state=42)  # defining the model
dt_demo_model.fit(x_train, y_train)
dt_feature_importance = pd.Series(dt_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
dt_features = dt_feature_importance.index.tolist()  # features of the decision tree model

# naive bayes
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(x_train, y_train)
naive_features = x_train.columns[selector.get_support()].tolist()

models = {
    'Adaboost': (AdaBoostClassifier(n_estimators=200, learning_rate=0.2, random_state=42), ad_features),
    'XGBoost': (XGBClassifier(n_estimators=200, learning_rate=0.2, random_state=42), xg_features),
    'Random Forest': (RandomForestClassifier(random_state=42), rf_features),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), dt_features),
    'Naive-Bayes': (GaussianNB(), naive_features)
}  # dictionary of the defined models

best_score, best_name, best_model, best_features = 0, None, None, None

for name, (model, features) in models.items():  # selecting the best model using cross validation score
    score = cross_val_score(model, x_train[features], y_train, cv=5).mean()  # cross-validation score
    print(f"cross validation score of {name}: {score}")

    if score > best_score:
        best_score, best_name, best_model, best_features = score, name, model, features

print(
    f"{best_name} is the best model with {best_score} cross-validation score.")  # printing the best model name and its cross-validation score

# training the best model
best_model.fit(x_train[best_features], y_train)  # trained
predictions = best_model.predict(x_test[best_features])  # predictions using the best model


def overfitting_check(model, name, features):
    train_predictions = model.predict(x_train[features])
    test_predictions = model.predict(x_test[features])

    train_accuracy = (y_train == train_predictions).mean()
    test_accuracy = (y_test == test_predictions).mean()

    if abs(train_accuracy - test_accuracy) > 0.1:
        print(f"--------> Warning: Overfitting")
    else:
        print(f"--------> No Significant Overfitting.")


print(f"Overfitting Checking of the best model: {best_name}")
overfitting_check(best_model, best_name, best_features)
print("\n" * 2)

print(f"Overfitting checking of all the models:\n\n")
for name, (model, features) in models.items():
    model.fit(x_train[features], y_train)
    print(f"{name} Overfitting Checking:")
    overfitting_check(model, name, features)

joblib.dump(best_model, 'model.pkl')  # dumping the best model

# classification report
print(classification_report(y_test, predictions))

# confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# learning curves
row, column = 0, 0
row_max, column_max = 2, 3
fig, ax = plt.subplots(row_max, column_max, figsize=(20, 20))

for name, (model, features) in models.items():
    train_sizes, training_score, testing_score = learning_curve(
        model, x_train[features], y_train, train_sizes=np.linspace(0.1, 1, 10),
        cv=5, scoring='accuracy'
    )

    if column == column_max:
        row += 1
        column = 0

    ax[row, column].set_xlabel('Training Sizes')
    ax[row, column].set_ylabel('Accuracy Scores')
    ax[row, column].set_title(f'{name} Learning Curves')
    ax[row, column].plot(train_sizes, training_score.mean(1), color='red', label='training curve')
    ax[row, column].plot(train_sizes, testing_score.mean(1), color='green', label='testing curve')
    ax[row, column].legend()
    ax[row, column].grid(True)

    column += 1

plt.tight_layout()
plt.show()

# feature importance graph
feature_importance = {
    'Adaboost': ad_feature_importance, 'XGBoost': xg_feature_importance, 'Random Forest': rf_feature_importance,
    'Decision Tree': dt_feature_importance
}

row, column = 0, 0
fig, bx = plt.subplots(row_max, column_max, figsize=(20, 20))

for name, value in feature_importance.items():
    if column == column_max:
        row += 1
        column = 0

    bx[row, column].set_xlabel('Features')
    bx[row, column].set_ylabel('Feature Importance')
    bx[row, column].set_title(f'{name} feature importance')
    bx[row, column].bar(value.index, value.values, color='blue')
    bx[row, column].grid(True)

    column += 1

plt.tight_layout()
plt.show()