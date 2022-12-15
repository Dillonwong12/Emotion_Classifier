import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

# Emotions to be ignored (limited examples)
IGNORE = ['threat', 'sarcasm', 'commitment']

K_FOLDS = 5

def k_foldCV(X, y, classifiers_dict):
    '''
    Performs K-fold CV to compare different models. SVM -> LR -> RF -> XGB
    '''

    # Finding the best hyperparameters for SVM
    print("SVM tuning...")
    SVM_params = {
        'C': [1],
        'gamma': [0.0],
        'kernel': ['linear']
    }

    SVM_grid = GridSearchCV(
        estimator=SVC(decision_function_shape='ovr'),
        param_grid=SVM_params,
        cv=K_FOLDS,
        refit=True,
        n_jobs=-1,
        verbose=1
    )

    SVM_grid.fit(X, y)
    print(SVM_grid.best_params_)
    print(SVM_grid.best_estimator_)

    # Finding the best hyperparameters for LR
    print("Logistic Regression tuning...")
    LR_params = {
        'solver': ['sag'],
        'C': [1]
    }

    LR_grid = GridSearchCV(
        estimator=LogisticRegression(multi_class='ovr'),
        param_grid=LR_params,
        cv=K_FOLDS,
        refit=True,
        n_jobs=-1,
        verbose=1
    )

    LR_grid.fit(X, y)
    print(LR_grid.best_params_)
    print(LR_grid.best_estimator_)

    # Finding the best hyperparameters for RF
    print("Random Forest tuning...")
    RF_params = {
        'n_estimators': [200],
        'max_depth': [20], 
        'class_weight': ['balanced']
    }

    RF_grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=RF_params,
        cv=K_FOLDS,
        refit=True,
        n_jobs=-1,
        verbose=1
    )

    RF_grid.fit(X, y)
    print(RF_grid.best_params_)
    print(RF_grid.best_estimator_)

    
    # Finding the best hyperparameters for XGB
    print("XGBoost tuning...")
    XGB_params = {
        "learning_rate": [0.01],
        "max_depth": [20],
        "gamma": [0.0]}
    
    XGB_grid = GridSearchCV(
        estimator=XGBClassifier(),
        param_grid=XGB_params,
        cv=K_FOLDS,
        refit=True,
        n_jobs=-1,
        verbose=1
    )

    XGB_grid.fit(X, y)
    print(XGB_grid.best_params_)
    print(XGB_grid.best_estimator_)

    classifiers_dict['SVM'] = SVM_grid
    classifiers_dict['LR'] = LR_grid
    classifiers_dict['RF'] = RF_grid
    classifiers_dict['XGB'] = XGB_grid
    

def get_classification_report(classifier_name, y_test, y_pred):
    '''
    Outputs a classification report for each classifier.
    '''

    print('\n', classifier_name, ' Classification Report')
    print('======================================================')
    print('\n', classification_report(y_test, y_pred))


data = pd.read_excel("emotion_classification_eng_2.xlsx")
print("Before removing ignored emotions: \n", data['emotion'].unique())

'''
fig, ax = plt.subplots()
data['emotion'].value_counts().plot(ax=ax, kind='bar', xlabel='emotions', ylabel='frequency')

plt.show()'''

data = data[data.emotion.isin(IGNORE) == False]
print("After removing ignored emotions: \n", data['emotion'].unique())

# Encoding emotions 
emotion_dict = {}
for idx, label in enumerate(data['emotion'].unique()):
    emotion_dict[label] = idx

print(emotion_dict.items())

data['emotion'] = data['emotion'].apply(lambda x: emotion_dict[x])
print(data.head())

# Split into X and y
X = data.iloc[:, 0].values
print(X[0])
y = data.iloc[:, 1].values
print(y[0])

# Build a TF-IDF matrix from the corpus of texts
tfidf = TfidfVectorizer(max_features=4000)
X = tfidf.fit_transform(X).toarray()

with open("emotion_tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

classifiers_dict = {}
predictions = {}

# K-Fold Cross Validation for model evaluation
k_foldCV(X_train, y_train, classifiers_dict)

# print classification reports
for key, value in classifiers_dict.items():
    get_classification_report(key, y_test, value.predict(X_test))
    print(key, ": ", value.predict(tfidf.transform(["fuck off", ]).reshape(1, -1).toarray()))

# save the trained models
with open("emotion_svm_model.pkl", "wb") as file:
    pickle.dump(classifiers_dict['SVM'], file)

with open("emotion_lr_model.pkl", "wb") as file:
    pickle.dump(classifiers_dict['LR'], file)

with open("emotion_rf_model.pkl", "wb") as file:
    pickle.dump(classifiers_dict['RF'], file)

with open("emotion_xgb_model.pkl", "wb") as file:
    pickle.dump(classifiers_dict['XGB'], file)