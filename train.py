import pandas as pd

df_train = pd.read_csv("train.csv")

df_test=pd.read_csv("test.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


df_train.rename(columns={"blue": "has bluetooth",'fc':"Front Camera px","m_dep":"m_depth","mobile_wt":"m_Weight","pc":"Primary Camera px",}, inplace=True)
df_test.rename(columns={"blue": "has bluetooth",'fc':"Front Camera px","m_dep":"m_depth","mobile_wt":"m_Weight","pc":"Primary Camera px",}, inplace=True)


df_train["screen_area"] = df_train["sc_h"] * df_train["sc_w"]
df_train.drop(columns=["sc_h", "sc_w"], inplace=True)
df_test["screen_area"] = df_test["sc_h"] * df_test["sc_w"]
df_test.drop(columns=["sc_h", "sc_w"], inplace=True)


numerical_column=['battery_power','clock_speed','int_memory',
    'm_Weight','px_height','px_width','ram',"screen_area",'talk_time']


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.Q1_ = pd.DataFrame(X).quantile(0.25)
        self.Q3_ = pd.DataFrame(X).quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        lower_bound = self.Q1_ - self.factor * self.IQR_
        upper_bound = self.Q3_ + self.factor * self.IQR_
        return X.clip(lower=lower_bound, upper=upper_bound, axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier', IQRCapper()),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_column)
    ])


x=df_train.drop("price_range",axis=1)
y=df_train["price_range"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Base models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100,random_state=42)
svm = SVC(probability=True)


#voting models

from sklearn.ensemble import VotingClassifier


voting_model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('rf', rf),
        ('svm', svm)
    ],
    voting='soft'
)


#stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifier
stacking_model = StackingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('rf', rf),
        ('svm', svm)
    ],
    final_estimator=RidgeClassifier(),
    cv=5
)


#Models to train
Models_to_train = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "SVM": svm,
    "Voting Ensemble": voting_model,
    "Stacking Ensemble": stacking_model
}


result=[]


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



for name,model in Models_to_train.items():
    pipe=Pipeline(steps=[
        ("preprocessor",preprocessor),
         ("model",model)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,average='weighted')
    recall=recall_score(y_test,y_pred,average='weighted')
    f1=f1_score(y_test,y_pred,average='weighted')


    result.append({
        "Model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1 Score":f1
    })


result_df=pd.DataFrame(result).sort_values(by="Accuracy",ascending=False)


#Best model

best_model=result_df.iloc[0]["Model"]
best_model_obj=Models_to_train[best_model]

final_pipe=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",best_model_obj)
])

final_pipe.fit(x_train,y_train)
y_final_pred=final_pipe.predict(x_test)
accuracy=accuracy_score(y_test,y_final_pred)



#

print(f"Accuracy: {accuracy}")
precision=precision_score(y_test,y_final_pred,average='weighted')
print(f"Precision: {precision}")
recall=recall_score(y_test,y_final_pred,average='weighted')
print(f"Recall: {recall}")
f1=f1_score(y_test,y_final_pred,average='weighted')
print(f"F1 Score: {f1}")


from sklearn.model_selection import cross_val_score
lr_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",lr)
])
cv_scores=cross_val_score(lr_pipeline,x_train,y_train,cv=10,scoring="accuracy")


cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")


from sklearn.model_selection import GridSearchCV


# Parameter grid
param_grid_lr = {
    "model__C": [0.01, 0.1, 1, 10, 100,500,1000],
    "model__penalty": ["l1", "l2", "elasticnet"],
    "model__solver": ["lbfgs", "saga","liblinear", "newton-cg"]
}


# Grid Search
grid_lr = GridSearchCV(
    lr_pipeline,
    param_grid=param_grid_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


grid_lr.fit(x_train,y_train)
print( grid_lr.best_score_)
print(grid_lr.best_params_)


final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=1000,
        penalty='l2',
        solver='newton-cg',
        max_iter=2000
          # suitable for multiclass
    ))
])



final_model.fit(x_train, y_train)


y_pred = final_model.predict(x_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1-score: {f1:.4f}")


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


y_pred_on_test_data_set = final_model.predict(df_test)

import pickle

filename="Mobile_price_prediction_model.pkl"
with open(filename,"wb") as f:
  pickle.dump(final_model, f)




