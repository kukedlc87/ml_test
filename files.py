import numpy as np
import pandas as pd

data = pd.read_csv('Loan_default.csv')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
obj_col = ['HasCoSigner', 'LoanPurpose', 'HasDependents', 'HasMortgage', 'MaritalStatus', 'EmploymentType', 'Education']
for col in obj_col:
    data[col] = le.fit_transform(data[col])

data = data.drop(['LoanID'], axis=1)

X = data.drop(['Default'], axis=1)
y = data['Default']



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.25, random_state=42)
import joblib
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
score2 = accuracy_score(y_pred, y_test)
print(score2)


joblib.dump(model2, 'modelo2.pkl')
