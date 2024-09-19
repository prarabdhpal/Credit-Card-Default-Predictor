import pandas as pd

credit_default = pd.read_csv('CreditDefault.csv')

credit_default.head(5)  # First we saw, what the data looks like.
credit_default.info()  # We checked for the deatils of the Data.
credit_default.describe()  # We discribed the data for the analysis.
credit_default['Default'].value_counts()  # We checked how many entries have default in the data.
credit_default.columns  # We see the coloumn headdings once-again.

x = credit_default.drop(['Default'], axis= 1)
y = credit_default['Default']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter= 1000)
model.fit(x_train,y_train)

model.intercept_
model.coef_

y_pred =model.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))