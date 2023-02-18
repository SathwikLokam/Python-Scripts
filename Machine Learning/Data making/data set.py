import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# importing datasets
data_set = pd.read_csv('Dataset.csv')

# Extracting independent variable
X = data_set.iloc[:, :-1].values

# Extracting dependent variable
y = data_set.iloc[:, -1].values

# Handling missing data (Replacing missing data with the mean value)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
print(X,"\n")
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X,"\n")

# Encoding categorical x
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
######print(X_train,"\n",X_test)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train,"\n",X_test)

print(X_train[:,2],X_train[:,3])
(plt.plot(X_train[:,2],X_train[:,3]))
plt.show()