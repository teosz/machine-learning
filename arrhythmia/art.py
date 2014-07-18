import csv
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Read from the CSV file
csvf = open("arrhythmia.data")
csvfile = csv.reader(csvf)
 
# The data has missing values noted by '?' (ignore this part)
# In this part of code we just replace them with np.nan
X = np.array(np.zeros(shape=(1, 280)))
for row in csvfile:
    for idx, item in enumerate(row):
        if item == '?':
            row[idx] = np.nan
        row[idx] = np.float32(row[idx])
    row = np.array(row)
    row = row.reshape(1, X.shape[1])
    X = np.append(X, row, axis=0)
X = np.delete(X, (0), axis=0)
rfc = RandomForestClassifier()
gnb = GaussianNB()
kfold = KFold(n=len(X), n_folds=10, indices=None, shuffle=False, random_state=None)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
nrm = Normalizer(norm='l1', copy=1)

# X is your data vector: with rows as samples and last column being the labels

# TODO: use scikit-learn Imputer class to use mean values for missing fields

X = imp.fit_transform(X) 
# TODO: extract labels from X into a new variable Y
Y= X[:, -1]
X= X[:, 0:-1]

# TODO: normalize using l1 norm (or other, checkout what works best!)
X = nrm.fit_transform(X);
# TODO: Use scikit-learn KFold class to create 10 folds
t = 0
p = 0
for tri, tei in kfold:
  X_train = X[tri]
  X_test  = X[tei]
  Y_train = Y[tri]
  Y_test  = Y[tei]
  gnb.fit(X_train,Y_train)
  rfc.fit(X_train,Y_train)
  t +=  gnb.score(X_test, Y_test)
  p +=  rfc.score(X_test, Y_test)

t = t/10
p = p/10
print "Gaussian Naive Bayes : " ,  (t * 100) ,"%"
print "Randomi Forest Classifier : " , (p * 100), "%"
print "RFC bate cu : ", ( p - t ) * 100 , "%"


# TODO: run GaussinNB classifier to fit and predict for each
 
# TODO: print the averaged error rate across all the runs
 

# TODO: Bonus, use RandomForestClassifier and checkout the error rate!
