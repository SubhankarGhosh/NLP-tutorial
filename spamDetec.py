from sklearn.naive_bayes import MultinomialNB	#Multinomial Naive Bayes for classification
import pandas as pd
import numpy as np

table_of_data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(table_of_data)

X = table_of_data[:,:48]	#All rows and all columns except last
Y = table_of_data[:,:-1]	#Last column only which gives the output feature

Xtrain = X[:-100,]	#Training set contains all rows except last 601 entries
Ytrain = Y[:-100,]

Xtest = X[-100:,]	#Test set contains last 601 emails
Ytest = Y[-100:,]

model = MultinomialNB()	#Create the Multinomial Naive Bayes model
model.fit(Xtrain, Ytrain)
print "Accuracy score for MultinomialNB:", model.score(Xtest, Ytest)

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print "Accuracy score for AdaBoost:", model.score(Xtest, Ytest)

