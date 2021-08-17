import pandas as pd
import numpy as np
import pickle
data = pd.read_csv(r"C:\Users\chand\OneDrive\Desktop\naive bayees self learn\for_Data_split__new.csv")
data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data['message'], data['sentiment'], test_size = 0.20, random_state = 42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#import model 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


MNB = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', MultinomialNB(alpha = 1))])

 #fit the model 
MNB.fit(X_train, Y_train)             
from sklearn import metrics 

#test
predicted = MNB.predict(X_test)
accuracy_score1 = metrics.accuracy_score(predicted, Y_test)
print(str('{:04.2f}'.format(accuracy_score1*100))+'%')

pickle.dump(MNB,open('model.pkl', 'wb'))



