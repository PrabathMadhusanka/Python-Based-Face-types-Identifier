import pickle

data=pickle.load(open('data.pickle','rb'))
target=pickle.load(open('labels.pickle','rb'))

from sklearn.neighbors import KNeighborsClassifier

clsfr=KNeighborsClassifier()
clsfr.fit(data,target)

from sklearn.externals import joblib

joblib.dump(clsfr,'Model.sav')
