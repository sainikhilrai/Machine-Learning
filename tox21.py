#importing the necessary packages
import pandas as pd
import numpy as np
from PyBioMed import Pymolecule
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

def getRepresentation(smi,feature,i):
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)
    res = mol.GetFingerprint(FPName=feature)
    return np.asarray(res[i]).reshape(1,1024)
    
def preProcess(filename,featureName):
    data = pd.read_csv(filename,delimiter=',',dtype='str')
    for i in range(len(data.formula)):
        try:
            if i==0:
                features = getRepresentation(data.formula[i],featureName,0)
                labels = np.array([int(data.label[i])])
            else:
                features = np.vstack((features,getRepresentation(data.formula[i],featureName,0)))
                labels = np.vstack((labels,np.array([int(data.label[i])])))
        except: #leave out those compunds for which ECFP6 features cannot be calculated
            continue
    return features,labels

def calculateAccuracy(model,name,features,labels):
    kf = KFold(n_splits=5,shuffle=True)
    accuracy = []
    for train_index, test_index in kf.split(features,labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if name == 'Neural Network':
            model.fit(X_train, y_train,epochs=100,batch_size=32,verbose=False)
            predictions = model.predict(X_test)
            #accuracy.append(accuracyForNN(predictions,y_test))
        else:
            model.fit(X_train,y_train)
            accuracy.append(accuracy_score(y_test,model.predict(X_test)))

    print 'Accuracy for',name,'without smoting is',round(np.mean(accuracy)*100,2)
    
    kf = KFold(n_splits=5,shuffle=True)
    sm = SMOTETomek()
    features,labels = sm.fit_sample(features,labels)
    accuracy = []
    for train_index, test_index in kf.split(features,labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if name == 'Neural Network':
            model.fit(X_train, y_train,epochs=100,batch_size=32,verbose=False)
            predictions = model.predict(X_test)
            accuracy.append(accuracyForNN(predictions,y_test))
        else:
            model.fit(X_train,y_train)
            accuracy.append(accuracy_score(y_test,model.predict(X_test)))

    print 'Accuracy for',name,'with smoting is',round(np.mean(accuracy)*100,2)
    
#assays = ['nrahr.csv','nrar.csv','nrarlbd.csv','nraromatase.csv','nrer.csv','nrerlbd.csv','nrppargamma.csv']
assays = ['nrarlbd.csv','nraromatase.csv','nrer.csv','nrerlbd.csv','nrppargamma.csv']
print '5 fold cross-validation results'
for assay in assays:
    print 'For the assay',assay
    
    print 'using ECFP4 features'
    features,labels = preProcess(assay,'ECFP4')
    calculateAccuracy(svm.SVC(kernel='linear'),'linear SVM',features,labels)
    calculateAccuracy(RandomForestClassifier(n_estimators=25),'Random forest',features,labels)
    #calculateAccuracy(defineNN(),'Neural Network',features,labels)
    
    print 'using ECFP6 features'
    features,labels = preProcess(assay,'ECFP6')
    calculateAccuracy(svm.SVC(kernel='linear'),'linear SVM',features,labels)
    calculateAccuracy(RandomForestClassifier(n_estimators=25),'Random forest',features,labels)
    #calculateAccuracy(defineNN(),'Neural Network',features,labels)