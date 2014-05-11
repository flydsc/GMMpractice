'''
http://scikit-learn.org/0.5/modules/gmm.html
@author: Local-Admin
'''
import numpy as np
from sklearn import cross_validation
from sklearn import mixture

def read_data():#loading the data from iris dataset
    classset = []
    f =open("bezdekIris.data.txt", 'r')
    lines=[line.strip() for line in f.readlines()]  
    f.close()
    lines=[line.split(",") for line in lines if line]
    #There are three classes which means we have to separate all the dataset into three parts
    class1=np.array([line[:4] for line in lines if line[-1]=="Iris-setosa"], dtype=np.float)
    
    class2=np.array([line[:4] for line in lines if line[-1]=="Iris-versicolor"], dtype=np.float)
      
    class3=np.array([line[:4] for line in lines if line[-1]=="Iris-virginica"], dtype=np.float)
    #restore the data without class label
    data = np.array([line[:4] for line in lines], dtype=np.float)
    label = [line[4] for line in lines]
    #data = []
    #data += [line for line in lines]
    classset += [class1]
    classset += [class2]
    classset += [class3]

    return classset,data,label#return data in different classes and the whole data without class label

def guerror(train,test,tlabel):
    g = mixture.GMM(n_components=3)
    g.fit(list(train[:4]))
    error=[]
    for i in range(len(test)):
        num = int(g.predict([list(test[i][:4])])[0])
        label = classlabelfun(tlabel[i])
        print num,label
        if num == label:
            error.append(0)
        else: 
            error.append(1)
    print '==================================================='
    return error

def classlabelfun(label):
    if label == "Iris-setosa":
        classlabel = 1
    elif label == 'Iris-versicolor':
        classlabel = 0
    elif label == 'Iris-virginica':
        classlabel = 2
    return classlabel

def crossvali(data,label):
    cv = cross_validation.KFold(len(data), n_folds= 10, indices=True)#the corss validation from library
    results = []  
    for traincv, testcv in cv: #get the train and test data
        train =[]
        test = []
        tlabel =[]
        for i in range(len(traincv)):
            train.append(data[traincv[i]])#train data
        for i in range(len(testcv)):
            test.append(data[testcv[i]])# test data
            tlabel.append(label[testcv[i]])
        error = guerror(train,test,tlabel)#run the knn
        results.append(1-float(sum(error))/float(len(error)))#get the total error
    print results
            

if __name__ == '__main__':
    classset,data,label = read_data()
    classg=[]
    for i in range(len(classset)):
        g = mixture.GMM(n_components=1)
        g.fit(classset[i][:,:4])
        classg.append(g)  
    #print [classg[i].means_ for i in range(len(classg))]
    #print [classg[i].covars_ for i in range(len(classg))]
    g = mixture.GMM(n_components=3)
    g.fit(data[:,:4])
    #print g.means_
    #print g.covars_
    #print classset[0][0,:4]
    #print classset[0][20,-1]
    #for i in range(len(label)):
    #    print label[i],g.predict([list(data[i])])
    crossvali(data,label)

    