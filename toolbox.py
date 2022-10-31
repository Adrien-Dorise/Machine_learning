from os import path,makedirs
import scipy.io as spio
import numpy as np 
import pandas as pd
import math
from csv import writer

#class MatplotlibMissingError(RuntimeError):
#    pass
#try:
#    import matplotlib.pyplot as plt
#    except ImportError:
#        raise MatplotlibMissingError('This feature requires the optional dependency matpotlib, you can install it using `pip install matplotlib`.')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Stats
import scipy.stats as stat


#Data split and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Performance analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

#Decision/Random tree classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Knn
from sklearn.neighbors import KNeighborsClassifier

#Naïve Bayes
from sklearn.naive_bayes import GaussianNB

#SVM
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM

#Elliptic envelope
from sklearn.covariance import EllipticEnvelope

#Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

#Isolation Forest
from sklearn.ensemble import IsolationForest

#Kmeans
from sklearn.cluster import KMeans

#Hierarchical Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Time series rupture package
# If package not installed, type in you IDE console 'pip install ruptures --user'
import ruptures as rpt

#Parallel coordinates
from pandas.plotting import parallel_coordinates
 
#Saving/loading models
from joblib import dump,load

#Wavelet
import pywt





def classifier(data,classes,classifierChoice='Knn',RocDefectValue=1,ensemble_estimators=25,tree_criterion='entropy',knn_n_neighbors=5,random_state=None,svmKernel='rbf',svmLinearDegree=3,save=0,splitSize=0.20,plot=0,xlabel='X',ylabel='Y',classesName='',folderPath='',figName='Classification',randomColors = 0):
    '''
    Uses a classification methods to predict classes among given data set. The data set is splitted into a train set and a test set.
    Implemented methods are 'K-nn', 'SVM', 'Decision tree', 'Random forest', 'Naïve Bayes'.

    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes : Serie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classifierChoice : string, optional
        Classification method selected. Choices are 'Knn', 'svm', 'decision_tree_classifier, 'random_forest_classifier' and 'naive_bayes'. The default is 'Knn'.
    RocDefectValue : int, optional
        Positive label used to plot the ROC curve. Roc evaluation only works when one label is known as a defect, otherwise,the ROC curve evaluation will be pointless.
    ensemble_estimators : int, optional
        Number of estimators(ex: trees in random forest) used for ensemble algorithms. The default is 25.
    tree_criterion : string, optional
        The function to measure the quality of a split (used in Decision tree/Random forest classifier). Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. The default is 'entropy'.
    knn_n_neighbors : int, optional
        Number of neighbors used in K-nn algorithm. The default is 5.
    random_state : int, optional
        Determines random number generation. Use an int to make the randomness deterministic. The default is None.
    splitSize : int, optional
        Determines the size of test and train sets. The value given indiquates the percentage of data from the data set used for the test set. The default is 0.25.
    classesName : list of strings, optional
        Names of the different classes used for plotting. Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), otherwise the name will not match the classes leading to false diagnostic!!! 
        Needed if plot=1. The default is ''.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the classifier will always be added to it. The default is'Classification'
 The Default is '' the name of the classifier will always be added to it. Example: 'scaled_datas'
    randomColors : int,optional
        Put 1 if you want matplotlib to choose the colors. Otherwise, the colors will be chosen using the function 'ClassificationColors' to keep the same colors each time. The default is 0.

    Returns
    -------
    object : Algo
        Classification algorithm trained by the data set.
    ROC : list
        Datas of the ROC curve (Can be use to plot multiple ROC curve on one figure).
    confusionMatrix : nparrray of dim (2,2)
        Confusion Matrix of the classifier results.
    accuracy : float
        Accuracy of the classifier regarding the test set.

    '''
    print('\nModel creation of '+classifierChoice+' in progress ...')
    # features contains the features allowing us to diagnostic the datas in a category
    features = pd.DataFrame(data);
    # Splitting dataset into training set and test set
    features_train,features_test,classes_train,classes_test = train_test_split(features,classes, test_size=splitSize,random_state=0);
    features_train = features_train.to_numpy()
    features_test = features_test.to_numpy()
    # Fitting classifier to the training set       
    if classifierChoice == 'Knn':   
        classifier = KNeighborsClassifier(n_neighbors = knn_n_neighbors, metric = 'minkowski', p=2)
    elif classifierChoice == 'svm':
        classifier = SVC(random_state = random_state, kernel=svmKernel,degree=svmLinearDegree)
    elif classifierChoice == 'decision_tree_classifier':
        classifier = DecisionTreeClassifier(criterion = tree_criterion, random_state = random_state)
    elif classifierChoice == 'random_forest_classifier':
        classifier = RandomForestClassifier(n_estimators=ensemble_estimators,criterion=tree_criterion,random_state=random_state)
    elif classifierChoice == 'naive_bayes':
        classifier = GaussianNB()    
    else:
        print('Error in classifier: classifierChoice unknown')
    classifier.fit(features_train,classes_train) #Training
    y_pred = classifier.predict(features_test) #Testing
    # Confusion matrix
    cm = confusion_matrix(classes_test,y_pred)
    if len(cm) == 1:
        print('WARNING IN CLASSIFIER: Only one class found in "classes" parameter. Impossible to calculate the confusion matrix accuracy')
        cmAccuracy = 1;
    else:
        cmAccuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
    
    #ROC curve
    roc_fpr, roc_tpr, _ = roc_curve(classes_test, y_pred,pos_label=RocDefectValue);

    if plot == 1:
        if len(classesName) == 0: #If no classes names are give, we cannot continue
            print('WARNING IN CLASSIFIER: No classes names given. Impossible to plot the result')
        else:
            if randomColors != 1:
                color = classificationColors(classes_test)
            else:
                color = ['']    
            figName = figName + '_' + classifierChoice
            plotClassification(features_train, classes_train,classifier ,classesName=classesName,randomColors=randomColors,colors=color,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(training_set)'),save=save,folderPath=folderPath)
            plotClassification(features_test, classes_test,classifier,classesName=classesName,randomColors=randomColors,colors=color,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(test_set)'),save=save,folderPath=folderPath)    
            plotROC(roc_fpr,roc_tpr,np.array([classifierChoice]),save=save,folderPath=folderPath,figName=(figName+'_ROC_curve'))
    print(classifierChoice+' model done !\n')
    return classifier,[roc_fpr,roc_tpr],cm, cmAccuracy






def oneClassClassification(data,classifierChoice='OCSVM',svmKernel='rbf',svmNu=0.01,ECsupportFraction=0.9,lofNeighbours=10,ifestimators=50,random_state=None,withPoints=0,save=0,plot=0,xlabel='X',ylabel='Y',folderPath='',figName='OCC'):
    '''
    Uses a one class classification methods to predict normal classes among given data set. The data set is splitted into a train set and a test set.
    Implemented methods are 'elliptic classification', 'OCSVM', 'LOF', 'isolation forest', 'Auto encoder'.

    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2)
    classifierChoice : string, optional
        Classification method selected. Choices are 'Knn', 'svm', 'decision_tree_classifier, 'random_forest_classifier' and 'naive_bayes'. The default is 'Knn'..
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 0
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the classifier will always be added to it. The default is'Classification'
        The Default is '' the name of the classifier will always be added to it. Example: 'scaled_datas'

    Returns
    -------
    object : Algo
        Classification algorithm trained by the data set.


    '''
    print('\n'+classifierChoice+' in progress ...')
    # features contains the features allowing us to diagnostic the datas in a category
    features_train = np.array(data)
    classes_train = np.zeros(len(features_train))
    # Splitting dataset into training set and test set
    # Fitting classifier to the training set       
    if classifierChoice == 'OCSVM':   
         classifier = OneClassSVM(kernel=svmKernel,gamma='scale',nu=svmNu)
    elif classifierChoice == 'elliptic classification':
        classifier = EllipticEnvelope(support_fraction=ECsupportFraction, contamination=0.05)
    elif classifierChoice == 'LOF':
        classifier = LocalOutlierFactor(n_neighbors=lofNeighbours,novelty=True,contamination=0.05)
    elif classifierChoice == 'isolation forest':
        classifier = IsolationForest(n_estimators = ifestimators,contamination=0.05)








    else:
        print('Error in classifier: classifierChoice unknown')
    classifier.fit(features_train) #Training

    if plot == 1: 
            figName = figName + '_' + classifierChoice
            plotOCC(features_train, classes_train,classifier,withPoints=withPoints,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(training_set)'),save=save,folderPath=folderPath)
    print(classifierChoice+' done !\n')
    return classifier




def importTextData(filePath,numberOfDatas = 1,separator = "\t",removeFirstLines = 3):
    '''
    Import the current data contained in a txt file inside a given folder to a Python data set

    Parameters
    ----------
    filePath : string
        Path of the file containing the datas. It must be as follow : [Time(s),Voltage(V),Current(A)] and separed using a tabulation
    numberOfData : int
        Number of different datas in the file. 
    separator : string, optional
        Separator used in the txt file to split the columns. Default is "\t"
    removeFirstLines : int, optional
        Number of lines to remove in the begining of the data file. For example when there is text explaining the data set in the begining. Default is 3
    
    Returns
    -------
    data : nparray of dim (n,3)
        Array containing the datas. It is as follow: [Time(s),Voltage(V),Current(mA)].

    '''
    file = open(filePath,'r');
    contents= file.readlines();
    del contents[0:removeFirstLines]; #Remove non data lines
    data = np.zeros((len(contents),numberOfDatas))
    for i in range(len(contents)):
        dataSplit = contents[i].split(separator);
        for j in range(numberOfDatas):
            data[i,j] = float(dataSplit[j])  
            
    file.close()
    return data;




def timeRupture (data,dataReference=None,penaltyValue=5,plot=0,xlabel='Indice',ylabel='Y',folderPath='',save=0,figName='time_series_rupture'):
    '''
    Algorithm finding the ruptures inside a time serie. It uses the Python package ruptures by Charles Truong, Laurent Oudre and Nicolas Vayatis.

    Parameters
    ----------
    data : nparray of dim (n,)
        The time serie we want to study
    dataReference : nparray of dim (n,), optional
        If the time serie is a latch up serie, it is possible to plot the the normal current on the same figure using this parameter. The default is None.
    penaltyValue : float, optional
        Penalty value of the alogrithm (>0). The default is 5.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'time_series_rupture'. Example: 'scaled_datas'

    Returns
    -------
    object : ruptures type
        Algo trained by data set.
    breakPoints : nparray of dim (n,)
        Position of ruptures found in data set
    dataBreaks : nparray of object dim (n,2)
        Array containing the portion of data between ruptures. First colunm contains the position of each value in the original data set and second colunm contains values.

    '''
    print('\ntimeRupture in progress ...') 
    dataBreak = []
    dataIndice = []
    ruptureAlgo = rpt.Pelt(model="rbf").fit(data)
    breakPoints = ruptureAlgo.predict(pen=penaltyValue)
    breakPoints = np.insert(breakPoints,0,int(0))
    breakPoints[len(breakPoints)-1] = breakPoints[len(breakPoints)-1] - 1 
    if len(breakPoints) == 1 and breakPoints[0] == len(data):   #If no rupture was found in the data set.
        print('\nWarning in timeRupture: No break point found in the given data set\n')
    else:
        breakPointsTemp = np.insert(breakPoints,0,0)    #We add a 0 to implement all values in dataBreak
        for i in range(len(breakPointsTemp)-1):
            dataBreak.append(data[breakPointsTemp[i]:breakPointsTemp[i+1]])
            dataIndice.append(np.arange(breakPointsTemp[i],breakPointsTemp[i+1],1))
        if plot == 1:
            label = None
            if dataReference is not None:
                plt.plot(dataReference,c='green',label='No Latch Data')
                label = 'Latch Data'
            plt.plot(data,c='blue',label=label)
            for points in breakPoints:
                plt.axvline(x=points - 0.5,
                            color='black',
                            linewidth=1.75,
                            linestyle='--')
            plt.title(figName)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)        
            plt.legend()
            saveFigure(save,folderPath,figName=('/'+figName+'.png'))
            plt.show()  
    print('timeRupture done !\n')
    return ruptureAlgo, np.array(breakPoints), np.array([dataIndice,dataBreak],dtype=object).transpose




def getClass(indices,classes,classesValuesToFind=[0,1]):
    '''
    Return the class of a given set of points. The matrix must have been created by the dataGenerator Matlab function
    If there are multiple points, the class chosen is the more present in the set.
    !!!!! Warning: if multiple classes have the same number of iterations, the first one to appear in the classesName array will be chosen!!!!!!!
    
    Parameters
    ----------
    indice : nparray of dim(n) or int
        Positons of datas in the data set.
    classes : nparray
        Array containing the class values for the data set.
    classesValuesToFind : array of int
        Values of the possible classes to find

    Returns
    -------
    classValue : int
        Return the class indice in classesName list
    

    '''
    if type(indices)==int:
        return classes[indices]
    
    classCount = np.zeros(len(classesValuesToFind))
    for i in range(len(classesValuesToFind)):   #We count the number of time each class appears in the set
        classCount[i] = np.count_nonzero(classes[indices[0]:indices[1]] == classesValuesToFind[i])
    mainClass = np.array(np.where(classCount == classCount.max())).transpose()
    # if len(mainClass) != 1:
    #     print('Warning in getClass: Multiple classes have same number of iteration. Only one will be selected')
    return mainClass[0][0]



def statsPerPoint(data,pointAccuracy=20,save=0,plot=0,folderPath='',figName='StatsPerPoint'):
    '''
    Compute the stats of each points contained in the given data. To calculate the stats of each points, we take the points close to them. 
    Note that for firsts lasts value, their stats will remain the same because they all have the same closest points (aka, closest indices in the data set, not closest values.)
    UPDATE: Now instead of taking the window before and after, we are taking the points after the "i" indice. The last points of the data set are not taken into consideration
    
    Parameters
    ----------
    data : nparray of dim (n,)
        Stats will be computed from this data set of n points. It is better to have a time serie.
    pointAccuracy : int, optional
        Accuracy of the stat calculation. This parameter decide how many points are taken from the closest point of our data set (ex if pointAccuracy=20, 10 points will be taken before and after the current point to calculate its stats values). The default is 20.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is ''. Example: 'scaled_data'


    Returns
    -------
    statData : nparray of dim (n,11)
        ['index','min', 'max', 'mean', 'variance', 'skewness', 'kurtosis','standard error of mean', 'median absolute deviation', 'geometric standard deviation', 'k statistic'])      

    '''
    print("\nStatsPerPoint in progress ...")
    statsDataTemp = {};
    a={}
    b={}
    c={}
    d={}
    e={}
    statsData = np.array(np.zeros((len(data),11)));
    labels=['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis','Standard error of mean', 'Median absolute deviation', 'Geometric standard deviation', 'k-statistic', 'Bayesian confidence interval', 'Wassenrstein distance']
    for i in range(len(statsData)):
        # if i < (pointAccuracy)//2:
        #     indice = i + pointAccuracy
        #     dataTemp = data[0:indice]
        # elif i+(pointAccuracy//2) > len(data):
        #     indice = len(data) - pointAccuracy + (len(data)-i)
        #     dataTemp = data[indice:len(data)]
        # else:
        #     dataTemp = data[i-(pointAccuracy//2):i+(pointAccuracy//2)]
        if i + pointAccuracy < len(data):
            dataTemp = data[i:i+pointAccuracy]
        
        statsDataTemp = list(stat.describe(dataTemp))
        a=stat.sem(dataTemp)
        b=stat.median_absolute_deviation(dataTemp)
        # c=list(stat.gmean(dataTemp))
        # c=stat.gstd(dataTemp)
        c=0
        d=stat.kstat(dataTemp)
        # bayes_mvs
        # e=list(stat.wasserstein_distance(dataTemp,range()))

        
        statsData[i]=[i,statsDataTemp[1][0],statsDataTemp[1][1],statsDataTemp[2],statsDataTemp[3],statsDataTemp[4],statsDataTemp[5],a,b,c,d]


    
    print('StatsPerPoint done!\n')
    return statsData



def FourierAnalysis(timeSerie,samplingTime,removeFreqZero=0,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo'):
    '''
    This function do a frequence analysis of a time serie

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Array containing the data set. In must have these 2 sets [Time(s),Data]
    samplingTime : int
        Sampling time used for the data set (in second). Exemple: 0.5s
    removeFreqZero : int, optional
        Put 1 if you want to remove the 0Hz from the resulting frequence. As it can be way higher than other frequence, it is not possible to correctly visualises other values of frequence
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'Signal_Fourier_Transfo'. Example: 'scaled_datas'

    Returns
    -------
    freqResult : nparray of dim(i,2)
        Frequence results for the time serie. It gives column 0 is the frequence axis and the column 1 is the amplitude of the corresponding frequence
    object : nparray of dim(i)
        Complex values of the FDT transoformation
    
    

    '''
    # print("\nFourierAnalysis in progress ...")
    if len(np.shape(timeSerie)) == 2:
        if np.shape(timeSerie)[1] == 2:
            fourier = np.fft.fft(timeSerie[:,1])
            if removeFreqZero:
                fourier=np.delete(fourier,0)
            sampleNumber=len(timeSerie[:,1])
            spectre = np.absolute(fourier)*2/sampleNumber
            freq = np.fft.fftfreq(sampleNumber,d=samplingTime)
            if removeFreqZero:
                freq=np.delete(freq,0)
            if plot:
                plt.xlabel('Freq (Hz)')
                plt.ylabel('Amplitude')
                plt.title(figName)
                plt.plot(freq[0:int(len(freq)/2)],spectre[0:int(len(spectre)/2)])
                saveFigure(save,folderPath,'/'+figName+'.png')
                plt.show()
            fourierFreqResult = np.array([freq,spectre]).transpose()
            # print('FourierAnalysis done!\n')
            return fourierFreqResult,fourier
        else:
            print("\nWARNING IN FOURIERANALYSIS: Dimension of timeSerie is incorrect\n")
            return 0,0





def FourierAnalysisPerPoint(timeSerie,samplingTime,removeFreqZero=0,pointAccuracy=50,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo_PerPoint'):
    '''
    This function do a frequence analysis of a time serie point per point

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Array containing the data set. In must have these 2 sets [Time(s),Data]
    samplingTime : int
        Sampling time used for the data set (in second). Exemple: 0.5s
    pointAccuracy : int, optional
        Accuracy of the stat calculation. This parameter decide how many points are taken from the closest point of our data set (ex if pointAccuracy=20, 10 points will be taken before and after the current point to calculate its stats values). The default is 20.
    removeFreqZero : int, optional
        Put 1 if you want to remove the 0Hz from the resulting frequence. As it can be way higher than other frequence, it is not possible to correctly visualises other values of frequence
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'Signal_Fourier_Transfo'. Example: 'scaled_datas'

    Returns
    -------
    spectreMean : nparray of dim(n)
        Array with the mean frequency for each point of the timeSerie
    
    

    '''
    print("\nFourierAnalysisPerPoint in progress ...")
    
    if len(np.shape(timeSerie)) == 2:
       if np.shape(timeSerie)[1] == 2:
        spectreTemp = [];
        # X = np.array(np.zeros((len(data),7)));
        
        fourier = np.fft.fft(timeSerie[:,1])
        if removeFreqZero:
            fourier=np.delete(fourier,0)
        sampleNumber=len(timeSerie[:,1])
        spectre = np.absolute(fourier)*2/sampleNumber
        freq = np.fft.fftfreq(sampleNumber,d=samplingTime)
        if removeFreqZero:
            freq=np.delete(freq,0)
        
        
        for i in range(len(timeSerie)):
            if i < (pointAccuracy)//2:
                indice = i + pointAccuracy
                timeSerieTemp = timeSerie[0:indice]
            elif i+(pointAccuracy//2) > len(timeSerie):
                indice = len(timeSerie) - pointAccuracy + (len(timeSerie)-i)
                timeSerieTemp = timeSerie[indice:len(timeSerie)]
            else:
                timeSerieTemp = timeSerie[i-(pointAccuracy//2):i+(pointAccuracy//2)]
            
            fourierFreqResult,fourier = FourierAnalysis(timeSerieTemp,samplingTime,removeFreqZero)    
            spectreTemp.append(fourierFreqResult[:,1].mean())
    
        if plot:
            plt.xlabel('Points')
            plt.ylabel('Frequence Mean Value')
            plt.title(figName)
            plt.plot(freq,spectre)
            saveFigure(save,folderPath,'/'+figName+'.png')
            plt.show()
        spectreMean = np.array(spectreTemp)    
        print('FourierAnalysisPerPoint done!\n')
        return spectreMean
    else:
            print("\nWARNING IN FOURIERANALYSIS: Dimension of timeSerie is incorrect\n")
            return 0;


def plotCurrent(x,noLatchData,latchData,xlabel='Time (s)',ylabel='Supply current (mA)',save=0,folderPath='',figName=' '):
    '''
    Plot the current of a given current set. It shows the current with and without the latch up current. Most efficient with the data sets created by the Matlab function 'dataGenerator'

    Parameters
    ----------
    x : nparray of dim (n,)
        Data set for x axis.
    noLatchData : nparray of dim (n,)
        Data set of the normal current.
    latchData : Tnparray of dim (n,)
        Data set of the latch current.
    xlabel : string, optional
        Name for the x axis. The default is 'Time (s)'.
    ylabel : string, optional
        Name for the y axis. The default is 'Current (mA)'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure. Used to  name the file if save=1. The default is 'Consumption_current'.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    # Plot of data set
    fig=plt.figure()
    plt.plot(x, latchData, "r", label="Anomaly");
    plt.plot(x, noLatchData, "b", label="Normal");
    plt.title(figName);
    plt.grid(True);
    plt.xlim(min(x)-10,max(x)+10);
    plt.xlabel(xlabel);
    plt.ylim(0, np.max(latchData)+np.max(latchData)*0.2);
    plt.ylabel(ylabel);
    plt.legend();
    saveFigure(save,folderPath,'/'+figName+'.png')
    plt.show();
    return fig

def saveFigure(save=0,folderPath='',figName='/figure.png'):
    '''
    Used to save the current figure inside a given folder

    Parameters
    ----------
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is '/figure.png'.


    Returns
    -------
    None.

    '''
    if save:
        if folderPath == '':
            print('Problem for' + figName +': No folder path given for saving the plot')
        else:
            if not path.exists(folderPath):
                makedirs(folderPath)
            figName = folderPath + figName;
            plt.savefig(figName,dpi=175,bbox_inches = 'tight');
            print('file save in: ' + figName)
            
            

def plotClassification(features_set,classes_set,classifier,classesName=['',''],colors=['green','purple', 'blue','pink','yellow','red','orange'],randomColors=0,withPoints=1,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    This function is used with the function 'classifier'. It plots the points inside the referef class obtained in 'classifier' function. It also shows a colored map to distinguish the different areas of each class.
    Note that the dim of classesName must be the >= of the max value of 'classes_set'. Note2: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : erie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classifier : classifier algo
        Classifier algorithm trained using the function 'classifer'.
    classesName : list of strings, optional
        Names of the different classes used for plotting.
        Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), 
        otherwise the name will not match the classes leading to false diagnostic!!!
        Needed if plot=1. The default is ''.
    colors : list of strings, optional
        Colors used for plotting the points. The default is ['green','purple', 'blue','pink','yellow','red','orange'].
    randomColors : int,optional
        Put 1 if you want matplotlib to choose the colors. Otherwise, the colors will be chosen using colors parameter. The default is 0.
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 1
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    scaling : int, optional
        Put 1 if you want to scale the data set. Not scaling can lead to wrong results using clustering algorithm as a feature very high from the others will be dominant. The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    
   
    classifierValues=classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    if randomColors == 1:
        plt.contourf(x1, x2, classifierValues, alpha = 0.75)
    else:
        #We change the scale levels to be sure every class will appear in the plot (otherwise some of them might fuze because too close in value)
        levels = []
        for i in range(int(classifierValues.max()+1)):
            if i in classifierValues:
                if len(levels) == 0:
                    levels.append(i-0.1)
                levels.append(i+0.1)
        #It is possible that the classifier does not map all classes if some are under represanted in regards of other classes. In that case, the color map will be incorrect
        #This condition remove non desired colors      
        colorsContour = colors.copy()
        if len(levels) <= len(colors):
            print('\nWARNING IN PlotClassification: The number of classes found by the classifier mismatch the real number of classes. The colors will be placed arbitrary as follow: [NormalCurrent=Green=0, Calculation=Purple=1, I/O=Blue=2, Temperature=Pink=3, Reset=Yellow=4, Latch=Red=5, Front Latch=Orange=6]\n!!!Be careful as this order might not suit your application!!!\n')  
            if ('green' in colorsContour) and (0 not in classifierValues):
                colorsContour.remove('green')
            if ('purple' in colorsContour) and (1 not in classifierValues):
                colorsContour.remove('purple')
            if ('blue' in colorsContour) and (2 not in classifierValues):
                colorsContour.remove('blue')
            if ('pink' in colorsContour) and (3 not in classifierValues):
                colorsContour.remove('pink')
            if ('yellow' in colorsContour) and (4 not in classifierValues):
                colorsContour.remove('yellow')
            if ('red' in colorsContour) and (5 not in classifierValues):
                colorsContour.remove('red')
            if ('orange' in colorsContour) and (6 not in classifierValues):
                colorsContour.remove('orange')          
        plt.contourf(x1, x2, classifierValues, alpha = 0.75, levels=levels, colors=colorsContour)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    if withPoints == 1:
        for i, j in enumerate(np.unique(classes_set)):
            i=int(i)
            j=int(j)
            if randomColors == 1:
                plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                        label = classesName[j])
            else:
                if len(colors) < i+1:
                    colors.insert(i,'black')
                plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                            c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig



def plotOCC(features_set,classes_set,classifier,withPoints=1,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    This function is used with the function 'classifier'. It plots the points inside the referef class obtained in 'classifier' function. It also shows a colored map to distinguish the different areas of each class.
    Note that the dim of classesName must be the >= of the max value of 'classes_set'. Note2: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : serie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data. (0 is normal and 1 is anomaly)
    classifier : classifier algo
        Classifier algorithm trained using the function 'classifer'.
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 1
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    scaling : int, optional
        Put 1 if you want to scale the data set. Not scaling can lead to wrong results using clustering algorithm as a feature very high from the others will be dominant. The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    colors = ['green','red', 'orange']
    classesName = ['normal', 'anomaly', 'WARNING']
    classifierValues=classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    #We switch normal value form -1 to 0) 
    for row in classifierValues:
        for i,col in enumerate(row):
            if col == -1:  
                row[i] = 1
            if col == 1:  
                row[i] = 0
    #classifierValues[np.where(classifierValues == -1)[0]] = 1  NOT WORKING
    plt.contourf(x1, x2, classifierValues, alpha = 0.75, levels = [-0.5,0.5,1.5], colors = colors)

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    if withPoints == 1:
        for i, j in enumerate(np.unique(classes_set)):
            i=int(i)
            j=int(j)
            
            plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                        c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig, classifierValues




def plotClustering(data,clusteringAlgo,n_clusters,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    Plot the labelled points of the give data set. It also shows the area of each clusters along with their centroids of the clustering algorithm trained with the function 'clustering'
    Note: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    clusteringAlgo : clustering algorithm
        Algo obtained using the 'clustering' function.
    n_clusters : int
        Number of clusters to show.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
        

    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.plot()
  
    for i in range(0,n_clusters):
        plt.scatter(data[clusteringAlgo.predict(data) == i,0], data[clusteringAlgo.predict(data) == i,1])
        # plt.scatter(data[classes == 1,0], data[classes == 1,1],s=100,label='Cluster 2')
    
    x1, x2 = np.meshgrid(np.arange(start = data[:, 0].min() - 1, stop = data[:, 0].max() + 1, step = 0.25),
                         np.arange(start = data[:, 1].min() - 1, stop = data[:, 1].max() + 1, step = 0.25))
    plt.contourf(x1, x2, clusteringAlgo.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75) 
    plt.scatter(clusteringAlgo.cluster_centers_[:,0],clusteringAlgo.cluster_centers_[:,1],s=100,c='red',label='Centroids')
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName=('/'+figName + '_' + str(n_clusters) + '_clusters.png'))
    plt.show()
    return fig


def plotLabelPoints(features_set,classes_set,classesName,colors=['green','purple', 'blue','pink','yellow','red','orange'],xlabel='X',ylabel='Y',figName='Labelled_points',save=0, folderPath=''):
    '''
    Plot the points inside the features_set attached with their classes in classes_set. This is used to better look a data set before doing a classification algorithm.

    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : erie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classesName : list of strings
        Names of the different classes used for plotting.
        Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), 
        otherwise the name will not match the classes leading to false diagnostic!!!
        Needed if plot=1. The default is ''.
    colors : list of strings, optional
        Colors used for plotting the points. The default is ['green','purple', 'blue','pink','yellow','red'].
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Labelled_points'.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(classes_set)):
        i = int(i)
        j = int(j)
        if len(colors) < i+1:
            colors.insert(i,'black')
        plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                    c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig


def preprocessing(dataPath,dataIndice = 1,dataChoice = 1,diagDataChoice = 2, windowSize = 20,dataName = 'dataSet',dataColumnChoice=2, plotFeatures = 0,savePath='',save=0,addNewLatchClass = 0):
    colors = ['green','red']
    plotStats = 0
    plotFrequency = 0
    className = ['normal','latch','','','','latch','front de latch up']
    
    if dataChoice == 2:
      #Folder path initialisation
    # dataPath = 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets'   
    # dataPath = dataPath + '\\AllDefectAdded.txt'
        separator = '\t'
        samplingDataTime = 75
        # testDataPath = dataPath + '\\All16.txt'
    

    
    #Data path initialisation
    if dataChoice == 1:
        dataPath = dataPath + '/datas'+ str(dataIndice) +'/diagData.txt'
        dataFolder = dataPath.split('/diagData.txt')[0];
        separator = ','
        samplingDataTime = 1000

    
    
    
    
    #Import Data
    timeSerie = importTextData(dataPath,3,separator)
    if separator == '\t': #Data coming from dataAcquisition and need to bu put in mA
        timeSerie[:,2] = timeSerie[:,2]*1000 
        plotCurrent(timeSerie[:,0],timeSerie[:,2],timeSerie[:,2],save=save,folderPath=savePath,figName = dataName + ' ')
    elif separator == ',':
        plotCurrent(timeSerie[:,0],timeSerie[:,1],timeSerie[:,2],save=save,folderPath=savePath,figName = dataName + ' ')
    

    
    
    #Class creation for trainSet
    if separator == '\t': #Data coming from dataAcquisition 
        dataClass = []
        for i in range(0,9989):
            dataClass.append(0)
        for i in range(9989,len(timeSerie)):
            dataClass.append(5)    
        dataClass = pd.DataFrame(data=dataClass,columns=['Latch up']).iloc[:,0]
    else:
        dataClass = pd.DataFrame(data=importTextData(dataFolder+'/statusData.txt',6,separator=',')[:,5],columns=['latch up']).iloc[:,0]
   
    
        
    #Adding new latch class
    if addNewLatchClass:
        colors.append('orange')
        firstLatchIndex=np.where(dataClass==5)[0][0]
        dataClass[firstLatchIndex:firstLatchIndex+45]=6

        
    
    
   
    #Finding Features
        #Stats
    statDataPerPoint = statsPerPoint(timeSerie[:,dataColumnChoice], pointAccuracy = windowSize, save=save,folderPath=savePath,plot=plotStats,figName=dataName + '_set_StatsPerPoint')
    [dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint,dataSEMPerPoint,dataMedianPerPoint,dataStdPerPoint,dataKstatPerPoint] = [statDataPerPoint[:,1],statDataPerPoint[:,2],statDataPerPoint[:,3],statDataPerPoint[:,4],statDataPerPoint[:,5],statDataPerPoint[:,6],statDataPerPoint[:,7],statDataPerPoint[:,8],statDataPerPoint[:,9],statDataPerPoint[:,10]]
        
        #Frequency
    dataFourier = FourierAnalysisPerPoint(timeSerie[:,[0,dataColumnChoice]], samplingDataTime,removeFreqZero=1, pointAccuracy = windowSize, plot=plotFrequency,save=save,folderPath=savePath,figName=dataName + '_frequency')
    
    
    #DiagData Creation
    if diagDataChoice == 1:
        diagData =  np.array([dataVariancePerPoint,dataMeanPerPoint]).transpose()
        ylabel = 'Mean current (mA)'
        xlabel = 'Variance'
        featureChoice = 'variance'
        featureName = 'Variance'
    elif diagDataChoice == 2:
        diagData =  np.array([dataFourier,dataMeanPerPoint]).transpose()
        ylabel = 'Mean current (mA)'
        xlabel = 'Fourier transformation'
        featureChoice = 'fourier'
        featureName = 'Fourier'
    elif diagDataChoice == 3:
        diagData =  np.array([dataVariancePerPoint,dataFourier]).transpose()
        ylabel = 'Fourier transformation'
        xlabel = 'Variance'
        featureChoice = 'var_fourier'
        featureName = 'Variance / Fourier'
    elif diagDataChoice == 4:
        diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint]).transpose()
        ylabel = 'Y'
        xlabel = 'X'
        featureChoice = 'stats'
        featureName = 'Stats'
    elif diagDataChoice == 5:
        diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint,dataFourier]).transpose()
        ylabel = 'Y'
        xlabel = 'X'
        featureChoice = 'stats_fourier'
        featureName = 'Stats / Fourier'
    elif diagDataChoice == 6:
            diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSEMPerPoint,dataMedianPerPoint,dataKstatPerPoint]).transpose()
            ylabel = 'Y'
            xlabel = 'X'
            featureChoice = 'bigStats'
            featureName = 'bigStats'
    
        
    
    
    
    
    #Scaling of datas
    sc1 = StandardScaler();
    diagDataScale = sc1.fit_transform(diagData);  
    
    
    
    
    #Plotting of features
    if plotFeatures:
        #Non scaled datas
            plotLabelPoints(diagData, dataClass, className,figName=dataName + '_NonScaled_'+featureChoice,colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        #Scaled datas
            plotLabelPoints(diagDataScale, dataClass, className,figName=dataName + '_Scaled_'+featureChoice,colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
            
    return timeSerie, diagData, diagDataScale, dataClass,featureChoice,xlabel,ylabel



def ifacDataFrame (index = 'index0'):
    return pd.DataFrame(index = [index], 
                        columns=['test used', 'knn time', 'knn accuracy','knn k','knn weight','knn metric',
                             'svm time', 'svm accuracy','svm kernel','svm gamma','svm weigth',
                             'decision tree time', 'decision tree accuracy','decision tree criterion','decision tree splitter','decision tree min split',
                             'random forest time', 'random forest accuracy','random forest estimators','random forest criterion','random forest min split',
                             'kmeans time', 'kmeans accuracy','kmeans n cluster','kmeans init','kmeans n init', 'kmeans max iter',
                             'HC time', 'HC accuracy','HC n cluster','HC affinity','HC linkage',
                             'svm time', 'svm accuracy','svm epsilon','svm min sample','svm distance',
                             'dyclee time', 'dyclee accuracy','dyclee g_size','dyclee outlier rejection'])
