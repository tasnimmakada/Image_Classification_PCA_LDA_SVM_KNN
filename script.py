import os, os.path, sys
import re
import numpy as np
from PIL import Image
import math
import operator
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers

## Read image (pgm) files ##
def read_pgm(filename):
    X=[]
    im = Image.open(filename)
    ## convert image matrix into a single array
    X = [item for sublist in np.asarray(im) for item in sublist]
    return X

## Create a dictionary with all data ##
def get_data_list():
    folderList = {}
    b = os.listdir(DIR)
    print('Reading files from the data folder')
    for foldername in b:
        fileList = []
        fileList.extend([name for name in os.listdir(DIR+"/"+foldername) 
            if os.path.isfile(os.path.join(DIR+"/"+foldername, name))])
        folderList[foldername] = fileList

    return folderList

def readAllImages(whichFold, resize):
    testFolderSequence=[]
    trainFolderSequence=[]
    trainNotTransposed=[]
    testNotTransposed=[]

    print('Creating training and testing arrays for fold '+ str(whichFold))
    if(resize):
        print('Resizing images')
    for foldername in folderList:
        temp = folderList[foldername]
        for j in range(0,10):
            if(resize):
                image = resize_and_read_image(DIR+"/"+foldername+"/"+temp[j])
            else:
                image = read_pgm(DIR+"/"+foldername+"/"+temp[j])
            aRow=[]
            aRow.append( image )
            parts = foldername.split('s')
            if(  j == (whichFold * 2) or j == ( (whichFold * 2)+1)  ):
                testNotTransposed.extend( aRow )
                testFolderSequence.append([int(parts[1])])
            else:
                trainNotTransposed.extend( aRow )
                trainFolderSequence.append([int(parts[1])])
    
    trainNotTransposed = np.array( trainNotTransposed )
    testNotTransposed = np.array( testNotTransposed )

    return trainNotTransposed, testNotTransposed, trainFolderSequence, testFolderSequence

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if (testSet[x][-1] == predictions[x]):
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def knn_with_pca(X):
    print('\nTask 1: KNN with PCA\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, False)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        ## Get reduced features and W ##
        print('Start Principle Component Analysis (PCA)')
        windowTitle = 'knn_with_pca_fold'+ str(fold)
        feature_vector, principle_comp = pca(training_data, windowTitle)
        feature_vector = feature_vector.transpose()

        ## Get reduced feature vector for testing data using W ##
        print('Calculating new feature set and performing 1-NN for prediction')
        reduced_test_set = np.dot(principle_comp.T, test_array.T)
        reduced_test_set = reduced_test_set.transpose()

        ## Get training and testing data with classes ##
        train_with_class = np.hstack((feature_vector, trainFolderSequence))
        test_with_class = np.hstack((reduced_test_set, testFolderSequence))

        ## Get neighbors for a test image using 1-NN##
        predictions=[]
        for img in test_with_class:
            neighbors = getNeighbors(train_with_class, img, 1)
            result = getResponse(neighbors)
            predictions.append(result)

        accuracy = getAccuracy(test_with_class, predictions)
        final_accuracy = final_accuracy + accuracy
        print('Accuracy for fold' + str(fold+1) + ': ' + repr(accuracy) + '%')

    final_accuracy = final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%')

def knn_with_pca_resized_image(X):
    print('\nTask 2: KNN with PCA using resized images\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, True)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        print('Start Principle Component Analysis (PCA)')
        windowTitle = 'knn_with_pca_resized_image_fold'+ str(fold)
        feature_vector, principle_comp = pca(training_data, windowTitle)
        feature_vector = feature_vector.transpose()

        ## Get reduced feature vector for testing data using W ##
        print('Calculating new feature set and performing 1-NN for prediction')
        reduced_test_set = np.dot(principle_comp.T, test_array.T)
        reduced_test_set = reduced_test_set.transpose()

        ## Get training and testing data with classes ##
        train_with_class = np.hstack((feature_vector, trainFolderSequence))
        test_with_class = np.hstack((reduced_test_set, testFolderSequence))
     
        ## Get neighbors for a test image using 1-NN##
        predictions=[]
        for img in test_with_class:
            neighbors = getNeighbors(train_with_class, img, 1)
            result = getResponse(neighbors)
            predictions.append(result)

        accuracy = getAccuracy(test_with_class, predictions)
        final_accuracy = final_accuracy + accuracy
        print('Accuracy for fold ' + str(fold+1) + ': ' + repr(accuracy) + '%\n')

    final_accuracy = final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%\n')

def knn_with_lda(X):
    print('\nTask 3: KNN with LDA\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, False)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        ## Get reduced features and W ##
        print('Start Linear Discriminant Analysis (LDA)')
        windowTitle = 'knn_with_lda_fold'+ str(fold)
        feature_vector, linear_discriminant = lda(training_data, trainFolderSequence, windowTitle)

        ## Get reduced feature vector for testing data using W ##
        reduced_test_set = np.dot(linear_discriminant.T, test_array.T)
        reduced_test_set = reduced_test_set.transpose()

        print('Calculating new feature set and performing 1-NN for prediction')
        ## Get training and testing data with classes ##
        train_with_class = np.hstack((feature_vector, trainFolderSequence))
        test_with_class = np.hstack((reduced_test_set, testFolderSequence))

        ## Get neighbors for a test image using 1-NN##
        predictions=[]
        for img in test_with_class:
            neighbors = getNeighbors(train_with_class, img, 1)
            result = getResponse(neighbors)
            predictions.append(result)

        accuracy = getAccuracy(test_with_class, predictions)
        final_accuracy = final_accuracy + accuracy
        print('Accuracy for fold' + str(fold+1) + ': ' + repr(accuracy) + '%\n')

    final_accuracy = final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%\n')

def knn_with_pca_lda(X):
    print('\nTask 4: KNN with PCA and LDA\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, False)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        ## Get reduced features and W ##
        print('Start Principle Component Analysis (PCA)')
        windowTitle = 'knn_with_pca-lda_pca_fold'+ str(fold)
        ## Image dimensionality equal to length of training data ##
        dimens = len(training_data)
        feature_vector, principle_comp = pca(training_data, windowTitle, dimens)
        feature_vector = feature_vector.transpose()

        ## Get reduced feature vector for testing data using W ##
        reduced_test_set = np.dot(principle_comp.T, test_array.T)
        reduced_test_set = reduced_test_set.transpose()

        ## Get reduced features and W ##
        print('Start Linear Discriminant Analysis (LDA)')
        windowTitle = 'knn_with_pca-lda_lda_fold'+ str(fold)
        feature_vector2, linear_comp = lda(feature_vector, trainFolderSequence, windowTitle)

        ## Get reduced feature vector for testing data using W ##
        print('Calculating new feature set and performing 1-NN for prediction')
        reduced_test_set2 = np.dot(linear_comp.T, reduced_test_set.T)
        reduced_test_set2 = reduced_test_set2.transpose()

        ## Get training and testing data with classes ##
        train_with_class = np.hstack((feature_vector2, trainFolderSequence))
        test_with_class = np.hstack((reduced_test_set2, testFolderSequence))

        ## Get neighbors for a test image using 1-NN##
        predictions=[]
        for img in test_with_class:
            neighbors = getNeighbors(train_with_class, img, 1)
            result = getResponse(neighbors)
            predictions.append(result)

        accuracy = getAccuracy(test_with_class, predictions)
        final_accuracy = final_accuracy + accuracy
        print('Accuracy for fold' + str(fold+1) + ': ' + repr(accuracy) + '%\n')

    final_accuracy = final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%\n')

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(float(distance))

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def resize_and_read_image(filename):
    X=[]
    im = Image.open(filename)
    img_anti = im.resize((56, 46), Image.ANTIALIAS)
    ## convert image matrix into a single array
    X = [item for sublist in np.asarray(img_anti) for item in sublist]
    return X    

def get_principle_comp(eig_vals, eig_vecs, dimensions):
    ## We sort the eigvals and return the indices for the
    # ones we want to include (specified by "dimensions" paramater) ##
    eigval_max = np.argsort(-eig_vals)[:dimensions]
    eigvec_max = eig_vecs[:,eigval_max]
    return eigvec_max

def get_linear_analysis_vector(eig_vals, eig_vecs, dimensions):
    ## We sort the eigvals and return the indices for the
    # ones we want to include (specified by "dimensions" paramater) ##
    eigval_max = np.argsort(-eig_vals)[:dimensions]
    eigvec_max = eig_vecs[:,eigval_max]
    return eigvec_max

def pca(X, windowTitle, dimens=None):
    ## Centerize the image data ##
    print('Centerizing image')
    all_data = X.transpose((1,0))
    all_data = all_data - np.mean(all_data, 0)
    ## Calculate transpose matrix ##
    C = np.cov(all_data)
    ## Calculate eigenvalues and eigenvectors ##
    print('Calculating eigenvalues and eigenvectors')
    eig_val_cov, eig_vec_cov= np.linalg.eigh(C)
    ## Plot eigenvalues to decide dimensions ##
    dims = plot_eigenvalues(windowTitle, eig_val_cov)
    ## Calculate the principle components ##
    if(dimens == None):
        principle_comp = get_principle_comp(eig_val_cov, eig_vec_cov, dims )
    else:
        principle_comp = get_principle_comp(eig_val_cov, eig_vec_cov, dimens )
    ## Calculate the new feature vector ##
    feature_vector = np.dot(principle_comp.T, all_data)
    return feature_vector, principle_comp

def lda(X, y, windowTitle):
    ## Centerize the image ##
    print('Centerizing image')
    all_data = X.transpose((1,0))
    all_data = all_data - np.mean(all_data, 0)
    all_data = all_data.transpose((1,0))
    ## Calculate mean vectors classwise for each feture##
    mean_vectors = []
    newMean=[]
    currentClass=''
    i=0
    for aClass in y:
        myclass = aClass[0] 
        if((currentClass=='') or (currentClass!=myclass)):
            if not newMean:
                currentClass=myclass
                newMean.append(all_data[i])
            else:
                mean_vectors.append(np.mean(newMean, axis=0))
                newMean=[]
                currentClass=myclass
                newMean.append(all_data[i])

            i=i+1
        elif(i==(len(y)-1)):
            mean_vectors.append(np.mean(newMean, axis=0))
        else:
            newMean.append(all_data[i])
            i=i+1

    mean_vectors = np.array(mean_vectors)
    ## Calculate Sw by calculating covariace ##
    S_W = np.cov(all_data.T)
    current_dims = len(S_W)
    overall_mean = np.mean(all_data, axis=0)
    overall_mean = np.array(overall_mean)
    ## Calculate Sb using the scatter matrix
    S_B = np.zeros((current_dims,current_dims))
    overall_mean = np.vstack(overall_mean) 
    for mean_vec in (mean_vectors):  
        n = 8
        mean_vec = np.vstack(mean_vec) 
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    print('Calculating eigenvalues and eigenvectors')
    ## Calculate eigenvalues and eigenvectors ##
    eig_vals, eig_vecs = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
    ## Plot eigenvalues and get the size of new dimensions ##
    dims = plot_eigenvalues(windowTitle, eig_vals)
    ## Calculate the principle components ##
    W = get_linear_analysis_vector(eig_vals, eig_vecs, dims )
    ## Calculate the new feature vector ##
    feature_vector = np.dot(W.T, all_data.T)

    return feature_vector.T, W

def plot_eigenvalues(windowTitle, eig_vals):
    print('Plotting eigenvalues and calculating the cut down value')
    positives=[]
    negatives=[]
    for a in eig_vals:
        if a>0:
            positives.append(a)
        else:
            negatives.append(a)

    fig = plt.figure()
    fig.canvas.set_window_title(windowTitle)
    x1=np.arange(len(negatives))
    plt.scatter(x1, negatives, color='r')
    dd=len(eig_vals)-len(negatives)
    x2=np.arange(len(negatives), len(eig_vals))
    plt.scatter(x2, positives, color='b')
    plt.annotate('+ve eigenvals start here', xy=(len(negatives)+1,positives[1]), xytext=(1200,100000), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('total +ve eigenvals=' + str(len(positives)), xy=(len(eig_vals)+1,positives[-1]), xytext=(1600,positives[-1]+15000))
    plt.savefig(windowTitle+'.png')
    print('Plot saved as: '+ windowTitle+'.png')
    plt.close()
    return len(positives)

##### SVM #####
## Predict probability for each test image for all classes ##
def predict(image, wbs):
    prediction_list = {}
    for foldername in wbs:
        w_temp = wbs[foldername]
        w = w_temp[0]
        b = w_temp[1]
        pred_val = np.dot(image, w) + b
        prediction_list[foldername]=pred_val
    
    return prediction_list

## Train input by calculating alpha, and using it to calculate w and b ##
def train(X, y):
    X = np.asarray(X, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    
    num_samples, num_features = X.shape

    temp_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            temp_mat[i,j] = np.dot(X[i], X[j])
    
    yTrans = y.transpose()
    part = np.outer(y,yTrans)
    H = part*temp_mat
    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(np.ones(num_samples) * -1)    
    G = cvxopt.matrix(np.diag(np.ones(num_samples) * -1))
    h = cvxopt.matrix(np.zeros(num_samples))    
    A = cvxopt.matrix(  y, (1,num_samples), tc='d' )    
    b = cvxopt.matrix(0.0)
    
    ## solve QP problem using cvxopt lib ##
    cvxopt.solvers.options['show_progress'] = False
    ans = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(ans['x'])
    
    ## find Support vectors that have greater values ##
    sv = alpha > 1e-9
    sv = np.array(sv,dtype=bool)
    ind = np.arange(len(alpha))[sv]
    s = alpha[sv]
    new_sv=X[sv]
    sv_y=y[sv]
    
    ## Weight vector ##
    w = np.zeros(num_features)
    for n in range(len(s)):
        w += s[n] * sv_y[n] * new_sv[n]
    
    ## b ##
    b = 0
    for n in range(len(s)):
        b += sv_y[n]
        b -= np.sum(s * sv_y * temp_mat[ind[n],sv])
    b /= len(s)
    
    return w, b

## Read all files, divide in training and testing data, perform training and testing for accuracy
def svm(folderList):
    print('\nTask 5: SVM\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, False)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        ## Get unique class names ##
        uniquetrainFolderSequence = np.unique(trainFolderSequence)

        all_data = np.hstack(( training_data, trainFolderSequence )) 
        wbs={}

        ## Create Support Vectors ##
        print('Creating support vectors')
        for claz in uniquetrainFolderSequence:
            counts=8
            currentClass = []
            otherClasses = []
            for arow in all_data:
                if( arow[-1] == claz ):
                    if ( counts < 9):
                        currentClass.append( arow[:-1] )
                    else:
                        otherClasses.append(arow[:-1])

                    counts=counts-1
                else:
                    otherClasses.append(arow[:-1])
            
            y1 = np.ones(len(currentClass))    
            y2 = np.ones(len(otherClasses)) * -1
            X_train = np.vstack((currentClass, otherClasses))
            y_train = np.hstack((y1, y2))
            w, b = train(X_train, y_train)
            wbs[claz] = [w,b]

        ## Testing phase ##
        print('Predicting class for each test file')
        accuracy = 0
        sorted_pred = {}
        f=0
        for thing in test_array:
            prediction_list = predict(thing, wbs)
            mymax = 0
            myclass = ''
            i=0
            for key in prediction_list:
                if (i == 0):
                    mymax = prediction_list[key]
                    myclass = key
                    i=i+1
                else:
                    if(prediction_list[key] > mymax):
                        mymax = prediction_list[key]
                        myclass = key
                
            ## check if predicted value is same as actual value ##
            actualClass = testFolderSequence[f][0]
            if myclass == actualClass:
                accuracy = accuracy + 1

            f=f+1

        final_acc = (accuracy/(80))*100
        final_accuracy=final_accuracy+final_acc
        print('Accuracy for fold' + str(fold+1) + ': ' + repr(final_acc) + '%\n')

    final_accuracy=final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%\n')

## Read all files, divide in training and testing data, perform training and testing for accuracy
def svm_with_pca(folderList):
    print('\nTask 6: SVM with PCA\n')
    final_accuracy=0
    # 5-fold data cross-validation    
    for fold in range(0,5):
        training_data, test_array, trainFolderSequence, testFolderSequence = readAllImages(fold, False)

        trainFolderSequence = np.swapaxes(trainFolderSequence,1,1)
        testFolderSequence = np.swapaxes(testFolderSequence,1,1)

        ## Get reduced features and W ##
        print('Start Principle Component Analysis (PCA)')
        windowTitle = 'svm_with_pca_fold'+ str(fold)
        feature_vector, principle_comp = pca(training_data, windowTitle)
        feature_vector = feature_vector.transpose()
        ## Get reduced feature vector for testing data using W ##
        print('Calculating new feature set and performing SVM for prediction')
        reduced_test_set = np.dot(principle_comp.T, test_array.T)
        reduced_test_set = reduced_test_set.transpose()
        ## Get unique class names ##
        uniquetrainFolderSequence = np.unique(trainFolderSequence)

        all_data = np.hstack(( feature_vector, trainFolderSequence )) 
        wbs={}

        ## Create Support Vectors ##
        for claz in uniquetrainFolderSequence:
            counts=8
            currentClass = []
            otherClasses = []
            for arow in all_data:
                if( arow[-1] == claz ):
                    if ( counts < 9):
                        currentClass.append( arow[:-1] )
                    else:
                        otherClasses.append(arow[:-1])

                    counts=counts-1
                else:
                    otherClasses.append(arow[:-1])
            
            y1 = np.ones(len(currentClass))    
            y2 = np.ones(len(otherClasses)) * -1
            X_train = np.vstack((currentClass, otherClasses))
            y_train = np.hstack((y1, y2))
            w, b = train(X_train, y_train)
            wbs[claz] = [w,b]

        ## Testing phase ##
        accuracy = 0
        sorted_pred = {}
        f=0
        for thing in reduced_test_set:
            prediction_list = predict(thing, wbs)
            mymax = 0
            myclass = ''
            i=0
            for key in prediction_list:
                if (i == 0):
                    mymax = prediction_list[key]
                    myclass = key
                    i=i+1
                else:
                    if(prediction_list[key] > mymax):
                        mymax = prediction_list[key]
                        myclass = key
                
            ## check if predicted value is same as actual value ##
            actualClass = testFolderSequence[f][0]
            if myclass == actualClass:
                accuracy = accuracy + 1

            f=f+1

        final_acc = (accuracy/(80))*100
        final_accuracy=final_accuracy+final_acc
        print('Accuracy for fold' + str(fold+1) + ': ' + repr(final_acc) + '%\n')

    final_accuracy=final_accuracy/5
    print('Final accuracy for 5-fold cross validation is: '+ repr(final_accuracy) + '%\n')


## Main ##
DIR = './data'
folderList = get_data_list()
knn_with_pca(folderList)
knn_with_pca_resized_image(folderList)
knn_with_lda(folderList)
knn_with_pca_lda(folderList)
svm(folderList)
svm_with_pca(folderList)