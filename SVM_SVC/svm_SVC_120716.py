
# coding: utf-8

# In[2]:

#Development notes
#this script was last modified on 120916

#What this script does:
##Help you define C to get the best estimator 
###(plot wide range then optomize within smaller range and pick kernel)

##Run the model and get the accuracy

##Test the significane of this model by permutation

##Plot a confusion matrix of the predictions

##Plot a precision recall plot

##Plot feature selection and their weights (univariant, svm weight and weight post selection)

###Am still getting the below error when run optomization of C:
##UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
  ##'precision', 'predicted', average, warn_for)


# In[38]:

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from decimal import Decimal
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.model_selection import permutation_test_score, ShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score 
from sklearn.pipeline import make_pipeline
from sklearn import metrics  
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
import numpy as np
import itertools
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


# In[27]:

##get and plot see smv_c plotting
def plotCvalSearch(X,y,cRange):
    '''plot a curve of the scores from a large rangew of Cs
    Modified from 
    http://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html 
    Warning: for smaller sets of samples you will get an error for the 0.7 test/train split
    X = OTU table (pandas Dataframe)
    y = meta data (pandas Dataframe)
    cRange = rang of Cs want to test, reccomend starting with cRange=np.logspace(-4,3)
    '''
    X=np.array(X)
    y=np.array(y)
    n_samples=len(X)
    clf_sets = [(SVC(C=1,kernel='linear'), cRange, X, y)]
    colors = ['navy', 'cyan', 'darkorange']
    lw = 2
    for fignum, (clf, cs, X, y) in enumerate(clf_sets):
        # set up the plot for each regressor
        plt.figure(fignum, figsize=(9, 10))
        for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
            param_grid = dict(C=cs)
            # To get nice curve, we need a large number of iterations to
            # reduce the variance
            grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                                cv=ShuffleSplit(train_size=train_size,
                                                n_splits=250, random_state=1))
            
            try:
                grid.fit(X, y)
            #If the dataset is smaller all the test/train divisions might result in an error
            except ValueError:
                print("single classifier in train size ", train_size)
                break 
            scores = grid.cv_results_['mean_test_score']

            scales = [(1, 'No scaling'),
                      ((n_samples * train_size), '1/n_samples'),
                      ]

            for subplotnum, (scaler, name) in enumerate(scales):
                plt.subplot(2, 1, subplotnum + 1)
                plt.xlabel('C')
                plt.ylabel('CV Score')
                grid_cs = cs * float(scaler)  # scale the C's
                plt.semilogx(grid_cs, scores, label="fraction %.2f" %
                             train_size, color=colors[k], lw=lw)
                plt.title('scaling=%s' % name)
                plt.legend(loc="best")
    plt.show()


# In[5]:

def getCwCrossVal(X, y, cRange, cv):
    '''
    Create model, and determine best C for data.
    Modified from http://scikit-learn.org/0.15/auto_examples/grid_search_digits.html
    X = OTU table (pandas Dataframe)
    y = meta data (pandas Dataframe)
    cRange = range of Cs want to test, can refine from plotCvalSearch()
    cv = reccomend using LeaveOneOut()
    '''
    ##split to test and train for C definition 
    X=np.array(X)
    y=np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': cRange},
                    {'kernel': ['linear'], 'C': cRange}]
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=cv,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


 
    


# In[6]:

def cross_val_ClassPred(X, y, C, cv, kernel):
    '''
    Leave out samples, create model, and predict left out samples. 
    Then return accuracy. Add in plotting of predictions later.
    X = OTU table (pandas Dataframe)
    y = meta data (pandas Dataframe)
    C = defined from getCwCrossVal()
    cv = reccomend using LeaveOneOut()
    kernel = kernel wish to use, linear or rbf
    '''
    X=np.array(X)
    y=np.array(y)
    #removed Standard Scaler as it removes the mean and does an additional normalization, 
    ##but my data already normalized
    clf=SVC(C=C, kernel=kernel)
    predicted = cross_val_predict(clf, X, y, cv=cv)
    score = cross_val_score(clf, X, y, cv=cv)
    accuracy=metrics.accuracy_score(y, predicted)
    print(accuracy)
    return(predicted,score)



# In[ ]:




# In[7]:

#testing permutation test plotting 
def plot_permuteTest(X, y, C, cv, kernel):
    '''modified from 
    http://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py
    X = OTU table (pandas Dataframe)
    y = meta data (pandas Dataframe)
    C = defined from getCwCrossVal()
    cv = reccomend using LeaveOneOut()
    kernel = kernel wish to use, linear or rbf'''
    
    X=np.array(X)
    y=np.array(y)
    n_classes = np.unique(y).size
    svm = SVC(C=C,kernel=kernel)
    score, permutation_scores, pvalue = permutation_test_score(
        svm, X, y, scoring="accuracy", cv=cv, n_permutations=1000, n_jobs=1)
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    # View histogram of permutation scores
    plt.hist(permutation_scores, 20, label='Permutation scores')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, 
             label='Luck (%s)' % (1. / n_classes))
    plt.ylim(ylim)
    plt.legend(loc='best')
    plt.xlabel('Score')
    plt.show()


# In[8]:


def plot_confusMatrix(y, predicted, twoXtwo=False):
    '''
    modified from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    y = meta data (pandas Dataframe)
    predicted = [0] output of cross_val_ClassPred()
    twoXtwo = how many classes do you have? if two hen True, if more then False'''
    y=np.array(y)
    classes = np.unique(y)
    cm = confusion_matrix(y, predicted)
    np.set_printoptions(precision=2)
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if twoXtwo:
        oddsratio, pvalue=st.fisher_exact(cm)
        plt.title('Confusion matrix'+'\n'+'p-value: %s' % ('%.2E' % Decimal(pvalue)))
    else:
        plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[21]:

def plot_precisionRecall(y,score, classNum=3):
    '''
    Uses [1] output of cross_val_ClassPred() as an input
    modified from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    y = meta data (pandas Dataframe)
    score = [1] output of cross_val_ClassPred()
    classNum = number of classes you have'''
    y=np.array(y)
    y = label_binarize(y, classes=(list(range(0,classNum))))
    score= label_binarize(score, classes=(list(range(0,classNum))))
    n_classes = y.shape[1]
    print(n_classes)
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(0,n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                            score[:, i])
        average_precision[i] = average_precision_score(y[:, i], score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
        score.ravel())
    average_precision["micro"] = average_precision_score(y, score,
                                                         average="micro")


    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy',
         label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()


# In[51]:

def plot_featureWeight(X, y, C, kernel):
    '''modified from 
    http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py
    X = OTU table (pandas Dataframe)
    y = meta data (pandas Dataframe)
    C = defined from getCwCrossVal()
    kernel = kernel wish to use, linear or rbf'''
    #not sure if need to run this with cross validation, couldn't quite fig how to do that
    y=np.array(y)
    X=np.array(X)
    X_indices = np.arange(X.shape[-1])
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)
    #scores = cross_val_score(clf, X, y, cv=cv)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='darkorange')
    clf=SVC(C=C, kernel=kernel)
    clf.fit(X, y)
    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()

    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
            color='navy')

    clf_selected=SVC(C=C, kernel=kernel)
    clf_selected.fit(selector.transform(X), y)

    svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.max()

    plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection', color='c')


    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()


# In[ ]:



