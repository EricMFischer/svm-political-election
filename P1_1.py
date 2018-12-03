'''
Part 1: Face Social Traits Classification (or Regression)
Train binary SVMs or SVRs to predict the perceived traits (social attributes) from photos.
You can use the pre-computed facial keypoint locations and extract HoG (histogram of oriented gradient) features using the enclosed MATLAB function. You can further try your own favorite features.
Note: We do not explicitly divide the image set into train and test sets. Therefore, you need to perform k-fold cross-validation and report the obtained accuracy.

1.1: Classification by Landmarks (features are landmarks)
Train 14 SVMs or SVRs only using the provided facial landmarks as features.
Write a script which reads the annotation file and the landmark file.
Then train 14 models -- one for each attribute dimension using the training examples.
After training, you should apply the learned classifiers on the test examples and measure performance (classification accuracy) of the classifiers.
Since the labels are imbalanced, you should report the average precisions.
Peform k-fold cross-validation to choose the LIBSVM parameters.
Report:
1) Average accuracies and precisions on training and testing data for each of the 14 models.
(After doing SVR, set threshold to do classification and report accuracies and precision.)
2) LIBSVM parameters of the 14 models (e.g. Gamma, Epsilon, and C values)
Note: When training SVM classifiers with LIBSVM or other libraries, you can specify a parameter C to control the trade-off between classification accuracy and regularization. Tune this parameter if you believe your classifiers are over-fitting.

1.2: Classification by Rich Features (features are landmarks + hog features)
Next step is to extract richer visual features (appearance) from the images.
Here, include the HoG features and additionally choose whatever features you want to try.
Then repeat the earlier step to train and test your models, but using augmented features:
[landmark] + [new appearance feature]. (You can concatenate two types of feature vectors into one.)
Compare the performance with the previous one.
Report:
1) and 2) from before.
3) Names of the features you have used (if more than landmarks and hog features).
4) Comparison to classfication by landmarks.
'''
import os
import scipy.io as sio
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, precision_recall_fscore_support
from skimage import exposure
from skimage.feature import hog
from skimage.io import imread_collection
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def save_data(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def get_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

# ---------------------------------------- Training ----------------------------------------

def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

def train_svr_trait_models(X_train, y_train):
    '''
    Standard ranges for C and γ are C∈[2^−5,2^15] and  γ∈[2^−15,2^3]
    The reason for the exponential grid is that both C and gamma are scale parameters that act multiplicatively, so doubling gamma is as likely to have roughly as big an effect (but in the other direction) as halving it. This means that if we use a grid of approximately exponentially increasing values, there is roughly the same amount of "information" about the hyper-parameters obtained by the evaluation of the model selection criterion at each grid point.
    I usually search on a grid based on integer powers of 2, which seems to work out quite well (I am working on a paper on optimising grid search - if you use too fine a grid you can end up over-fitting the model selection criterion, so a fairly coarse grid turns out to be good for generalisation as well as computational expense.).
    As to the wide range, unfortunately the optimal hyper-parameter values depends on the nature of the problem, and on the size of the dataset and cannot be determine a-priori. The reason for the large, apparently wasteful grid, is to make sure good values can be found automatically, with high probability.
    '''
    # param_grid = [{ # v5 (using)
    #     'kernel': ['rbf'],
    #     'C': [2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15],
    #     'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5],
    #     'epsilon': [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]
    # }]
    # param_grid = [{ # v5 (fast)
    #     'kernel': ['rbf'],
    #     'C': [2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13],
    #     'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2],
    #     'epsilon': [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2]
    # }]
    param_grid = [{ # v7
        'kernel': ['rbf'],
        'C': [2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15],
        'gamma': [2**-17, 2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5],
        'epsilon': [2**-9, 2**-7, 2**-5, 2**-3, 2**-1]
    }]
    SVR_trait_models = []
    SVR_trait_params = []
    SVR_trait_scores = []
    for i in range(0,14):
        X = X_train
        y = y_train[:, i] # 1 trait's values for training imgs

        svr = SVR(kernel='rbf')
        print('Initiating grid search for SVR', i)
        clf = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        clf.fit(X, y)

        print('Best parameters found on set:', clf.best_params_)
        print('Best score found on set:', clf.best_score_)

        SVR_trait_models.append(clf)
        SVR_trait_params.append(clf.best_params_)
        SVR_trait_scores.append(clf.best_score_)

    save_data(SVR_trait_models, 'svr_trait_models_v7.pkl')
    save_data(SVR_trait_params, 'svr_trait_params_v7.pkl')
    save_data(SVR_trait_scores, 'svr_trait_scores_v7.pkl')

def assign_binary_labels(y_true, y_pred):
    mean = np.mean(y_true) # regression threshold based on mean
    y_true = np.where(y_true >= mean, 1, -1)
    y_pred = np.where(y_pred >= mean, 1, -1)
    return y_true, y_pred

# ---------------------------------------- Plotting ----------------------------------------

# acc = (tp + tn) / (p + n) -- i.e. true / total
# precision (ppv) = tp / (tp + fp) -- i.e. true positive / all positive
# i.e. ability of the classifier not to label as positive a negative sample
def get_clf_accuracy_precision(X, y, rich_feats=False):
    svr_trait_models = get_data('svr_trait_models_v5.pkl')
    if rich_feats:
        svr_trait_models = get_data('svr_trait_models_v5_rich.pkl')
    accuracy = []
    precision = []
    for i, clf in enumerate(svr_trait_models):
        y_true, y_pred = y[:, i], clf.predict(X)
        y_true, y_pred = assign_binary_labels(y_true, y_pred)
        # print(classification_report(y_true, y_pred))

        accuracy.append(len(np.where(y_true == y_pred)[0]) / len(y_true))
        precision.append(precision_recall_fscore_support(y_true, y_pred, average='weighted')[0])
    return accuracy, precision

def disp_clf_accuracy_precision(X_train, y_train, X_test, y_test, X_train_r=None, y_train_r=None, X_test_r=None, y_test_r=None):
    train_acc, train_prec = get_clf_accuracy_precision(X_train, y_train)
    test_acc, test_prec = get_clf_accuracy_precision(X_test, y_test)
    if X_train_r is not None:
        train_acc_r, train_prec_r = get_clf_accuracy_precision(X_train_r, y_train_r, True)
        test_acc_r, test_prec_r = get_clf_accuracy_precision(X_test_r, y_test_r, True)
    print('train_acc: ', np.mean(train_acc), train_acc)
    print('test_acc: ', np.mean(test_acc), test_acc)
    print('train_prec: ', np.mean(train_prec), train_prec)
    print('test_prec: ', np.mean(test_prec), test_prec)

    X = list(range(1, 15))
    if X_train_r is not None:
        plt.plot(X, train_acc, marker='o', color='navy', lw=.3, linestyle='--', label='Poor Train Accuracy')
        plt.plot(X, test_acc, marker='o', color='red', lw=.3, linestyle='--', label='Poor Test Accuracy')
        plt.plot(X, train_acc_r, marker='o', color='navy', lw=1, label='Rich Train Accuracy')
        plt.plot(X, test_acc_r, marker='o', color='red', lw=1, label='Rich Test Accuracy')
    else:
        plt.plot(X, train_acc, marker='o', color='navy', lw=1, linestyle='--', label='Train Accuracy')
        plt.plot(X, test_acc, marker='o', color='red', lw=1, label='Test Accuracy')
    plt.xlabel('Traits')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.show()

    if X_train_r is not None:
        plt.plot(X, train_prec, marker='o', color='navy', lw=.3, linestyle='--', label='Poor Train Precision')
        plt.plot(X, test_prec, marker='o', color='red', lw=.3, linestyle='--', label='Poor Test Precision')
        plt.plot(X, train_prec_r, marker='o', color='navy', lw=1, label='Rich Train Precision')
        plt.plot(X, test_prec_r, marker='o', color='red', lw=1, label='Rich Test Precision')
    else:
        plt.plot(X, train_prec, marker='o', color='navy', lw=1, label='Train Precision')
        plt.plot(X, test_prec, marker='o', color='red', lw=1, label='Test Precision')
    plt.xlabel('Traits')
    plt.ylabel('Classification Precision')
    plt.legend()
    plt.show()


def get_test_mse(X_test, y_test, X_test_r=None, y_test_r=None):
    svr_trait_models = get_data('svr_trait_models_v5.pkl')
    if X_test_r is not None:
        svr_trait_models = get_data('svr_trait_models_v5_rich.pkl')
        X_test, y_test = X_test_r, y_test_r

    test_mse = []
    for i, clf in enumerate(svr_trait_models):
        y_true, y_pred = y_test[:, i], clf.predict(X_test)
        test_mse.append(mean_squared_error(y_true, y_pred))
    return test_mse

def disp_mse(X_test, y_test, X_test_r=None, y_test_r=None):
    test_mse = get_test_mse(X_test, y_test)
    train_mse = np.array(get_data('svr_trait_scores_v5.pkl')) * -1
    if X_test_r is not None:
        test_mse_r = get_test_mse(X_test, y_test, X_test_r, y_test_r)
        train_mse_r = np.array(get_data('svr_trait_scores_v5_rich.pkl')) * -1
    # print('test_mse: ', np.mean(test_mse), test_mse)

    X = list(range(1, 15))
    if X_test_r is not None:
        plt.plot(X, train_mse, marker='o', color='navy', lw=.3, linestyle='--', label='Poor Train MSE')
        plt.plot(X, test_mse, marker='o', color='red', lw=.3, linestyle='--', label='Poor Test MSE')
        plt.plot(X, train_mse_r, marker='o', color='navy', lw=1, label='Rich Train MSE')
        plt.plot(X, test_mse_r, marker='o', color='red', lw=1, label='Rich Test MSE')
    else:
        plt.plot(X, train_mse, marker='o', color='navy', lw=1, label='Train MSE')
        plt.plot(X, test_mse, marker='o', color='red', lw=1, label='Test MSE')
    plt.xlabel('Traits')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()

def print_params_scores():
    print('v5 params: ', get_data('svr_trait_params_v5.pkl'))
    v5_scores = get_data('svr_trait_scores_v5.pkl')
    print('v5 scores: ', np.mean(v5_scores), v5_scores)

    print('v7 params: ', get_data('svr_trait_params_v7.pkl'))
    v7_scores = get_data('svr_trait_scores_v7.pkl')
    print('v7 scores: ', np.mean(v7_scores), v7_scores)

# ---------------------------------------- HOG Features ----------------------------------------
'''
Compute a Histogram of Oriented Gradients (HOG) by:
1) (optional) global image normalization
2) computing the gradient image in row and col
3) computing gradient histograms
4) normalizing across blocks
5) flattening into a feature vector
'''
def get_hog_features():
    if os.path.exists('./hog_features.pkl'):
        return get_data('hog_features.pkl')

    imgs = imread_collection('./img/*.jpg')
    hog_feats = []
    for i, img in enumerate(imgs):
        # pixels_per_cell default is (8,8), cells_per_block at its best
        # more orientations, or num_bins, would take longer to process
        # fd, hog_img = hog(img, orientations=8, pixels_per_cell=(32,32), cells_per_block=(1, 1),visualize=True, multichannel=True) # fast
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L1')
        print('fd: ', i, np.shape(fd), np.max(fd), np.min(fd), type(fd))
        hog_feats.append(fd)

    return normalize(hog_feats)

def run_svr():
    # ------------------------------------ SVM: Question 1.1 ------------------------------------
    f = sio.loadmat('./train-anno.mat')
    img_landmarks = normalize(f['face_landmark']) # 491 imgs x 160 landmark coordinates
    img_traits = f['trait_annotation'] # 491 imgs x 14 traits
    X_train, X_test, y_train, y_test = train_test_split(img_landmarks, img_traits, test_size=0.2, random_state=42)

    # train 14 models -- one for each attribute dimension using the training examples
    # train_svr_trait_models(X_train, y_train)

    # disp_mse(X_test, y_test)
    # disp_clf_accuracy_precision(X_train, y_train, X_test, y_test)

    # ------------------------------------ SVM: Question 1.2 ------------------------------------
    '''
    1.2: Classification by Rich Features (features are landmarks + hog features)
    Next step is to extract richer visual features (appearance) from the images.
    Here, include the HoG features and additionally choose whatever features you want to try.
    Then repeat the earlier step to train and test your models, but using augmented features:
    [landmark] + [new appearance feature]. (You can concatenate two types of feature vectors.)
    Compare the performance with the previous one.
    Report:
    1) and 2) again.
    3) Names of the features you have used (if more than landmarks and hog features).
    4) Comparison to classfication by landmarks.
    '''
    hog_feats = get_hog_features()
    rich_feats = np.concatenate([img_landmarks, hog_feats], axis=1)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(rich_feats, img_traits, test_size=0.2, random_state=42)

    # # train 14 models with concatenated landmarks and hog features
    # train_svr_trait_models(X_train_r, y_train_r)

    disp_mse(X_test, y_test, X_test_r, y_test_r)
    disp_clf_accuracy_precision(X_train, y_train, X_test, y_test, X_train_r, y_train_r, X_test_r, y_test_r)


def main():
    run_svr()

if __name__ == "__main__":
    main()
