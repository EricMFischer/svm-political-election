'''
Part 2: Election Outcome Prediction
2:1: Prediction by Rich Features (classify election outcome w/ rich features)
Using the same features from section 1.2, train a classifier to classify the election outcome.
Report: Average accuracies on training and testing data. The chosen model parameters.
Perform k-fold cross-validation and report the average accuracy/precisions.
The point is to achieve an accuracy higher than chance.

2.2: Prediction by Face Social Traits (classify outcome w/ 14 features of real-valued traits)
We finally consider a two-layer-model in which we first project each facial image in a 14-dimesional attribute space and then perform binary classification of the election outcome in the obtained feature space.
Specifically, you need to apply the rich classifiers that you trained in the section 1.2 to each politician's image and collect all the outputs of the 14 classifiers (use real-valued confidence instead of label). Treat these outputs in 14 categories as a new feature vector that represents the image.

(Since each race comprises two candidates, you can define a pair of politicians as one data point by subtracting a train feature vector A from another vector B, and train a binary classifier: F_AB = F_A - F_B. Do not include a bias term. Then you can again train SVM classifiers using these new feature vectors. Compare the result with direct prediction in 2.1.)
Report: 1) Average accuracies on training and testing data. 2) Chosen model params.
3) Comparison to direct prediction by rich features (2.1).

2.3: Analysis of Results
At a minimum, show the correlations between the facial attributes and the election outcomes. What are the facial attributes that lead to the electoral success?
Report: Correlation coefficients of each of the facial attributes with the election outcomes.
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
    # param_grid = [{ # v5
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
    param_grid = [{ # v6 (fast)
        'kernel': ['rbf'],
        'C': [2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6],
        'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7],
        'epsilon': [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2]
    }]
    print('In HOG training')
    # param_grid = [{ # v6
    #     'kernel': ['rbf'],
    #     'C': [2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15],
    #     'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5],
    #     'epsilon': [2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]
    # }]
    SVR_trait_models = []
    SVR_trait_params = []
    SVR_trait_scores = []
    for i in range(0,14):
        X = X_train # (n, 160) -- landmark values for split training imgs
        y = y_train[:, i] # (n,) -- 1 trait's values for split training imgs

        svr = SVR(kernel='rbf')
        print('Initiating grid search for 1 SVR...')
        clf = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        clf.fit(X, y)

        print('Best parameters found on set:', clf.best_params_)
        print('Best score found on set:', clf.best_score_)

        SVR_trait_models.append(clf)
        SVR_trait_params.append(clf.best_params_)
        SVR_trait_scores.append(clf.best_score_)

    save_data(SVR_trait_models, 'svr_trait_models_v6_rich.pkl')
    save_data(SVR_trait_params, 'svr_trait_params_v6_rich.pkl')
    save_data(SVR_trait_scores, 'svr_trait_scores_v6_rich.pkl')

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
    # print('train_acc: ', np.mean(train_acc), train_acc)
    # print('test_acc: ', np.mean(test_acc), test_acc)
    # print('train_prec: ', np.mean(train_prec), train_prec)
    # print('test_prec: ', np.mean(test_prec), test_prec)

    X = list(range(1, 15))
    plt.plot(X, train_acc, marker='o', color='navy', lw=1, label='Train Accuracy')
    plt.plot(X, test_acc, marker='o', color='red', lw=1, label='Test Accuracy')
    if X_train_r is not None:
        plt.plot(X, train_acc_r, marker='o', color='blue', lw=1, label='Rich Train Accuracy')
        plt.plot(X, test_acc_r, marker='o', color='orange', lw=1, label='Rich Test Accuracy')
    plt.xlabel('Traits')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.show()

    plt.plot(X, train_prec, marker='o', color='navy', lw=1, label='Train Precision')
    plt.plot(X, test_prec, marker='o', color='red', lw=1, label='Test Precision')
    if X_train_r is not None:
        plt.plot(X, train_prec_r, marker='o', color='blue', lw=1, label='Rich Train Precision')
        plt.plot(X, test_prec_r, marker='o', color='orange', lw=1, label='Rich Test Precision')
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
    # print('train_mse: ', train_mse)
    # print('test_mse: ', test_mse)

    X = list(range(1, 15))
    plt.plot(X, train_mse, marker='o', color='navy', lw=1, label='Train MSE')
    plt.plot(X, test_mse, marker='o', color='red', lw=1, label='Test MSE')
    if X_test_r is not None:
        plt.plot(X, train_mse_r, marker='o', color='blue', lw=1, label='Rich Train MSE')
        plt.plot(X, test_mse_r, marker='o', color='orange', lw=1, label='Rich Test MSE')
    plt.xlabel('Traits')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()

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
    if os.path.exists('./hog_features_fast.pkl'): # use for now
        return get_data('hog_features_fast.pkl')

    imgs = imread_collection('./img/*.jpg')
    hog_feats = []
    for i, img in enumerate(imgs):
        # pixels_per_cell default is (8,8), cells_per_block at its best
        # more orientations, or num_bins, would take longer to process
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L1')
        # fd, hog_img = hog(img, orientations=8, pixels_per_cell=(32,32), cells_per_block=(1, 1),visualize=True, multichannel=True)
        print('fd: ', i, np.shape(fd), np.max(fd), np.min(fd), type(fd))
        hog_feats.append(fd)

    return normalize(hog_feats)

def run_svr():
    # ------------------------------------ SVM: Question 1.1 ------------------------------------
    f = sio.loadmat('./train-anno.mat')
    img_landmarks = normalize(f['face_landmark']) # 491 imgs x 160 landmark coordinates
    img_traits = f['trait_annotation'] # 491 imgs x 14 traits
    X_train, X_test, y_train, y_test = train_test_split(img_landmarks, img_traits, test_size=0.2, random_state=42)

    # ------------------------------------ SVM: Question 1.2 ------------------------------------
    hog_feats = get_hog_features()
    rich_feats = np.concatenate([img_landmarks, hog_feats], axis=1)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(rich_feats, img_traits, test_size=0.2, random_state=42)

    # ------------------------------------ SVM: Question 2.1 ------------------------------------
    '''
    Using the same features from section 1.2, train a classifier to classify the election outcome.
    Report: Average accuracies on training and testing data. The chosen model parameters.
    Perform k-fold cross-validation and report the average accuracy/precisions.
    The point is to achieve an accuracy higher than chance.
    For RANK SVM: Luyao tried C from 2^-15 to 2^15 and found that when C = 2^-4 the model gives the best testing prediction.
    Other student found 2^2 for governers best and 2^6 for senators.
    Training, test acc for governers: 0.70, 0.61
    Training, test acc for senators: 0.89, 0.63
    '''

    # train 14 models with concatenated landmarks and hog features
    # train_svr_trait_models(X_train_r, y_train_r)

    disp_mse(X_test, y_test, X_test_r, y_test_r)
    disp_clf_accuracy_precision(X_train, y_train, X_test, y_test, X_train_r, y_train_r, X_test_r, y_test_r)


def main():
    run_svr()

if __name__ == "__main__":
    main()
