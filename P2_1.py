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
import scipy.io as sio
import numpy as np
import pickle
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, precision_recall_fscore_support
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

def train_svc_trait_models(X_train, y_train, senators=False):
    # param_grid = [{ # v5
    #     'kernel': ['rbf'],
    #     'C': [2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15],
    #     'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5],
    #     'epsilon': [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]
    # }]
    param_grid = [{ # v5 (fast)
        'C': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15],
        'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8]
    }]

    svc = SVC(kernel='rbf')
    clf = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    print('Best parameters found on set:', clf.best_params_)
    print('Best score found on set:', clf.best_score_)

    if senators:
        save_data(clf, 'svc_model_sen.pkl')
        save_data(clf.best_params_, 'svc_params_sen.pkl')
        save_data(clf.best_score_, 'svc_scores_sen.pkl')
    else:
        save_data(clf, 'svc_model_gov.pkl')
        save_data(clf.best_params_, 'svc_params_gov.pkl')
        save_data(clf.best_score_, 'svc_scores_gov.pkl')

def assign_binary_labels(real_values):
    return np.where(real_values >= 0, 1, -1)

# ---------------------------------------- Plotting ----------------------------------------

# acc = (tp + tn) / (p + n) -- i.e. true / total
# precision (ppv) = tp / (tp + fp) -- i.e. true positive / all positive
# i.e. ability of the classifier not to label as positive a negative sample
def get_clf_accuracy_precision(X, y, senators=False):
    clf = get_data('svc_model_sen.pkl') if senators else get_data('svc_model_gov.pkl')
    y_pred = clf.predict(X)
    # print(classification_report(y, y_pred))
    accuracy = len(np.where(y == y_pred)[0]) / len(y)
    precision = precision_recall_fscore_support(y, y_pred, average='weighted')[0]
    return accuracy, precision

# ---------------------------------------- HOG Features ----------------------------------------
def get_hog_features(imgs):
    hog_feats = []
    for i, img in enumerate(imgs):
        # pixels_per_cell default is (8,8), cells_per_block at its best
        # more orientations, or num_bins, would take longer to process
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L1')
        hog_feats.append(fd)
    return normalize(hog_feats)

def run_svc():
    f_gov, f_sen = sio.loadmat('./stat-gov.mat'), sio.loadmat('./stat-sen.mat')
    gov_lm = normalize(f_gov['face_landmark']) # (112 governors, 160 lm coordinates)
    sen_lm = normalize(f_sen['face_landmark']) # (116 senators, 160 lm coordinates)
    gov_vote_diff, sen_vote_diff = f_gov['vote_diff'].flatten(), f_sen['vote_diff'].flatten()
    gov_elec_outcomes = assign_binary_labels(gov_vote_diff) # (112, 1)
    sen_elec_outcomes = assign_binary_labels(sen_vote_diff) # (116, 1)

    gov_hog_feats = get_data('gov_hog_features.pkl') # (112, 1800)
    gov_feats = np.concatenate([gov_lm, gov_hog_feats], axis=1) # (112, 1960)
    X_train_gov, X_test_gov, y_train_gov, y_test_gov = train_test_split(gov_feats, gov_elec_outcomes, test_size=0.2, random_state=42)

    sen_hog_feats = get_data('sen_hog_features.pkl') # (116, 1800)
    sen_feats = np.concatenate([sen_lm, sen_hog_feats], axis=1) # (116, 1960)
    X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(sen_feats, sen_elec_outcomes, test_size=0.2, random_state=42)

    # ------------------------------------ SVM: Question 2.1 ------------------------------------
    # for governors, train 14 SVC models with landmarks and hog feats
    # train_svc_trait_models(X_train_gov, y_train_gov)
    # train_acc, train_prec = get_clf_accuracy_precision(X_train_gov, y_train_gov)
    # test_acc, test_prec = get_clf_accuracy_precision(X_test_gov, y_test_gov)
    # print('train acc and prec: ', train_acc, train_prec)
    # print('test acc and prec: ', test_acc, test_prec)

    # for senators, train 14 SVC models with landmarks and hog feats
    # train_svc_trait_models(X_train_sen, y_train_sen, True)
    # train_acc, train_prec = get_clf_accuracy_precision(X_train_sen, y_train_sen, True)
    # test_acc, test_prec = get_clf_accuracy_precision(X_test_sen, y_test_sen, True)
    # print('train acc and prec: ', train_acc, train_prec)
    # print('test acc and prec: ', test_acc, test_prec)

    # ------------------------------------ SVM: Question 2.2 ------------------------------------


def main():
    run_svc()

if __name__ == "__main__":
    main()
