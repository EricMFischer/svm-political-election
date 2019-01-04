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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage.io import imread_collection
from scipy.stats.stats import pearsonr

MODEL = 'svc_model_sen_14_feats.pkl'
SCORES = 'svc_params_sen_14_feats.pkl'
PARAMS = 'svc_scores_sen_14_feats.pkl'

MODELS_RICH = 'svr_trait_models_rich.pkl'

def save_data(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def get_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

# --------------------------------------- Training ---------------------------------------
def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

def train_svc_trait_model(X_train, y_train):
    param_grid = [{
        'C': [2**-4]
    }]
    svc = LinearSVC(fit_intercept=False, max_iter=100000, tol=1e-05)
    clf = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, return_train_score=True)
    clf.fit(X_train, y_train)
    # print('cv_results: ', clf.cv_results_)

    print('Best parameters found on hold-out data in set:', clf.best_params_)
    print('Mean cross-validated score of best estimator:', clf.best_score_)

    save_data(clf, MODEL)
    save_data(clf.best_params_, PARAMS)
    save_data(clf.best_score_, SCORES)
    return clf

# ------------------------------------- Predictions --------------------------------------
def assign_binary_labels(real_values):
    return np.where(real_values >= 0, 1, -1)

# acc = (tp + tn) / (p + n) -- i.e. true / total
def get_clf_accuracy(clf, X, y):
    # clf = get_data(MODEL)
    # y_pred = clf.predict(X)
    # accuracy = len(np.where(y == y_pred)[0]) / len(y)
    # return accuracy
    return clf.score(X, y)

def get_svr_models_pred(X):
    svr_models = get_data(MODELS_RICH)
    svr_preds = [clf.predict(X) for clf in svr_models]
    return np.array(svr_preds).T

# svr_preds: (112, 14) for gov, (116, 14) for sen
def get_election_features(svr_preds):
    feats = []
    it = iter(svr_preds)
    for feat in it:
        feats.append(feat - next(it))
    neg_feats = []
    it_2 = iter(svr_preds)
    for feat in it_2:
        neg_feats.append(next(it_2) - feat)
    return feats + neg_feats

def get_svr_model_corr_coefficients(preds, vote_diff):
    feats = [] # (56,14)
    it = iter(preds)
    for feat in it:
        feats.append(feat - next(it))

    vote_diffs = []
    it = iter(vote_diff)
    for diff in it:
        vote_diffs.append(diff)
        next(it)
    vote_diffs = np.multiply(vote_diffs, -1)

    feats_T = np.array(feats).T
    trait_corr = []
    for trait_vals in feats_T:
        trait_corr.append(pearsonr(trait_vals, vote_diffs)[0])
    return trait_corr

def run_linear_svc():
    f_gov, f_sen = sio.loadmat('./stat-gov.mat'), sio.loadmat('./stat-sen.mat')
    gov_lm = normalize(f_gov['face_landmark']) # (112 governors, 160 lm)
    sen_lm = normalize(f_sen['face_landmark']) # (116 senators, 160 lm)
    gov_vote_diff, sen_vote_diff = f_gov['vote_diff'].flatten(), f_sen['vote_diff'].flatten()
    gov_elec_outcomes = assign_binary_labels(gov_vote_diff) # (112, 1)
    sen_elec_outcomes = assign_binary_labels(sen_vote_diff) # (116, 1)

    # -------------------------------- SVM: Question 2.1 ---------------------------------
    # for governors, train linearSVC model with landmarks and hog feats
    gov_hog_feats = get_data('hog_features_gov.pkl') # (112, 1800)
    gov_feats = np.concatenate([gov_lm, gov_hog_feats], axis=1) # (112, 1960)

    X_train_gov, X_test_gov, y_train_gov, y_test_gov = train_test_split(gov_feats, gov_elec_outcomes, test_size=0.2)

    # clf = train_svc_trait_model(X_train_gov, y_train_gov)
    # train_acc = get_clf_accuracy(clf, X_train_gov, y_train_gov)
    # test_acc = get_clf_accuracy(clf, X_test_gov, y_test_gov)
    # print('gov train acc: ', train_acc)
    # print('gov test acc: ', test_acc)

    # for senators, train linearSVC model with landmarks and hog feats
    sen_hog_feats = get_data('hog_features_sen.pkl') # (116, 1800)
    sen_feats = np.concatenate([sen_lm, sen_hog_feats], axis=1) # (116, 1960)

    X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(sen_feats, sen_elec_outcomes, test_size=0.2)

    # clf = train_svc_trait_model(X_train_sen, y_train_sen)
    # train_acc = get_clf_accuracy(clf, X_train_sen, y_train_sen)
    # test_acc = get_clf_accuracy(clf, X_test_sen, y_test_sen)
    # print('sen train acc: ', train_acc)
    # print('sen test acc: ', test_acc)

    # -------------------------------- SVM: Question 2.2 ---------------------------------
    # apply 14 rich trait models to gov and sen images to get 14 real-valued features per image
    gov_pred = get_svr_models_pred(gov_feats) # (112, 14) -- feat vector for img i: (i, :)
    sen_pred = get_svr_models_pred(sen_feats) # (116, 14)

    # represent 2 competing politicans as 1 feature vector F_AB = F_A - F_B
    # gov_elec_feats = normalize(get_election_features(gov_pred)) # (56, 14)
    # sen_elec_feats = normalize(get_election_features(sen_pred)) # (58, 14)

    # train SVC models again with new features for governors and senators
    # outcomes = ([1] * 56) + ([-1] * 56)
    # X_train_gov, X_test_gov, y_train_gov, y_test_gov = train_test_split(gov_elec_feats, outcomes, test_size=0.2)
    # clf = train_svc_trait_model(X_train_gov, y_train_gov)
    # train_acc = get_clf_accuracy(clf, X_train_gov, y_train_gov)
    # test_acc = get_clf_accuracy(clf, X_test_gov, y_test_gov)
    # print('gov train acc: ', train_acc)
    # print('gov test acc: ', test_acc)

    # outcomes = ([1] * 58) + ([-1] * 58)
    # X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(sen_elec_feats, outcomes, test_size=0.2)
    # clf = train_svc_trait_model(X_train_sen, y_train_sen)
    # train_acc = get_clf_accuracy(clf, X_train_sen, y_train_sen)
    # test_acc = get_clf_accuracy(clf, X_test_sen, y_test_sen)
    # print('sen train acc: ', train_acc)
    # print('sen test acc: ', test_acc)

    # -------------------------------- SVM: Question 2.3 ---------------------------------
    gov_corr_coefficients = get_svr_model_corr_coefficients(gov_pred, gov_vote_diff)
    print('gov correlation coefficients: ', gov_corr_coefficients)
    sen_corr_coefficients = get_svr_model_corr_coefficients(sen_pred, sen_vote_diff)
    print('sen correlation coefficients: ', sen_corr_coefficients)

def main():
    run_linear_svc()

if __name__ == "__main__":
    main()
