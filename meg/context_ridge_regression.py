import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
import os
from numpy.linalg import inv, svd
import time
from scipy.stats import zscore


def corr(X, Y):
    return np.mean(zscore(X) * zscore(Y), axis=0)

def R2(Pred, Real):
    SSres = np.mean((Real - Pred) ** 2, axis=0)
    SStot = np.var(Real, axis=0)
    return np.nan_to_num(1 - SSres / SStot)

def R2r(Pred, Real):
    R2rs = R2(Pred, Real)
    ind_neg = R2rs < 0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= -1
    return R2rs

def ridge(X, Y, lmbda):
    return np.dot(inv(X.T.dot(X) + lmbda * np.eye(X.shape[1])), X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error

def ridge_sk(X, Y, lmbda):
    rd = Ridge(alpha=lmbda)
    rd.fit(X, Y)
    return rd.coef_.T

def ridgeCV_sk(X, Y, lambdas):
    rd = RidgeCV(alphas=lambdas, solver='svd')
    rd.fit(X, Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge_sk(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error

def ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        d = s / (s ** 2 + lmbda)
        weights = np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error

def cross_val_ridge(train_features, train_data, n_splits=10, 
                    lambdas=np.array([10**i for i in range(-6, 10)]),
                    method='plain',
                    do_plot=False):
    
    ridge_1 = dict(plain=ridge_by_lambda,
                   svd=ridge_by_lambda_svd,
                   ridge_sk=ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain=ridge,
                   svd=ridge_svd,
                   ridge_sk=ridge_sk)[method]
    
    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        cost = ridge_1(train_features[trn], train_data[trn],
                       train_features[val], train_data[val], 
                       lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost, aspect='auto')
        r_cv += cost
    if do_plot:
        plt.figure()
        plt.imshow(r_cv, aspect='auto', cmap='RdBu_r')

    argmin_lambda = np.argmin(r_cv, axis=0)
    weights = np.zeros((train_features.shape[1], train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:, idx_vox] = ridge_2(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])

def eval(feature_name, feature_file, sub, save_regressed_y=False):  
    results_save_dir = "../pred/meg_sub" + str(sub) + "_predictions/"
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    print(feature_name)
    np.random.seed(9)
    kf = KFold(n_splits=4)
    features = np.load(feature_file, allow_pickle=True)

    save_dir = os.path.join(results_save_dir, feature_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if sub < 10:
        data = np.load('../subjects/new/sub0' + str(sub) + '-meg-data-ses0.npy', allow_pickle=True)
    else:
        data = np.load('sub-' + str(sub) + '-meg-data-ses0.npy', allow_pickle=True)
    
    for eachlayer in np.arange(1):   # Adjusted for layers
        print("layer " + str(eachlayer))
        if os.path.exists(os.path.join(save_dir, str(eachlayer) + "_r2s.npy")):
            continue
        
        print(data.shape)
        print(features.shape)
        r2s = np.zeros(208 * 81)   
        corr = np.zeros((208 * 81, 2))

        split_num = 0

        all_preds = []
        all_reals = []

        for train, test in kf.split(np.arange(4)):
            # Use contexts from features
            x_train = np.concatenate([np.array(features[train[0]][eachlayer]), 
                                      np.array(features[train[1]][eachlayer]), 
                                      np.array(features[train[2]][eachlayer])], axis=0)
            x_test = np.array(features[test[0]][eachlayer])
            # print(data[train[0]].shape, len(data[train[1]]), len(data[train[2]]))

            y_train = np.concatenate([np.array(data[train[0]]).reshape(len(data[train[0]]), 208 * 301), 
                                      np.array(data[train[1]]).reshape(len(data[train[1]]), 208 * 301), 
                                      np.array(data[train[2]]).reshape(len(data[train[2]]), 208 * 301)], axis=0)
            y_test = np.array(data[test[0]]).reshape(len(data[test[0]]), 208 * 301)
            
            print(x_train.shape)
            print(y_train.shape)
            print(x_test.shape)
            print(y_test.shape)

            weights, lbda = cross_val_ridge(x_train, y_train)        
            y_pred = np.dot(x_test, weights)

            np.save(os.path.join(save_dir, "{}_y_pred_{}".format(str(eachlayer), split_num)), np.nan_to_num(y_pred))
            np.save(os.path.join(save_dir, "{}_y_test_{}".format(str(eachlayer), split_num)), y_test)
            
            split_num += 1

            all_reals.append(y_test)
            all_preds.append(y_pred)

        all_reals = np.vstack(all_reals)
        all_preds = np.vstack(all_preds)

        r2s = r2_score(all_reals, all_preds, multioutput="raw_values")

        for i in range(all_reals.shape[1]):
            if np.nonzero(all_reals[:,i])[0].size > 0:
                corr[i] = stats.pearsonr(all_reals[:,i], all_preds[:,i])
            else:
                r2s[i] = 0
                corr[i][1] = 1

        print(np.max(r2s))

        np.save(os.path.join(save_dir, str(eachlayer) + "_r2s"), np.nan_to_num(r2s))
        np.save(os.path.join(save_dir, str(eachlayer) + "_corr"), np.nan_to_num(corr))

allfeatures = ['context_audio_joint_embeddings.npy']
for sub in np.arange(1,9):
   for eachfeature in allfeatures:
       eval( eachfeature.split('.')[0]+'_predictions', eachfeature,sub)