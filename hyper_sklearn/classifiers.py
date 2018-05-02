from functools import partial

import numpy
from hyperopt import hp
import hyperopt.pyll
import sklearn.discriminant_analysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier,
)
from sklearn.naive_bayes import (
    MultinomialNB,
    GaussianNB,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
from sklearn.svm import (
    SVC,
    LinearSVC,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#########
# Discriminant Analysis
#########

def hyper_lda_classifier(name, **kwargs):
    # name = 'ldac'
    solver_shrinkage = hp.choice(name + '_solverShrinkageDual',
                                 [
                                     ('svd', None),
                                     ('lsqr', None),
                                     ('lsqr', 'auto'),
                                     ('eigen', None),
                                     ('eigen', 'auto')
                                 ])
    params = {
        'solver': solver_shrinkage[0],
        'shrinkage': solver_shrinkage[1],
        'n_components': 4 * hyperopt.pyll.scope.int(
            hp.qloguniform(
                name + '_ncomponents',
                low=numpy.log(0.51),
                high=numpy.log(30.5),
                q=1.0))
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.discriminant_analysis.LinearDiscriminantAnalysis)(**params)


def hyper_qda_classifier(name, **kwargs):
    # name = 'qdac'
    params = {
        'reg_param': hp.uniform(
            name + '_regParam',
            0.0, 1.0),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)(**params)


##########
# Ensemble Methods
##########


def hyper_adaboost_classifier(name, **kwargs):
    # name = 'adac'
    params = {
        'n_estimators': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_nEstimators', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
        'learning_rate': hp.lognormal(name + '_learningRate', numpy.log(0.01),
                                      numpy.log(10.0)),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(AdaBoostClassifier)(**params)


# TODO
# def hyper_bagging_classifier(**kwargs):
#     name = 'bagc'
#     params = {
#         'n_estimators' : scope.int(
#             hp.qloguniform(name + 'nEstimators', numpy.log(10.5),
#                            numpy.log(1000.5), 1)),
#         'learning_rate': hp.lognormal(name + 'learningRate', numpy.log(0.01),
#                                       numpy.log(10.0)),
#     }
#     params.update(kwargs)
#     return Literal(BaggingClassifier)(**params)


def hyper_gboost_classifier(name, **kwargs):
    # name = 'gboost'
    params = {
        'loss': hp.choice(name + '_loss',
                          ['deviance', 'exponential']),
        'learning_rate': hp.lognormal(name + '_learningRate',
                                      numpy.log(0.01),
                                      numpy.log(10.0)),
        'n_estimators': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_nEstimators', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
        'max_depth': hp.pchoice(name + '_maxDepth', [
            (0.7, None),
            (0.1, 2),
            (0.1, 3),
            (0.1, 4),
        ]),
        'subsample': hp.pchoice(name + '_subsample', [
            (0.2, 1.0),  # default choice.
            (0.8, hp.uniform(name + '_subsample_sgb', 0.5, 1.0)),
        ]),
        'min_samples_split': 2,
        'min_samples_leaf': hp.choice(name + '_minSamplesLeaf', [
            1,
            hyperopt.pyll.scope.int(
                hp.qloguniform(name + '_minSamplesLeaf_gt1',
                               numpy.log(1.5),
                               numpy.log(50.5),
                               1))
        ]),
        'max_features': hp.pchoice(name + '_maxFeatures', [
            (0.2, 'sqrt'),  # most common choice.
            (0.1, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform(name + '_maxFeatures' + '_frac', 0., 1.))
        ]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(GradientBoostingClassifier)(**params)


def hyper_rf_classifier(name, **kwargs):
    # name = 'rf'
    params = {
        'criterion': hp.choice(name + '_criterion',
                               ['gini', 'entropy']),
        'n_estimators': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_nEstimators', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
        'max_features': hp.pchoice(name + '_maxFeatures', [
            (0.2, 'sqrt'),  # most common choice.
            (0.1, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform(name + '_maxFeatures' + '_frac', 0., 1.))
        ]),
        'max_depth': hp.pchoice(name + '_maxDepth', [
            (0.7, None),
            (0.1, 2),
            (0.1, 3),
            (0.1, 4),
        ]),
        'min_samples_leaf': hp.choice(name + '_minSamplesLeaf', [
            1,
            hyperopt.pyll.scope.int(
                hp.qloguniform(name + '_minSamplesLeaf_gt1',
                               numpy.log(1.5),
                               numpy.log(50.5),
                               1))
        ]),
        'bootstrap': hp.choice(name + '_bootstrap', [True, False]),
        'class_weight': hp.choice(name + '_classWeight',
                                  ['balanced', 'balanced_subsample',
                                   None]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(RandomForestClassifier)(**params)


def hyper_xtrees_classifier(name, **kwargs):
    # name = 'xtrees'
    params = {
        'n_estimators': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_nEstimators', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
        'criterion': hp.choice(name + '_criterion',
                               ['gini', 'entropy']),
        'max_features': hp.pchoice(name + '_maxFeatures', [
            (0.2, 'sqrt'),
            (0.1, 'log2'),
            (0.1, None),
            (0.6, hp.uniform(name + '_maxFeatures' + '_frac', 0., 1.))
        ]),
        'max_depth': hp.pchoice(name + '_maxDepth', [
            (0.7, None),
            (0.1, 2),
            (0.1, 3),
            (0.1, 4),
        ]),
        'min_samples_leaf': hp.choice(name + '_minSamplesLeaf', [
            1,
            hyperopt.pyll.scope.int(
                hp.qloguniform(name + '_minSamplesLeaf_gt1',
                               numpy.log(1.5),
                               numpy.log(50.5),
                               1))
        ]),
        'bootstrap': hp.choice(name + '_bootstrap', [True, False]),
        'class_weight': hp.choice(name + '_classWeight',
                                  ['balanced', 'balanced_subsample',
                                   None]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(ExtraTreesClassifier)(**params)


##########
# Gaussian Processes
##########


def hyper_gpc_classifier(name, **kwargs):
    # name = 'gpc'
    params = {
        'n_restarts_optimizer': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_nRestarts', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
        'max_iter_predict': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_maxIterPredict', numpy.log(10.5),
                           numpy.log(1000.5), 1)),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(GaussianProcessClassifier)(**params)


##########
# Generalized Linear Models
##########


def hyper_passaggr_classifier(name, **kwargs):
    # name = 'passaggr'
    params = {
        'C': hp.lognormal(
            name + '_C',
            numpy.log(0.01),
            numpy.log(10),
        ),
        'loss': hp.choice(
            name + '_loss',
            ['hinge', 'squared_hinge']),
        'max_iter': hyperopt.pyll.scope.int(hp.quniform(name + '_maxIter',
                                                        50, 3000, 50)),
        'tol': hp.loguniform(name + '_tol', numpy.log(5e-4),
                             numpy.log(1e-2)),
        'class_weight': hp.choice(name + '_classWeight',
                                  ['balanced', None]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(PassiveAggressiveClassifier)(**params)


def hyper_ridge_classifier(name, **kwargs):
    # name = 'ridge'
    params = {
        'alpha': hp.loguniform(
            name + '_alpha',
            numpy.log(1e-3),
            numpy.log(1e3)),
        'fit_intercept': hp.pchoice(
            name + '_fitIntercept',
            [(0.8, True), (0.2, False)]),
        'normalize': hp.pchoice(
            name + '_normalize',
            [(0.8, True), (0.2, False)]),
        'max_iter': hyperopt.pyll.scope.int(hp.quniform(name + '_maxIter',
                                                        50, 3000, 50)),
        'tol': hp.loguniform(name + '_tol', numpy.log(5e-4),
                             numpy.log(1e-2)),
        'class_weight': hp.choice(name + '_classWeight',
                                  ['balanced', None]),
        'solver': hp.choice(name + '_solver',
                            ['auto', 'svd', 'cholesky', 'lsqr',
                             'sparse_cg', 'sag', 'saga'])
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(RidgeClassifier)(**params)


def hyper_sgd_classifier(name, **kwargs):
    # name = 'sgdc'
    params = {
        'loss': hp.choice(name + '_loss',
                          ['hinge', 'log', 'modified_huber',
                           'squared_hinge', 'perceptron']),
        'penalty': hp.choice(name + '_penalty',
                             ['l2', 'l1', 'elasticnet']),
        'alpha': hp.loguniform(name + '_alpha', numpy.log(1e-6),
                               numpy.log(1e-1)),
        'l1_ratio': hp.uniform(name + '_l1_ratio', 0, 1),
        'fit_intercept': hp.choice(name + '_fit_intercept',
                                   [False, True]),
        'max_iter': hyperopt.pyll.scope.int(hp.quniform(name + '_maxIter',
                                                        50, 3000, 50)),
        'tol': hp.loguniform(name + '_tol', numpy.log(5e-4),
                             numpy.log(1e-2)),
        'shuffle': hp.choice(name + '_shuffle', [False, True]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(SGDClassifier)(**params)


##########
# Naive Bayes
##########


def hyper_multinb_classifier(name, **kwargs):
    # name = 'multinb'
    params = {
        'alpha': hp.quniform(name + '_alpha', 0, 1, 0.001),
        'fit_prior': hp.choice(name + '_fit_prior', [False, True]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(MultinomialNB)(**params)


def hyper_gaussnb_classifier(**kwargs):
    return hyperopt.pyll.Literal(GaussianNB)(**kwargs)


##########
# Nearest Neighbors
##########


def hyper_knn_classifier(name, **kwargs):
    # name = 'knn'
    metric_p = hp.pchoice(name + 'metric', [
        (0.55, ('euclidean', 2)),
        (0.15, ('manhattan', 1)),
        (0.15, ('chebyshev', 0)),
        (0.15, ('minkowski', hp.quniform(name + 'metric_p', 2.5, 5.5, 1))),
    ])
    params = {
        'metric': metric_p[0],
        'p': metric_p[1],
        'n_neighbors': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_numNghbrs', numpy.log(0.5),
                           numpy.log(50.5), 1)),
        'weights': hp.choice(name + '_weights', ['uniform', 'distance']),

    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(KNeighborsClassifier)(**params)


def hyper_radiusnbrs_classifier(name, **kwargs):
    # name = 'radiusnbrs'
    params = {
        'radius': hp.loguniform(name + '_radius', numpy.log(0.5),
                                numpy.log(5.5)),
        'weights': hp.choice(name + '_weights', ['uniform', 'distance']),
        'algorithm': hp.choice(name + '_algorithm',
                               ['auto', 'ball_tree', 'kd_tree']),
        'leaf_size': hp.quniform(name + '_leafSize', 5, 60, 5),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(RadiusNeighborsClassifier)(**params)


##########
# Neural network models
##########


# def hyper_mlpc_classifier(**kwargs):
#     name = 'mlpc'
#     num_layers = scope.int(
#         hp.qloguniform(
#             name + '_numLayers',
#             numpy.log(0.5),
#             numpy.log(10.5),
#             1.0))
#
#     def gen_layer_size(ii):
#         return scope.int(
#             hp.qloguniform(
#                 'layer_' + str(ii),
#                 numpy.log(2.5),
#                 numpy.log(301.0),
#                 5.0))
#
#     hidden_layer_sizes = scope.map(gen_layer_size,
#                                    scope.range(num_layers))
#
#     params = {
#         'hidden_layer_sizes': None,
#         'activation'        : hp.choice(name + '_activation',
#                                         ['identity', 'logistic', 'tanh',
#                                             'relu']),
#         'solver'            : hp.choice(name + '_solver',
#                                         ['lbfgs', 'sgd', 'adam']),
#         'alpha'             : hp.loguniform(name + '_alpha',
#                                             numpy.log(0.0001),
#                                             numpy.log(10.0)),
#         'learning_rate'     : hp.choice(name + '_learningRate',
#                                         ['constant', 'invscaling',
#                                             'adaptive']),
#         'max_iter'          : scope.int(
#             hp.qloguniform(
#                 name + '_maxIter',
#                 numpy.log(1),
#                 numpy.log(1000),
#                 q = 1,
#             )),
#     }
#     params.update(kwargs)
#     return Literal(MLPClassifier)(**params)


##########
# Support Vector Machines
##########


def _svm_hp_space(
    name_func,
    kernel,
    n_features=1,
    C=None,
    gamma=None,
    coef0=None,
    degree=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    cache_size=200.0,
    verbose=False,
    class_weight=None):
    """Generate SVM hyperparamters search space
    """
    if kernel in ['linear', 'rbf', 'sigmoid']:
        degree_ = 1
    else:
        degree_ = degree or hyperopt.pyll.scope.int(
            hp.quniform(name_func('degree'), 1.5, 6.5, 1))
    if kernel in ['linear']:
        gamma_ = 'auto'
    else:
        gamma_ = gamma or hp.loguniform(name_func('gamma'),
                                        numpy.log(1e-3),
                                        numpy.log(1e3))
        gamma_ /= n_features  # make gamma independent of n_features.

    if kernel in ['linear', 'rbf']:
        coef0 = 0.0

    if coef0 is None:
        if kernel == 'poly':
            coef0 = hp.pchoice(name_func('coef0'), [
                (0.3, 0),
                (0.7, gamma_ * hp.uniform(name_func('coef0val'), 0., 10.))
            ])
        elif kernel == 'sigmoid':
            coef0 = hp.pchoice(name_func('coef0'), [
                (0.3, 0),
                (0.7, gamma_ * hp.uniform(name_func('coef0val'), -10., 10.))
            ])
        else:
            coef0 = 0.0

    hp_space = dict(
        kernel=kernel,
        C=C or hp.loguniform(name_func('C'), numpy.log(1e-5),
                             numpy.log(1e5)),
        gamma=gamma_,
        coef0=coef0,
        degree=degree_,
        shrinking=shrinking or hp.choice(name_func('shrinking'),
                                         [False, True]),
        tol=tol or hp.loguniform(name_func('tol'), numpy.log(5e-4),
                                 numpy.log(1e-2)),
        max_iter=max_iter or hyperopt.pyll.scope.int(hp.quniform(name_func('maxIter'),
                                                                 50, 3000, 50)),
        verbose=verbose,
        class_weight=class_weight or hp.choice(name_func('class_weight'),
                                               [None, 'balanced']),
        cache_size=cache_size
    )
    return hp_space


def _random_state(name, random_state):
    """
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state
    """
    return random_state


def _svc_hp_space(name_func, random_state=None, probability=False):
    """Generate SVC specific hyperparamters
    """
    hp_space = dict(
        random_state=_random_state(name_func('rstate'), random_state),
        probability=probability
    )
    return hp_space


def hyper_svckernel_classifier(name, kernel, random_state=None,
                               probability=False,
                               **kwargs):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with a user specified kernel.

    See help(hpsklearn.components._svm_hp_space) for info on additional SVM
    arguments.
    """

    def _name(msg):
        return '%s_%s_%s' % (name, kernel, msg)

    hp_space = _svm_hp_space(_name, kernel=kernel, **kwargs)
    hp_space.update(_svc_hp_space(_name, random_state, probability))
    return hyperopt.pyll.Literal(SVC)(**hp_space)


def svc_linear(name, **kwargs):
    """Simply use the hyper_svc_classifier function with kernel fixed as
    linear to
    return an SVC object."""
    return hyper_svckernel_classifier(name, kernel='linear', **kwargs)


def svc_rbf(name, **kwargs):
    """Simply use the hyper_svc_classifier function with kernel fixed as rbf
    to return
    an SVC object."""
    return hyper_svckernel_classifier(name, kernel='rbf', **kwargs)


def svc_poly(name, **kwargs):
    """Simply use the hyper_svc_classifier function with kernel fixed as
    poly to
    return an SVC object.
    """
    return hyper_svckernel_classifier(name, kernel='poly', **kwargs)


def svc_sigmoid(name, **kwargs):
    """Simply use the hyper_svc_classifier function with kernel fixed as
    sigmoid to
    return an SVC object.
    """
    return hyper_svckernel_classifier(name, kernel='sigmoid', **kwargs)


def hyper_svc_classifier(name, kernels=None, **kwargs):
    # name = 'svc'
    if kernels is None:
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    svms = {
        'linear': partial(svc_linear, name=name),
        'rbf': partial(svc_rbf, name=name),
        'poly': partial(svc_poly, name=name),
        'sigmoid': partial(svc_sigmoid, name=name),
    }
    choices = [svms[kern](**kwargs) for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval


def hyper_linearsvc_classifier(name, **kwargs):
    # name = 'LinearSVC'
    loss_penalty_dual = hp.choice(name + '_lossPenaltyDual', [
        ('hinge', 'l2', True),
        ('squared_hinge', 'l2', True),
        ('squared_hinge', 'l1', False),
        ('squared_hinge', 'l2', False)
    ])

    params = {
        'C': hp.loguniform(name + '_C', numpy.log(1e-5),
                           numpy.log(10.)),
        'loss': loss_penalty_dual[0],
        'penalty': loss_penalty_dual[1],
        'dual': loss_penalty_dual[2],
        'tol': hp.loguniform(name + '_tol', numpy.log(5e-4),
                             numpy.log(1e-2)),
        'multi_class': hp.choice(name + '_multiclass',
                                 ['ovr', 'crammer_singer']),
        'intercept_scaling': hp.loguniform(name + '_intscaling',
                                           numpy.log(1e-1), numpy.log(50.)),
        'class_weight': hp.choice(name + '_classWeight',
                                  ['balanced', None]),
        'max_iter': hyperopt.pyll.scope.int(hp.quniform(name + '_maxIter',
                                                        50, 3000, 50)),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(LinearSVC)(**params)


# TODO: svm.NuSVC


##########
# Decision Trees
##########


def hyper_dcsntree_classifier(name, **kwargs):
    # name = 'dcsntree'
    params = {
        'criterion': hp.choice(
            name + '_criterion',
            ['gini', 'entropy']),
        'splitter': hp.pchoice(
            name + '_splitter',
            [(0.8, 'best'), (0.2, 'random')]),
        'max_depth': None,
        'min_samples_split': hyperopt.pyll.scope.int(hp.quniform(
            name + '_minSamplesSplit',
            2, 10, 1)),
        'max_features': hp.pchoice(
            name + '_maxFeatures', [
                (0.2, 'sqrt'),  # most common choice.
                (0.1, 'log2'),  # less common choice.
                (0.1, None),  # all features, less common choice.
                (0.6, hp.uniform(name + '_maxFeatures' + '_frac', 0., 1.))
            ]),
        'class_weight': hp.choice(
            name + '_classWeight',
            ['balanced', None]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(DecisionTreeClassifier)(**params)


##########
# XGBoost
##########

def hyper_xgboost_classifier(name, objective='binary:logistic', **kwargs):
    # name = 'xgboost'
    params = {
        'max_depth': hyperopt.pyll.scope.int(
            hp.uniform(name + '_maxDepth', 1, 11)),
        'learning_rate': hp.loguniform(
            name + '_learningRate',
            numpy.log(0.0001),
            numpy.log(0.5)) - 0.0001,
        'n_estimators': hyperopt.pyll.scope.int(
            hp.quniform(name + '_nEstimators', 100, 6000, 200)),
        'gamma': hp.loguniform(
            name + '_gamma', numpy.log(0.0001),
            numpy.log(5)) - 0.0001,
        'min_child_weight': hyperopt.pyll.scope.int(
            hp.loguniform(name + '_minChildWght', numpy.log(1),
                          numpy.log(100))),
        'subsample': hp.uniform(name + '_subsample', 0.5, 1),
        'colsample_bytree': hp.uniform(name + '_colSampleByTree', 0.5, 1),
        'colsample_bylevel': hp.uniform(name + '_colSampleByLevel', 0.5, 1),
        'reg_alpha': hp.loguniform(name + '_regAlpha',
                                   numpy.log(0.0001),
                                   numpy.log(1)) - 0.0001,
        'reg_lambda': hp.loguniform(name + '_regLambda', numpy.log(1),
                                    numpy.log(4)),
    }
    params.update(kwargs)
    params['objective'] = objective
    return hyperopt.pyll.Literal(XGBClassifier)(**params)
