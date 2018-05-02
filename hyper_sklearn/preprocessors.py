import numpy
from hyperopt import hp
import hyperopt.pyll
from sklearn.decomposition import PCA
import sklearn.feature_extraction.text


def hyper_tfidf_vectorizer_preprocessor(name, **kwargs):
    # name = 'tfidf'
    min_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_minNgram',
        1, 6, 1))
    max_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_maxNgram',
        0, 6, 1))
    params = {
        'ngram_range': (min_ngram, min_ngram + max_ngram),
        'analyzer': hp.choice(name + '_analyzer',
                              ['word', 'char', 'char_wb']),
        'stop_words': hp.choice(name + '_stopWords', ['english', None]),
        'lowercase': hp.choice(name + '_lowercase', [False, True]),
        'binary': hp.choice(name + '_binary', [False, True]),
        'norm': hp.choice(name + '_norm', ['l1', 'l2', None]),
        'use_idf': hp.choice(name + '_useIdf', [False, True]),
        'sublinear_tf': hp.choice(name + '_sublinearTf', [False, True]),
        'max_features': hyperopt.pyll.scope.int(hp.choice(name + '_maxFeatures',
                                                          [None,
                                                           hp.quniform(
                                                               name + '_maxFeatures',
                                                               2 ** 6,
                                                               2 ** 20,
                                                               2 ** 10)])),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.feature_extraction.text.TfidfVectorizer)(**params)


def hyper_hashing_vectorizer_preprocessor(name, **kwargs):
    # name = 'hashing'
    min_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_minNgram',
        1, 6, 1))
    max_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_maxNgram',
        0, 6, 1))
    params = {
        'ngram_range': (min_ngram, min_ngram + max_ngram),
        'analyzer': hp.choice(name + '_analyzer',
                              ['word', 'char', 'char_wb']),
        'stop_words': hp.choice(name + '_stopWords', ['english', None]),
        'lowercase': hp.choice(name + '_lowercase', [False, True]),
        'binary': hp.choice(name + '_binary', [False, True]),
        'norm': hp.choice(name + '_norm', ['l1', 'l2', None]),
        'alternate_sign': hp.choice(name + '_alternateSign', [False, True]),
        'n_features': hyperopt.pyll.scope.int(
            hp.qloguniform(name + '_maxFeatures',
                           numpy.log(1000),
                           numpy.log(2 ** 20),
                           2 ** 11)),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.feature_extraction.text.HashingVectorizer)(**params)


def hyper_tfidf_transformer_preprocessor(name, **kwargs):
    # name = 'tfidfTransfmr'
    params = {
        'norm': hp.choice(name + '_norm', ['l1', 'l2', None]),
        'use_idf': hp.choice(name + '_useIdf', [False, True]),
        'sublinear_tf': hp.choice(name + '_sublinearTf', [False, True]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.feature_extraction.text.TfidfTransformer)(**params)


def hyper_count_vectorizer_preprocessor(name, **kwargs):
    # name = 'hashing'
    min_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_minNgram',
        1, 6, 1))
    max_ngram = hyperopt.pyll.scope.int(hp.quniform(
        name + '_maxNgram',
        0, 6, 1))
    params = {
        'ngram_range': (min_ngram, min_ngram + max_ngram),
        'analyzer': hp.choice(name + '_analyzer',
                              ['word', 'char', 'char_wb']),
        'stop_words': hp.choice(name + '_stopWords', ['english', None]),
        'lowercase': hp.choice(name + '_lowercase', [False, True]),
        'binary': hp.choice(name + '_binary', [False, True]),
        'max_features': hyperopt.pyll.scope.int(hp.choice(name + '_maxFeatures',
                                                          [None,
                                                           hp.quniform(
                                                               name + '_maxFeatures',
                                                               2 ** 6,
                                                               2 ** 20,
                                                               2 ** 10)])),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(sklearn.feature_extraction.text.CountVectorizer)(**params)


def hyper_pca_preprocessor(name, **kwargs):
    # name = 'pca'
    params = {
        'n_components': hyperopt.pyll.scope.int(hp.qloguniform(
            name + '_ncomponents',
            low=numpy.log(0.51),
            high=numpy.log(30.5),
            q=1.0) * 4),
        'whiten': hp.choice(name + '_whiten', [False, True]),
    }
    params.update(kwargs)
    return hyperopt.pyll.Literal(PCA)(**params)
