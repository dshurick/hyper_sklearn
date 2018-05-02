import hyperopt.pyll
import sklearn


def hyper_pipeline(*args, **kwargs):
    return hyperopt.pyll.Literal(sklearn.pipeline.make_pipeline)(*args,
                                                                 **kwargs)


def hyper_union(*args, **kwargs):
    return hyperopt.pyll.Literal(sklearn.pipeline.make_union)(*args,
                                                              **kwargs)
