def print_coefs(feature_names, lr):
    for feature, coef in zip(feature_names, lr.coef_[0]):
        print('%s - %.3f' % (feature, coef))
