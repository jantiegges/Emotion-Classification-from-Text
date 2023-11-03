import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha # smoothing (1.0 for Laplace smoothing)
        self.fit_prior = fit_prior

        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        # compute the log prior for each class
        if self.fit_prior:
            class_counts = np.bincount(y)
            self.class_log_prior_ = np.log((class_counts + self.alpha) / (class_counts.sum() + n_classes * self.alpha))
        else:
            self.class_log_prior_ = np.log(np.full(n_classes, 1 / n_classes))

        # calc the log likelihood for each feature in each class
        feature_counts_per_class = np.zeros((n_classes, X.shape[1]))
        for class_label in range(n_classes):
            feature_counts_per_class[class_label] = X[y == class_label].sum(axis=0)
        self.feature_log_prob_ = np.log((feature_counts_per_class + self.alpha) / 
                                  (feature_counts_per_class.sum(axis=1, keepdims=True) + X.shape[1] * self.alpha))

    def predict(self, X):
        # calc the log likelihood for each class in each sample
        log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_
        return log_likelihood.argmax(axis=1)
    
    def evaluate_acc(self, y, y_pred):
        return (y == y_pred).mean()