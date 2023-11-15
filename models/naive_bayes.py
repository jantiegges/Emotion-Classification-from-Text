import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y, alpha_prior=10, alpha_likelihood=1.0):
        n_classes = len(np.unique(y))
        # compute the log prior for each class
        class_counts = np.bincount(y)
        self.class_log_prior_ = np.log((class_counts + alpha_prior) / (class_counts.sum() + n_classes * alpha_prior))

        # calc the log likelihood for each feature in each class
        feature_counts_per_class = np.zeros((n_classes, X.shape[1]))
        for class_label in range(n_classes):
            feature_counts_per_class[class_label] = X[y == class_label].sum(axis=0)
        self.feature_log_prob_ = np.log((feature_counts_per_class + alpha_likelihood) /
                                  (feature_counts_per_class.sum(axis=1, keepdims=True) + X.shape[1] * alpha_likelihood))

    def predict(self, X):
        # calc the log likelihood for each class in each sample
        log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_
        return log_likelihood.argmax(axis=1)
    
    def evaluate_acc(self, y, y_pred):
        return (y == y_pred).mean()