import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from .kernel import product_kernel


class ProductSpacePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, pm, max_epochs=1_000, patience=5, weights=None):
        self.pm = pm  # ProductManifold instance
        self.max_epochs = max_epochs
        self.patience = patience  # Number of consecutive epochs without improvement to consider convergence
        self.classes_ = None
        self.classifiers_ = {}  # Dictionary to store classifiers for one-vs-rest approach
        if weights is None:
            self.weights = torch.ones(len(pm.P), dtype=torch.float32)
        else:
            assert len(weights) == len(pm.P), "Number of weights must match the number of manifolds."
            self.weights = weights

    # # def fit(self, X, y):
    #     # Identify unique classes for multiclass classification
    #     self.classes_ = torch.unique(y).tolist()

    #     # Precompute kernel matrix
    #     Ks, _ = product_kernel(self.pm, X, None)
    #     K = torch.ones(X.shape[0], X.shape[0], dtype=X.dtype, device=X.device)
    #     for K_m, w in zip(Ks, self.weights):
    #         K += w * K_m

    #     # Relabel y to -1 and 1 for binary classification per class
    #     for class_label in self.classes_:
    #         # Binary classification shortcut
    #         if len(self.classes_) == 2 and class_label == self.classes_[1]:
    #             self.classifiers_[class_label] = -1 * self.classifiers_[self.classes_[0]]

    #         else:
    #             binary_y = torch.where(y == class_label, 1, -1)  # One-vs-rest relabeling

    #             # Initialize decision function g for this binary classifier
    #             g = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)

    #             n_epochs = 0
    #             i = 0

    #             while n_epochs < self.max_epochs:
    #                 if torch.sign(X[i] @ g) != binary_y[i]:
    #                     g += binary_y[i] * K[i] @ X
    #                 i = (i + 1) % X.shape[0]

    #                 n_epochs += 1

    #             # Store the classifier (decision function) for the current class
    #             self.classifiers_[class_label] = g

    #     return self

    def fit(self, X, y):
        # Identify unique classes for multiclass classification
        self.classes_ = torch.unique(y).tolist()
        n_samples = X.shape[0]

        # Precompute kernel matrix
        Ks, _ = product_kernel(self.pm, X, None)
        K = torch.ones((n_samples, n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights):
            K += w * K_m
        # K = X @ X.T

        # Store training data and labels for prediction
        self.X_train_ = X
        self.y_train_ = y

        # Initialize dictionary to store alpha coefficients for each class
        self.alpha = {}

        for class_label in self.classes_:
            # One-vs-rest labels
            y_binary = torch.where(y == class_label, 1, -1)  # Shape: (n_samples,)

            # Initialize alpha coefficients for this class
            alpha = torch.zeros(n_samples, dtype=X.dtype, device=X.device)

            for epoch in range(self.max_epochs):
                # Compute decision function: f = K @ (alpha * y_binary)
                f = K @ (alpha * y_binary)  # Shape: (n_samples,)

                # Compute predictions
                predictions = torch.sign(f)

                # Find misclassified samples
                misclassified = predictions != y_binary

                # If no misclassifications, break early
                if not misclassified.any():
                    break

                # Update alpha coefficients for misclassified samples
                alpha[misclassified] += 1

            # Store the alpha coefficients for the current class
            self.alpha[class_label] = alpha

        return self

    # def predict_proba(self, X):
    #     # Initialize matrix to store decision values for each class
    #     decision_values = torch.zeros((X.shape[0], len(self.classes_)), dtype=X.dtype, device=X.device)

    #     # Compute decision values for each classifier
    #     for idx, class_label in enumerate(self.classes_):
    #         g = self.classifiers_[class_label]
    #         decision_values[:, idx] = X @ g

    #     return decision_values

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decision_values = torch.zeros((n_samples, n_classes), dtype=X.dtype, device=X.device)

        # Compute kernel matrix between training data and test data
        Ks, _ = product_kernel(self.pm, self.X_train_, X)
        K_test = torch.ones((self.X_train_.shape[0], n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights):
            K_test += w * K_m
        # K_test = self.X_train_ @ X.T

        for idx, class_label in enumerate(self.classes_):
            alpha = self.alpha[class_label]  # Shape: (n_samples_train,)
            y_binary = torch.where(self.y_train_ == class_label, 1, -1)  # Shape: (n_samples_train,)

            # Compute decision function for test samples
            f = (alpha * y_binary) @ K_test  # Shape: (n_samples_test,)
            decision_values[:, idx] = f

        return decision_values

    def predict(self, X):
        decision_values = self.predict_proba(X)
        # Return the class with the highest decision value
        argmax_idx = torch.argmax(decision_values, dim=1)
        return torch.tensor([self.classes_[i] for i in argmax_idx])
