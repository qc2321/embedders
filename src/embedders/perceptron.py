import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class ProductSpacePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, pm, max_epochs=1000, patience=5):
        self.pm = pm  # ProductManifold instance
        self.max_epochs = max_epochs
        self.patience = patience  # Number of consecutive epochs without improvement to consider convergence
        self.classes_ = None
        self.classifiers_ = {}  # Dictionary to store classifiers for one-vs-rest approach
        self.R = []  # To store maximum radius for each hyperbolic manifold

    def fit(self, X, y):
        # Identify unique classes for multiclass classification
        self.classes_ = torch.unique(y).tolist()

        # Compute maximum radius for each manifold.
        self.R = [M.scale for M in self.pm.P]

        # Relabel y to -1 and 1 for binary classification per class
        for class_label in self.classes_:
            # Binary classification shortcut
            if len(self.classes_) == 2 and class_label == self.classes_[1]:
                self.classifiers_[class_label] = -1 * self.classifiers_[self.classes_[0]]

            else:
                binary_y = torch.where(y == class_label, 1, -1)  # One-vs-rest relabeling

                # Initialize decision function g for this binary classifier
                g = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)

                n_epochs = 0
                epochs_without_improvement = 0  # Track consecutive epochs without improvement
                best_error_count = float("inf")  # Best error count seen so far

                while n_epochs < self.max_epochs:
                    errors = 0
                    for n in range(X.shape[0]):
                        # Compute the decision function value for the current point
                        decision_value = g @ X[n]

                        # Check if the point is misclassified
                        if torch.sign(decision_value) != binary_y[n]:
                            # Calculate the kernel K(x, x_n) for the current point x_n
                            K = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)  # Start with the bias term

                            for i, (M, x) in enumerate(zip(self.pm.P, self.pm.factorize(X))):
                                # Compute kernel matrix between x[n:n+1] and all training points
                                if M.type == "E":
                                    K += M.scale * M.manifold.inner(x[n : n + 1], x)  # Kernel matrix for Euclidean
                                elif M.type == "S":
                                    K += M.scale * torch.asin(
                                        torch.clamp(M.manifold.inner(x[n : n + 1], x), -1, 1)
                                    )  # Kernel matrix for Spherical
                                elif M.type == "H":
                                    K += M.scale * torch.asin(
                                        torch.clamp((self.R[i] ** -2) * M.manifold.inner(x[n : n + 1], x), -1, 1)
                                    )  # Kernel matrix for Hyperbolic

                            # Update decision function using the computed kernel
                            g += binary_y[n] * X[n]  # Update with current point only
                            errors += 1  # Track the number of errors in this epoch

                    # Convergence check based on error improvement
                    if errors < best_error_count:
                        best_error_count = errors
                        epochs_without_improvement = 0  # Reset the counter if we have an improvement
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= self.patience:
                        # print(f"Converged for class {class_label} after {n_epochs} epochs (no improvement).")
                        break

                    n_epochs += 1

                # Store the classifier (decision function) for the current class
                self.classifiers_[class_label] = g

        return self

    def predict(self, X):
        # Initialize matrix to store decision values for each class
        decision_values = torch.zeros((X.shape[0], len(self.classes_)), dtype=X.dtype, device=X.device)

        # Compute decision values for each classifier
        for idx, class_label in enumerate(self.classes_):
            g = self.classifiers_[class_label]
            decision_values[:, idx] = X @ g

        # Return the class with the highest decision value
        # print(decision_values)
        argmax_idx = torch.argmax(decision_values, dim=1)
        return torch.tensor([self.classes_[i] for i in argmax_idx])
