"""Sequential model: a stack of layers trained with mini-batch gradient descent."""

import pickle

import numpy as np
from numpy.typing import ArrayLike

from . import losses, optimizers
from .layers import Layer


class Model:
    """Sequential neural network model.

    Layers are run in the order they were added. The API follows Keras:
    ``add`` → ``compile`` → ``fit`` → ``evaluate`` / ``predict`` → ``save``.

    Attributes:
        layers: Layers of the model, in order.
        optimizer: Optimizer set by ``compile``.
        loss: Loss function set by ``compile``.
    """
    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self.optimizer: optimizers.Optimizer | None = None
        self.loss: losses.CrossEntropy | None = None
        pass

    def add(self, layer: Layer) -> None:
        """Append a layer to the end of the model.

        Args:
            layer: Layer object, e.g. ``layers.Dense(64, activation_func='relu')``.
        """
        self.layers.append(layer)

    def compile(self, optimizer: str | None, loss: str | None, learning_rate: float = 0.001) -> None:
        """Choose the optimizer and the loss function.

        Must be called after all layers are added and before ``fit``.

        Args:
            optimizer: ``'sgd'`` or ``'momentum'`` (see ``optimizers.get_optimizer``).
            loss: ``'crossentropy'`` (see ``losses.get_loss``).
            learning_rate: Step size for the optimizer.
        """
        self.optimizer = optimizers.get_optimizer(optimizer)
        self.optimizer.lazy_load(self.layers, learning_rate)
        self.loss = losses.get_loss(loss)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Run the input through all layers (forward pass).

        The whole input is processed at once. Layers also cache their inputs,
        which ``back_prop`` needs during training.

        Args:
            X: Batch of samples, e.g. images of shape ``(N, 28, 28, 1)``.

        Returns:
            Output of the last layer, e.g. class probabilities of shape ``(N, 10)``.
        """
        # run through all layers
        out = self.layers[0].forward(np.asarray(X, dtype=np.float32))
        for i in range(1, len(self.layers)):
             out = self.layers[i].forward(out)
        return out

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through all layers.

        Goes from the last layer to the first. Each layer stores its own weight
        gradients along the way. Must be called after ``predict`` on the same batch.

        Args:
            grad: dL/d(output of the last layer), e.g. from ``loss.backward``.

        Returns:
            dL/d(model input).
        """
        for i in range(len(self.layers)-1, -1, -1):
           grad = self.layers[i].backward(grad)
        return grad

    def save(self, filepath: str = "model.pkl") -> None:
        """Save the model to a pickle file.

        Before saving, the cached data from the last batch (inputs, patches, gradients,
        momentum velocities) is cleared on every layer to keep the file small.

        Args:
            filepath: Path of the output file.
        """
        # clear cache before saving to keep file size small
        for layer in self.layers:
            if hasattr(layer, 'input'):
                layer.input = None
            if hasattr(layer, 'activation_func') and hasattr(layer.activation_func, 'input'):
                layer.activation_func.input = None
            if hasattr(layer, 'weight_gradient'):
                layer.weight_gradient = None
            if hasattr(layer, 'bias_gradient'):
                layer.bias_gradient = None
            if hasattr(layer, 'v_w'):
                layer.v_w = None
            if hasattr(layer, 'v_b'):
                layer.v_b = None
            if hasattr(layer, 'patches'):
                layer.patches = None
            if hasattr(layer, 'max_indices'):
                layer.max_indices = None
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = "model.pkl") -> "Model":
        """Load a model saved with ``save``.

        Only load files you trust: unpickling can run arbitrary code.

        Args:
            filepath: Path of the saved model.

        Returns:
            The loaded model, ready for ``predict`` and ``evaluate``.
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {filepath}")
        return model

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        epochs: int = 3,
        batch_size: int = 32,
        shuffle: bool = True,
        print_every: int = 50,
    ) -> None:
        """Train the model with mini-batch gradient descent.

        For every batch: forward pass, loss, backward pass, weight update.
        Prints the running loss and accuracy for the current epoch.

        Args:
            X: Training samples, e.g. images of shape ``(N, 28, 28, 1)``.
            y: One-hot labels, shape ``(N, classes)``.
            epochs: Number of passes over the whole training set.
            batch_size: Number of samples per weight update.
            shuffle: Shuffle the samples at the start of every epoch.
            print_every: Print progress every this many batches (and after the last batch).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        samples = X.shape[0]
        total_batches = (samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0

            # shuffle indices at the start of each epoch
            if shuffle:
                indices = np.random.permutation(samples)
            else:
                indices = np.arange(samples)

            # go through data in batches
            for batch_idx, i in enumerate(range(0, samples, batch_size), 1):
                batch_indices = indices[i : i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # forward
                y_pred = self.predict(X_batch)

                # loss
                loss_val = self.loss.forward(y_batch, y_pred)

                total_loss += loss_val * len(X_batch)

                # count correct
                preds = np.argmax(y_pred, axis=1)
                trues = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == trues)

                # backward
                start_grad = self.loss.backward(y_batch, y_pred, len(X_batch))

                self.back_prop(start_grad)

                # update weights
                self.optimizer.update()

                # show progress every print_every batches and on the final batch
                if batch_idx % print_every == 0 or batch_idx == total_batches:
                    processed = i + len(X_batch)
                    curr_loss = total_loss / processed
                    curr_acc = correct_predictions / processed
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {curr_loss:.4f} | Acc: {curr_acc:.4f}", end='', flush=True)

            print()

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> tuple[np.float32, np.float64]:
        """Compute the loss and accuracy on a dataset.

        The whole dataset is processed in one forward pass.

        Args:
            X: Samples, e.g. images of shape ``(N, 28, 28, 1)``.
            y: One-hot labels, shape ``(N, classes)``.

        Returns:
            ``(loss, accuracy)``, where accuracy is the fraction of correct predictions (0–1).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X)

        loss_val = self.loss.forward(y, y_pred)

        preds = np.argmax(y_pred, axis=1)
        trues = np.argmax(y, axis=1)
        acc = np.sum(preds==trues) / len(y)

        return loss_val, acc

