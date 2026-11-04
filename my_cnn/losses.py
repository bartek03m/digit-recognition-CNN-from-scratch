"""Loss functions."""

import numpy as np
from numpy.typing import ArrayLike


class CrossEntropy:
    """Cross-entropy loss for classification.

    Expects one-hot labels and probabilities from a ``softmax`` output layer.
    """
    def __init__(self) -> None:
        pass

    def forward(self, y_true: ArrayLike, y_pred: ArrayLike) -> np.float32:
        """Compute the mean cross-entropy over the batch.

        A small epsilon (1e-15) is added to the predictions to avoid ``log(0)``.

        Args:
            y_true: One-hot labels, shape ``(N, classes)``.
            y_pred: Predicted probabilities, shape ``(N, classes)``.

        Returns:
            Average loss per sample.
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        # -sum(y * log(p)), epsilon to avoid log(0)
        total_loss = -np.sum(y_true * np.log(y_pred + np.float32(1e-15)))
        batch_size = y_true.shape[0]
        # average loss over batch
        return np.float32(total_loss / batch_size)

    def backward(self, y_true: ArrayLike, y_pred: ArrayLike, batch_size: int) -> np.ndarray:
        """Return the gradient of softmax + cross-entropy with respect to the logits.

        Uses the shortcut ``(y_pred - y_true) / N``, which already includes the
        softmax derivative. That is why ``activations.Softmax.backward`` returns 1.

        Args:
            y_true: One-hot labels, shape ``(N, classes)``.
            y_pred: Predicted probabilities, shape ``(N, classes)``.
            batch_size: Number of samples ``N`` in the batch.

        Returns:
            dL/d(logits), shape ``(N, classes)``.
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        # shortcut gradient for softmax + cross-entropy
        return (y_pred - y_true) / np.float32(batch_size)


def get_loss(name: str | None) -> CrossEntropy:
    """Return a loss function by name.

    Args:
        name: ``'crossentropy'`` (case-insensitive) or ``None``.

    Returns:
        A new ``CrossEntropy`` object. ``None`` and unknown names also give ``CrossEntropy``.
    """
    if name is None:
        return CrossEntropy()
    elif name.lower() == 'crossentropy':
        return CrossEntropy()
    else:
        return CrossEntropy()
