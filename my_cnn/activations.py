"""Activation functions used inside ``Convolution2D`` and ``Dense`` layers.

Each activation caches what it needs in ``forward``. ``backward`` takes no arguments
and returns the local derivative, which the layer multiplies element-wise with the
upstream gradient.
"""

import numpy as np
from numpy.typing import ArrayLike


class ReLU:
    """ReLU activation: ``max(0, x)``.

    Attributes:
        input: Input from the last forward pass, needed for the derivative.
    """
    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Replace negative values with zero.

        Args:
            input: Pre-activation values, any shape.

        Returns:
            ``max(0, input)``, same shape as the input.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # kill negative values
        return np.maximum(0, input)

    def backward(self) -> np.ndarray:
        """Return the derivative of ReLU at the cached input.

        Returns:
            ``1`` where the input was positive, ``0`` elsewhere, same shape as the input.
        """
        # 1 where positive, 0 otherwise
        return (self.input > 0).astype(np.float32)


class Softmax:
    """Softmax activation over the last axis: turns logits into probabilities.

    Attributes:
        input: Logits from the last forward pass.
    """
    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Compute softmax for every row.

        The row maximum is subtracted before ``exp`` to avoid overflow.

        Args:
            input: Logits, shape ``(N, classes)``.

        Returns:
            Probabilities, shape ``(N, classes)``. Every row sums to 1.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # subtract max to prevent overflow
        max_vals = np.max(input, axis=1, keepdims=True)
        exps = np.exp(input - max_vals)
        # normalize so it sums to 1
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self) -> int:
        """Return 1, so the upstream gradient passes through unchanged.

        This is correct only together with ``losses.CrossEntropy``, whose
        ``backward`` already returns the combined softmax + cross-entropy
        gradient with respect to the logits.

        Returns:
            ``1``.
        """
        return 1


class Linear:
    """No activation (identity)."""
    def __init__(self) -> None:
        pass

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Return the input unchanged (as ``float32``).

        Args:
            input: Values, any shape.

        Returns:
            The same values as ``float32``.
        """
        return np.asarray(input, dtype=np.float32)

    def backward(self) -> int:
        """Return the derivative of the identity.

        Returns:
            ``1``.
        """
        return 1


def get_activation(name: str | None) -> "Activation":
    """Return an activation function by name.

    Args:
        name: ``'relu'`` or ``'softmax'`` (case-insensitive), or ``None``.

    Returns:
        A new activation object. ``None`` and unknown names give ``Linear``.
    """
    if name is None:
        return Linear()
    if name.lower() == 'relu':
        return ReLU()
    if name.lower() == 'softmax':
        return Softmax()
    return Linear()


# Any activation returned by get_activation
Activation = ReLU | Softmax | Linear
