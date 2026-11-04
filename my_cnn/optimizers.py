"""Optimizers that update layer weights using the gradients from backpropagation.

An optimizer gets the list of layers in ``lazy_load`` (called by ``Model.compile``)
and, after every backward pass, ``update`` changes the weights of every layer that
has a ``weight_gradient``.
"""

import numpy as np

from .layers import Layer


class SGD:
    """Stochastic Gradient Descent optimizer: ``w = w - lr * dL/dw``.

    Attributes:
        layers: Layers of the model, set by ``lazy_load``.
        learning_rate: Step size, set by ``lazy_load``.
    """
    def __init__(self) -> None:
        self.layers: list[Layer] | None = None
        self.learning_rate: np.float32 | None = None

    def lazy_load(self, layers: list[Layer], learning_rate: float) -> None:
        """Connect the optimizer to the model's layers.

        Args:
            layers: Layers of the model, in order.
            learning_rate: Step size.
        """
        self.layers = layers
        self.learning_rate = np.float32(learning_rate)

    def update(self) -> None:
        """Update weights and biases for trainable layers."""
        for layer in self.layers:
            # only layers with weights
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                # w = w - lr * dL/dw
                layer.weights -= self.learning_rate * layer.weight_gradient
                layer.biases -= self.learning_rate * layer.bias_gradient


class Momentum:
    """SGD with Momentum optimizer.

    Keeps a running sum of past gradients (velocity) and moves the weights along it:
    ``v = beta * v + dL/dw``, ``w = w - lr * v``. The velocity is stored on each layer
    as ``v_w`` and ``v_b``.

    Args:
        beta: How much of the previous velocity is kept at each step.

    Attributes:
        layers: Layers of the model, set by ``lazy_load``.
        learning_rate: Step size, set by ``lazy_load``.
        beta: Momentum factor.
    """
    def __init__(self, beta: float = 0.9) -> None:
        self.layers: list[Layer] | None = None
        self.learning_rate: np.float32 | None = None
        self.beta: np.float32 = np.float32(beta)

    def lazy_load(self, layers: list[Layer], learning_rate: float) -> None:
        """Connect the optimizer to the model's layers.

        Args:
            layers: Layers of the model, in order.
            learning_rate: Step size.
        """
        self.layers = layers
        self.learning_rate = np.float32(learning_rate)

    def update(self) -> None:
        """Update weights and biases with momentum.

        The velocity of a layer is created (as zeros) on its first update.
        """
        for layer in self.layers:
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                if not hasattr(layer, 'v_w') or layer.v_w is None:
                    layer.v_w = np.zeros_like(layer.weights)
                    layer.v_b = np.zeros_like(layer.biases)

                # v = beta * v + grad
                layer.v_w = self.beta * layer.v_w + layer.weight_gradient
                layer.v_b = self.beta * layer.v_b + layer.bias_gradient

                # w = w - lr * v
                layer.weights -= self.learning_rate * layer.v_w
                layer.biases -= self.learning_rate * layer.v_b


def get_optimizer(name: str | None) -> "Optimizer":
    """Return an optimizer by name.

    Args:
        name: ``'sgd'``, ``'momentum'`` or ``'sgd_momentum'`` (case-insensitive), or ``None``.

    Returns:
        A new optimizer object. ``None`` and unknown names give ``SGD``.
    """
    if name is None:
        return SGD()
    name_str = str(name).lower()
    if name_str == 'sgd':
        return SGD()
    elif name_str in ('momentum', 'sgd_momentum'):
        return Momentum()
    else:
        return SGD()


# Any optimizer returned by get_optimizer
Optimizer = SGD | Momentum
