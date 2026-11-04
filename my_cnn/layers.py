"""Network layers: input, rescaling, convolution, max pooling, flatten and dense.

Every layer has the same interface:

- ``forward(input)`` computes the output and caches what the backward pass needs,
- ``backward(upstream_gradient)`` takes dL/d(output) and returns dL/d(input).

Layers with weights (``Convolution2D``, ``Dense``) also store ``weight_gradient``
and ``bias_gradient`` after ``backward``, which the optimizer then uses.

Images use the channels-last layout ``(N, H, W, C)``: batch size, height, width, channels.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike

from . import activations


class Input:
    """Input layer, passes data through unchanged.

    Only converts the data to ``float32``. The shape is stored for reference, it is not checked.

    Args:
        shape: Shape of a single sample without the batch dimension, e.g. ``(28, 28, 1)``.
    """
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Return the input as a ``float32`` array.

        Args:
            input: Batch of samples.

        Returns:
            The same data as ``float32``.
        """
        return np.asarray(input, dtype=np.float32)

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Pass the gradient through unchanged.

        Args:
            upstream_gradient: dL/d(output).

        Returns:
            dL/d(input), equal to the upstream gradient.
        """
        return np.asarray(upstream_gradient, dtype=np.float32)


class Rescaling:
    """Scales input by a constant factor.

    Typically used as ``Rescaling(1 / 255)`` to map pixel values from 0–255 to 0–1.

    Args:
        scale: Factor that every input value is multiplied by.
    """
    def __init__(self, scale: float) -> None:
        self.scale: float = scale

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Multiply the input by ``scale``.

        Args:
            input: Batch of samples, any shape.

        Returns:
            ``input * scale`` as ``float32``, same shape as the input.
        """
        return np.asarray(input, dtype=np.float32) * np.float32(self.scale)

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Scale the gradient by the same factor (d(x * s)/dx = s).

        Args:
            upstream_gradient: dL/d(output).

        Returns:
            dL/d(input), same shape as the upstream gradient.
        """
        return np.asarray(upstream_gradient, dtype=np.float32) * np.float32(self.scale)


class Convolution2D:
    """2D convolution layer with same-padding.

    The input is zero-padded by ``kernel_size // 2`` on each side, so with ``strides=1``
    the output has the same height and width as the input. The forward pass turns all
    patches into rows of one matrix (im2col) and computes the convolution as a single
    matrix multiplication. Weights are created on the first ``forward`` call, when the
    number of input channels is known (He normal initialization, zero biases).

    Args:
        no_of_filters: Number of filters, i.e. output channels ``F``.
        kernel_size: Height and width of each filter ``k``.
        strides: Step between neighboring patches, the same in both directions.
        activation_func: Name of the activation (``'relu'``, ``'softmax'``) or ``None`` for no activation.

    Attributes:
        weights: Filters, shape ``(F, k, k, C)``. ``None`` until the first forward pass.
        biases: One bias per filter, shape ``(F,)``.
        input: Input from the last forward pass, shape ``(N, H, W, C)``.
        patches: im2col matrix from the last forward pass, shape ``(N * H_out * W_out, k * k * C)``.
        output_shape: ``(H_out, W_out)`` from the last forward pass.
        weight_gradient: dL/dW from the last backward pass, same shape as ``weights``.
        bias_gradient: dL/db from the last backward pass, same shape as ``biases``.
    """
    def __init__(
            self,
            no_of_filters: int,
            kernel_size: int,
            strides: int = 1,
            activation_func: str | None = None,
        ) -> None:
            self.no_of_filters: int = no_of_filters
            self.kernel_size: int = kernel_size
            self.strides: int = strides
            self.activation_func: activations.Activation = activations.get_activation(activation_func)
            self.weights: np.ndarray | None = None
            self.biases: np.ndarray | None = None
            self.input: np.ndarray | None = None
            self.weight_gradient: np.ndarray | None = None
            self.bias_gradient: np.ndarray | None = None
            self.patches: np.ndarray | None = None
            self.output_shape: tuple[int, int] | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Convolve the input with all filters and apply the activation.

        Args:
            input: Batch of images, shape ``(N, H, W, C)``.

        Returns:
            Feature maps, shape ``(N, H_out, W_out, F)``. With ``strides=1``
            ``H_out = H`` and ``W_out = W``.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape

        # init weights on first call with He (Kaiming) normal
        if self.weights is None:
            fan_in = self.kernel_size * self.kernel_size * input_channels
            std = np.sqrt(2.0 / fan_in)
            self.weights = np.random.normal(
                0.0,
                std,
                (self.no_of_filters, self.kernel_size, self.kernel_size, input_channels),
            ).astype(np.float32)
            self.biases = np.zeros(self.no_of_filters, dtype=np.float32)

        # zero-pad edges so output keeps the same size
        pad = self.kernel_size // 2
        input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)))

        # extract all kernel-sized patches
        windows = sliding_window_view(input, (self.kernel_size, self.kernel_size), axis=(1,2))

        # skip windows according to stride
        windows = windows[:, ::self.strides, ::self.strides, :, :, :]

        # rearrange to [N, H_out, W_out, kH, kW, C]
        windows = windows.transpose(0, 1, 2, 4, 5, 3)

        output_height, output_width = windows.shape[1], windows.shape[2]

        # flatten patches into rows: (N*H_out*W_out, kH*kW*C)
        patches = windows.reshape(batch_size * output_height * output_width, -1)
        self.patches = patches
        self.output_shape = (output_height, output_width)
        # flatten filters into rows: (num_filters, kH*kW*C)
        weights_flat = self.weights.reshape(self.no_of_filters, -1)

        # big matrix multiply
        output = patches @ weights_flat.T + self.biases

        # put it back into image shape
        output = output.reshape(batch_size, output_height, output_width, self.no_of_filters)

        # run through activation
        output = self.activation_func.forward(output)
        return output

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Compute gradients for the weights, biases and the layer input.

        Stores dL/dW in ``weight_gradient`` and dL/db in ``bias_gradient``.
        The input gradient is built by adding each patch gradient back to the
        input positions it came from (col2im). Overlapping patches sum up.

        Args:
            upstream_gradient: dL/d(output), shape ``(N, H_out, W_out, F)``.

        Returns:
            dL/d(input), shape ``(N, H, W, C)``.
        """
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        batch_size, input_height, input_width, input_channels = self.input.shape
        # undo activation
        upstream_gradient *= self.activation_func.backward()

        output_height, output_width = self.output_shape

        # dL/db = sum of gradients over all positions
        self.bias_gradient = np.sum(upstream_gradient, axis=(0, 1, 2))

        # reuse cached patches from forward pass
        patches = self.patches
        upstream_flat = upstream_gradient.reshape(batch_size * output_height * output_width, self.no_of_filters)

        # dL/dW = gradient^T @ patches
        self.weight_gradient = (upstream_flat.T @ patches).reshape(self.weights.shape)

        # dL/dX - gradient going to the previous layer (col2im via kernel loop)
        weights_flat = self.weights.reshape(self.no_of_filters, -1)
        grad_cols = (upstream_flat @ weights_flat).reshape(
            batch_size, output_height, output_width, self.kernel_size, self.kernel_size, input_channels
        )

        pad = self.kernel_size // 2
        padded_h = input_height + 2 * pad
        padded_w = input_width + 2 * pad
        downstream_padded = np.zeros((batch_size, padded_h, padded_w, input_channels), dtype=np.float32)

        # scatter patch gradients back using slice-based += (only kernel_size^2 iterations)
        for kh in range(self.kernel_size):
            for kw in range(self.kernel_size):
                downstream_padded[:, kh : kh + output_height * self.strides : self.strides,
                                     kw : kw + output_width  * self.strides : self.strides, :] += grad_cols[:, :, :, kh, kw, :]

        # strip padding
        if pad > 0:
            return downstream_padded[:, pad : -pad, pad : -pad, :]
        return downstream_padded


class MaxPooling2D:
    """2D max pooling layer.

    Uses non-overlapping ``pool_size x pool_size`` windows (the stride equals the pool size)
    and keeps the largest value of each window. If the height or width is not divisible
    by ``pool_size``, the leftover rows and columns are dropped.

    Args:
        pool_size: Height and width of each pooling window.

    Attributes:
        input: Input from the last forward pass, shape ``(N, H, W, C)``.
        max_indices: Position of the maximum inside each window, flattened to
            ``0 .. pool_size**2 - 1``, shape ``(N, H_out, W_out, C)``.
        output_shape: ``(H_out, W_out)`` from the last forward pass.
    """
    def __init__(self, pool_size: int) -> None:
        self.pool_size: int = pool_size
        self.input: np.ndarray | None = None
        self.max_indices: np.ndarray | None = None
        self.output_shape: tuple[int, int] | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Take the maximum of every pooling window.

        Args:
            input: Batch of feature maps, shape ``(N, H, W, C)``.

        Returns:
            Pooled feature maps, shape ``(N, H // pool_size, W // pool_size, C)``.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # split into pool_size x pool_size windows
        windows = sliding_window_view(input, (self.pool_size, self.pool_size), axis=(1, 2))

        # non-overlapping - step by pool_size
        windows = windows[:, ::self.pool_size, ::self.pool_size, :, :, :]

        # pick the max from each window
        flat_windows = windows.reshape(*windows.shape[:4], -1)
        self.max_indices = np.argmax(flat_windows, axis=4)  # (N, out_h, out_w, C)
        self.output_shape = (windows.shape[1], windows.shape[2])
        output = np.max(flat_windows, axis=4)
        return output

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Send each gradient only to the position of its window's maximum.

        All other input positions get a gradient of zero.

        Args:
            upstream_gradient: dL/d(output), shape ``(N, H_out, W_out, C)``.

        Returns:
            dL/d(input), shape ``(N, H, W, C)``.
        """
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        batch_size, input_height, input_width, input_channels = self.input.shape
        downstream_gradient = np.zeros_like(self.input)

        output_height, output_width = self.output_shape

        # convert cached flat index back to 2D row/col offset within the pool window
        max_i = self.max_indices // self.pool_size  # row offset
        max_j = self.max_indices % self.pool_size   # col offset

        # build index arrays with broadcasting-friendly shapes
        n_idx = np.arange(batch_size)[:, None, None, None]       # batch axis
        oh_idx = np.arange(output_height)[None, :, None, None]   # output row axis
        ow_idx = np.arange(output_width)[None, None, :, None]    # output col axis
        c_idx = np.arange(input_channels)[None, None, None, :]   # channel axis

        # actual position in the full input
        abs_i = oh_idx * self.pool_size + max_i
        abs_j = ow_idx * self.pool_size + max_j

        # only the max element gets the gradient
        np.add.at(downstream_gradient, (n_idx, abs_i, abs_j, c_idx), upstream_gradient)
        return downstream_gradient


class Flatten:
    """Flattens spatial dimensions into a 1D vector.

    Connects convolution/pooling layers to ``Dense`` layers.

    Attributes:
        input: Input from the last forward pass, kept for its shape.
    """
    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Flatten every sample into one row.

        Args:
            input: Batch of feature maps, shape ``(N, H, W, C)``.

        Returns:
            Array of shape ``(N, H * W * C)``.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # squash (H, W, C) into one dimension
        output = input.reshape((input.shape[0], -1))
        return output

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Reshape the gradient back to the input shape.

        Args:
            upstream_gradient: dL/d(output), shape ``(N, H * W * C)``.

        Returns:
            dL/d(input), shape ``(N, H, W, C)``.
        """
        # reshape back to (N, H, W, C)
        return np.asarray(upstream_gradient, dtype=np.float32).reshape(self.input.shape)


class Dense:
    """Fully connected layer.

    Computes ``y = activation(x @ W + b)``. Weights are created on the first ``forward``
    call, when the number of input features is known (He normal initialization, zero biases).

    Args:
        neurons: Number of output features (neurons).
        activation_func: Name of the activation (``'relu'``, ``'softmax'``) or ``None`` for no activation.

    Attributes:
        weights: Weight matrix, shape ``(n_inputs, neurons)``. ``None`` until the first forward pass.
        biases: One bias per neuron, shape ``(neurons,)``.
        input: Input from the last forward pass, shape ``(N, n_inputs)``.
        weight_gradient: dL/dW from the last backward pass, same shape as ``weights``.
        bias_gradient: dL/db from the last backward pass, same shape as ``biases``.
    """
    def __init__(self, neurons: int, activation_func: str | None = None) -> None:
        self.neurons: int = neurons
        self.activation_func: activations.Activation = activations.get_activation(activation_func)
        self.weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None
        self.input: np.ndarray | None = None
        self.weight_gradient: np.ndarray | None = None
        self.bias_gradient: np.ndarray | None = None

    def forward(self, input: ArrayLike) -> np.ndarray:
        """Compute ``activation(input @ W + b)``.

        Args:
            input: Batch of feature vectors, shape ``(N, n_inputs)``.

        Returns:
            Layer output, shape ``(N, neurons)``.
        """
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        n_inputs = input.shape[1]
        # init weights on first call with He (Kaiming) normal
        if self.weights is None:
            std = np.sqrt(2.0 / n_inputs)
            self.weights = np.random.normal(0.0, std, (n_inputs, self.neurons)).astype(np.float32)
            self.biases = np.zeros(self.neurons, dtype=np.float32)

        # y = x @ W + b
        output = np.dot(input, self.weights) + self.biases
        output = self.activation_func.forward(output)
        return output

    def backward(self, upstream_gradient: ArrayLike) -> np.ndarray:
        """Compute gradients for the weights, biases and the layer input.

        Stores dL/dW in ``weight_gradient`` and dL/db in ``bias_gradient``.

        Args:
            upstream_gradient: dL/d(output), shape ``(N, neurons)``.

        Returns:
            dL/d(input), shape ``(N, n_inputs)``.
        """
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        # undo activation
        upstream_gradient *= self.activation_func.backward()
        # dL/dW
        self.weight_gradient = np.dot(self.input.T, upstream_gradient)
        # dL/db
        self.bias_gradient = np.sum(upstream_gradient, axis=0)
        # dL/dX - pass to prev layer
        downstream_gradient = np.dot(upstream_gradient, self.weights.T)
        return downstream_gradient


# Any layer that can be added to a Model
Layer = Input | Rescaling | Convolution2D | MaxPooling2D | Flatten | Dense
