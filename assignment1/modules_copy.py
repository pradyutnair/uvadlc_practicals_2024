################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_features = in_features
        self.out_features = out_features
        self.input_layer = input_layer

        # Kaiming initialization for weights: W = N(0, 2 / in_features)
        self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)

        # Initialize biases to zeros
        self.params['bias'] = np.zeros((1, out_features))

        # Initialize gradients to zeros
        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Store input for backward pass
        self.x = x
        # Perform linear transformation Y = XW^T + b
        out = np.dot(x, self.params['weight'].T) + self.params['bias']
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Compute gradients: dL/dW = dl/dY^T * dY/dW = dout^T * X
        self.grads['weight'] = np.dot(dout.T, self.x)
        # Compute gradients: dL/db = dl/dY^T * dY/db = dout^T * 1
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)
        # Compute gradient with respect to input: dl/dX = dl/dY * dY/dX = dout * W
        dx = np.dot(dout, self.params['weight'])
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Store input for backward pass
        self.x = x
        # Clip x to prevent overflow in exp
        x_safe = np.clip(x, -88.0, 88.0)  # ln(max float32) ≈ 88
        # Compute ELU activation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
        out = np.where(x > 0, x, self.alpha * (np.exp(x_safe) - 1))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Clip stored x to prevent overflow in exp
        x_safe = np.clip(self.x, -88.0, 88.0)
        # Compute gradient: f'(x) = 1 if x > 0 else alpha * exp(x)
        dx = np.where(self.x > 0, dout, dout * self.alpha * np.exp(x_safe))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Store input for backward pass
        self.x = x
        # Compute softmax activation: f(x) = exp(x) / sum(exp(x))
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exps / np.sum(exps, axis=1, keepdims=True)
        self.out = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Initialize the gradient with respect to the input
        dx = self.out * (dout - np.sum(self.out * dout, axis=1, keepdims=True))
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Convert y to one-hot encoding
        y_true = np.zeros((y.size, x.shape[1]))
        y_true[np.arange(y.size), y] = 1

        # Shape assertion
        assert x.shape == y_true.shape, "CE Loss Forward Pass: Shape mismatch between x and y_true"
        
        # Compute cross-entropy loss
        # Adding small epsilon to avoid log(0)
        eps = 1e-15
        x_clipped = np.clip(x, eps, 1 - eps)
        out = -np.sum(y_true * np.log(x_clipped))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        y_true = np.zeros((y.size, x.shape[1]))
        y_true[np.arange(y.size), y] = 1

        # Shape assertion
        assert x.shape == y_true.shape, "CE Loss Backward Pass: Shape mismatch between x and y_true"

        # Compute gradient with respect to input: dl/dX = dl/dY * dY/dX
        dx = x - y_true
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx