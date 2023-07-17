import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input


class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs < 0] = 0
        return grad_input


# Example usage
input_size = 100
output_size = 10

fc_layer = FullyConnectedLayer(input_size, output_size)
relu_activation = ReLUActivation()

# Forward pass
inputs = np.random.randn(1, input_size)
fc_output = fc_layer.forward(inputs)
relu_output = relu_activation.forward(fc_output)

# Backward pass
grad_output = np.ones((1, output_size))
grad_relu = relu_activation.backward(grad_output)
grad_fc = fc_layer.backward(grad_relu, learning_rate=0.001)
