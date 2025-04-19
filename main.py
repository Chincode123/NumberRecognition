import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def execute(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Sigmoid(Activation):
    def execute(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return self.execute(x) * (1 - self.execute(x))
    
class ReLU(Activation):
    def execute(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, layers: list[int], activation: Activation = Sigmoid):
        self.weights = [None] * (len(layers) - 1)
        self.biases = [None] * (len(layers) - 1)
        for i in range(1, len(layers)):
            self.weights[i - 1] = np.matrix(np.random.randn(layers[i], layers[i - 1]))
            self.biases[i - 1] = np.matrix(np.random.randn(layers[i], 1))
        self.activation = activation()
    
    def feedforward(self, inputs):
        for w, b in zip(self.weights, self.biases):
            inputs = self.activation.execute(np.dot(w, inputs) + b)
        return inputs

def main():
    network = NeuralNetwork([784, 50, 30, 10])
    print("weights", network.weights, "\nbiases", network.biases, "\nactivation", network.activation)

if __name__ == "__main__":
    main()