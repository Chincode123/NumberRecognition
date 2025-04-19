import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [None] * (len(layers) - 1)
        self.biases = [None] * (len(layers) - 1)
        for i in range(1, len(layers)):
            self.weights[i - 1] = np.matrix(np.random.randn(layers[i], layers[i - 1]))
            self.biases[i - 1] = np.matrix(np.random.randn(layers[i], 1))

def main():
    network = NeuralNetwork([784, 50, 30, 10])
    print("weights", network.weights, "\nbiases", network.biases)

if __name__ == "__main__":
    main()