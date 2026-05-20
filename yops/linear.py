import numpy as np
from yops.yopsmodule import YModule


class Linear(YModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.rand(in_features, out_features)
        self.b = np.random.rand(out_features)
        self._parameters = {'W': self.W, 'b': self.b}

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad_W = x.T @ grad_output
        grad_b = grad_output.sum(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input, grad_W, grad_b
    
    def update_parameters(self, grad_W: np.ndarray, grad_b: np.ndarray, lr: float):
        self.W -= lr * grad_W
        self.b -= lr * grad_b
    