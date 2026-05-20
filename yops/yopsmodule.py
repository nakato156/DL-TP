class YModule:
    def __init__(self):
        self._parameters = {}
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        return self._parameters