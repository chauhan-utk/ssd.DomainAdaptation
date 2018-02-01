from torch.autograd import Function

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    # change lambd
    def set_lambd(self, lambd):
        self.lambd = lambd

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)
