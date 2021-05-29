import numpy as np
import GPflow

rng = np.random.RandomState(23)
X = rng.rand(4, 1)
Y = rng.randn(X.shape[0])[:, None] * 0.1 + np.sum(np.sin(3*X), 1)[:, None]
full = GPflow.gpr.GPR(X, Y, GPflow.kernels.ArcCosine(X.shape[1]))
full._compile()
print(full._objective(full.get_free_state())[1])
