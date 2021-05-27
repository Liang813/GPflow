import numpy as np
import gpflow

x = np.linspace(0.0, 0.5)
y = np.sin(x) + 1e-2*np.random.standard_normal(x.shape)

# var_val = 1e-2
var_val = 1e-7

model = gpflow.models.GPR(data = (x,y),
                          kernel = gpflow.kernels.RBF(),
                          noise_variance = var_val)

# gpflow.utilities.print_summary(model)
print("Warnning,The variance of the Gaussian likelihood must be strictly greater than 1e-06")
