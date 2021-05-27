import numpy as np
import gpflow
import traceback
try:
  x = np.linspace(0.0, 0.5)
  y = np.sin(x) + 1e-2*np.random.standard_normal(x.shape)

  # var_val = 1e-2
  var_val = 1e-7

  model = gpflow.models.GPR(data = (x,y),
                            kernel = gpflow.kernels.RBF(),
                            noise_variance = var_val)

  gpflow.utilities.print_summary(model)
except Exception as e:
  traceback.print_exc(file=open('/script/gpflow1407-buggy.txt','w+'))
