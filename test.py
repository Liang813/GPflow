import numpy as np
import tensorflow as tf
import GPflow

full_dim = 3
# Five data points in 3 dimensions
D_X = np.array([
 [31.55437035,  31.55437038,  32.36345678],
 [31.55437037,  31.55437037,  32.36345679],
 [31.55437037,  31.55437037,  32.36345679],
 [28.31802469,  21.84533348, -29.12711068],
 [28.31802469,  21.84533348, -29.12711111],
])
# Trying with different dataset does not show any problem
# D_X = np.random.uniform(-32, 32, 5*full_dim).reshape(-1, full_dim)

Y = np.random.uniform(2, 7, D_X.shape[0]).reshape(-1, 1)    # any random choice
X_test = np.random.uniform(-32, 32, 10*full_dim).reshape(-1, full_dim)  # random test set

# Matern52 test isotropic
kernel52_iso = GPflow.kernels.Matern52(input_dim=3, ARD=False)
kernel52_iso.lengthscales._array = np.array([0.01095283])     # critical lengthscale

GPmodel = GPflow.gpr.GPR(D_X, Y, kernel52_iso)
mean52_iso, vars52_iso = GPmodel.predict_f(X_test)  # posterior mean is all nan

K52_iso = kernel52_iso.K(D_X)
gg = K52_iso.graph
sess = tf.InteractiveSession(graph=gg)
npK52_iso = K52_iso.eval(session=sess)  # Covariance matrix contains nan

# Matern52 test ARD
kernel52_ARD = GPflow.kernels.Matern52(input_dim=3, ARD=True)
kernel52_ARD.lengthscales._array = np.array([0.01095283, 0.01095283, 0.01095283])     # critical lengthscales

GPmodel = GPflow.gpr.GPR(D_X, Y, kernel52_ARD)
mean52_ARD, vars52_ARD = GPmodel.predict_f(X_test)

K52_ARD = kernel52_ARD.K(D_X)
gg = K52_ARD.graph
sess = tf.InteractiveSession(graph=gg)
npK52_ARD = K52_ARD.eval(session=sess)

# Matern32 test isotropic
kernel32_iso = GPflow.kernels.Matern32(input_dim=3, ARD=False)
kernel32_iso.lengthscales._array = np.array([0.01095283])

GPmodel = GPflow.gpr.GPR(D_X, Y, kernel32_iso)
mean32_iso, vars32_iso = GPmodel.predict_f(X_test)

# Matern32 test ARD
kernel32_ARD = GPflow.kernels.Matern32(input_dim=3, ARD=True)
kernel32_ARD.lengthscales._array = np.array([0.01095283, 0.01095283, 0.01095283])

GPmodel = GPflow.gpr.GPR(D_X, Y, kernel32_ARD)
mean32_ARD, vars32_ARD = GPmodel.predict_f(X_test)

