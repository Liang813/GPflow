import GPflow

m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(3))
m.optimize()


m2 = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.Matern52(3)+ GPflow.kernels.White(input_dim=3))
m2.optimize()
