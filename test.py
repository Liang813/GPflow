import numpy as np
import GPflow

rng = np.random.RandomState(1)
X = rng.randn( 10, 1)
Y = rng.randn( 10, 1 ) 
Z = rng.randn( 3,1 )
model = GPflow.svgp.SVGP( X=X, Y=Y, kern=GPflow.kernels.RBF(1) ,likelihood=GPflow.likelihoods.Gaussian() , Z=Z )
model.compute_log_likelihood()
