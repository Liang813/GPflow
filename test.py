import numpy as np
import GPflow
import traceback
try:
  rng = np.random.RandomState(1)
  X = rng.randn( 10, 1)
  Y = rng.randn( 10, 1 ) 
  Z = rng.randn( 3,1 )
  model = GPflow.svgp.SVGP( X=X, Y=Y, kern=GPflow.kernels.RBF(1) ,likelihood=GPflow.likelihoods.Gaussian() , Z=Z )
  model.compute_log_likelihood()
except Exception as e:
  traceback.print_exc(file=open('/script/gpflow99-buggy.txt','w+'))
