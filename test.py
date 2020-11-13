import tensorflow as tf
from gpflow.quadrature import ndiag_mc

N = 10
X_dim = 2
Z_dim = 2
Y_dim = 4

def func(Z, X=None, Y=None):
    XZ = tf.concat([X, Z], axis=1)  # N x (X_dim + Z_dim)
    return tf.reduce_sum(XZ - Y, axis=1)  # N x 1

Z_mu = tf.zeros((N, Z_dim), dtype=tf.float64)
Z_var = tf.ones((N, Z_dim), dtype=tf.float64)

X = tf.random_normal((N, X_dim), dtype=tf.float64)
Y = tf.random_normal((N, Y_dim), dtype=tf.float64)

Ez = ndiag_mc(func, 1000, Z_mu, Z_var, logspace=False, X=X, Y=Y)

print(tf.Session().run(Ez))
