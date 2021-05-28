import pytest
import numpy as np
import tensorflow as tf
import gpflow.kernels as kernels
from gpflow.test_util import session_tf
from gpflow import settings


rng = np.random.RandomState(0)


class Datum:
    num_data = 100
    D = 100
    X = rng.rand(num_data, D) * 100


@pytest.mark.parametrize('kernel', [kernels.Matern12, kernels.Matern32, kernels.Matern52, kernels.Exponential, kernels.Cosine])
def test_kernel_euclidean_distance(session_tf, kernel):
    '''
    Tests output & gradients of kernels that are a function of the (scaled) euclidean distance
    of the points. We test on a high dimensional space, which can generate very small distances
    causing the scaled_square_dist to generate some negative values.
    '''

    k = kernel(Datum.D)
    K = k.compute_K_symm(Datum.X)
    assert not np.isnan(K).any(), 'There are NaNs in the output of the ' + kernel.__name__ + ' kernel.'
    assert np.isfinite(K).all(), 'There are Infs in the output of the ' + kernel.__name__ + ' kernel.'

    X = tf.placeholder(settings.float_type)
    dK = session_tf.run(tf.gradients(k.K(X, X), X)[0], feed_dict={X: Datum.X})
    assert not np.isnan(dK).any(), 'There are NaNs in the gradient of the ' + kernel.__name__ + ' kernel.'
    assert np.isfinite(dK).all(), 'There are Infs in the output of the ' + kernel.__name__ + ' kernel.'
