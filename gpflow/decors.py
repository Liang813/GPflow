# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import tensorflow as tf

from gpflow.misc import GPflowError
from gpflow.base import AutoFlow, TensorConverter
from gpflow.base import Build, CompilableNode


def params_as_tensors(method):
    @functools.wraps(method)
    def tensor_mode_wrapper(obj, *args, **kwargs):
        if not isinstance(obj, TensorConverter):
            raise GPflowError('Tensor mode works only with parmeterized object.')
        name = obj.__tensor_mode__
        attr_value = getattr(obj, name, None)
        setattr(obj, name, True)
        try:
            result = method(obj, *args, **kwargs)
        finally:
            if attr_value is not None:
                setattr(obj, name, attr_value)
            else:
                delattr(obj, name)
        return result
    return tensor_mode_wrapper


def name_scope(name=None):
    def name_scope_wrapper(method):
        @functools.wraps(method)
        def runnable(*args, **kwargs):
            scope_name = name if name is not None else method.__name__
            with tf.name_scope(scope_name):
                return method(*args, **kwargs)
        return runnable
    return name_scope_wrapper


def autoflow(*af_args, **af_kwargs):
    def autoflow_wrapper(method):
        @functools.wraps(method)
        def runnable(obj, *args, **kwargs):
            if not isinstance(obj, AutoFlow):
                raise ValueError('Passed object is not part of AutoFlow.')
            if not isinstance(obj, CompilableNode):
                raise ValueError('Passed object does not implement CompilableNode interface.')
            if obj.is_built_coherence(obj.graph) is Build.NO:
                raise GPflowError('Compilable object is not built.')
            name = method.__name__
            store = obj.get_autoflow(name)
            session = kwargs.pop('session', None)
            session = obj.enquire_session(session=session)
            if not store:
                scope_name = _name_scope_name(obj, name)
                with session.graph.as_default(), tf.name_scope(scope_name):
                    _setup_storage(store, *af_args, **af_kwargs)
                    _build_method(method, obj, store)
            return _session_run(session, store, *args)
        return runnable
    return autoflow_wrapper


def _setup_storage(store, *args, **_kwargs):
    store['arguments'] = [tf.placeholder(*arg) for arg in args]


def _name_scope_name(obj, name):
    return '/'.join(['autoflow', obj.name, name])


def _session_run(session, store, *args):
    feed_dict = dict(zip(store['arguments'], args))
    return session.run(store['result'], feed_dict=feed_dict)


def _build_method(method, obj, store):
    store['result'] = method(obj, *store['arguments'])