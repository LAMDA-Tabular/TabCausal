import functools
import numpy as onp
from avici.synthetic import MechanismModel, Distribution
from avici.synthetic.noise_scale import init_noise_dist, SimpleNoise
from avici.synthetic.utils import sample_recursive_scm

# ==============================================================================
# 1. Distributions ( )
# ==============================================================================

class Exponential(Distribution):
    def __init__(self, scale=1.0):
        self.scale = scale
    def __call__(self, rng, shape=None):
        return rng.exponential(scale=self.scale, size=shape)

class Gumbel(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
    def __call__(self, rng, shape=None):
        return rng.gumbel(loc=self.loc, scale=self.scale, size=shape)

class LogNormal(Distribution):
    def __init__(self, mean=0.0, sigma=1.0):
        self.mean = mean
        self.sigma = sigma
    def __call__(self, rng, shape=None):
        return rng.lognormal(mean=self.mean, sigma=self.sigma, size=shape)

class StudentT(Distribution):
    def __init__(self, df=4.0, scale=1.0):
        self.df = df
        self.scale = scale
    def __call__(self, rng, shape=None):
        return self.scale * rng.standard_t(df=self.df, size=shape)
    

class Tanh:
    def __repr__(self): return "Tanh"

class Sigmoid:
    def __repr__(self): return "Sigmoid"

class LeakyReLU:
    def __repr__(self): return "LeakyReLU"

class Cube:
    def __repr__(self): return "Cube"

# ==============================================================================
# 2. Helpers ( )
# ==============================================================================

def _resolve_param(rng, param):
    if isinstance(param, (list, tuple)):
        idx = rng.integers(0, len(param))
        return param[idx]
    return param

def _is_wrapper(obj):
    return type(obj).__name__ == 'CustomClassWrapper' or 'Wrapper' in type(obj).__name__

#  ： ，  overflow   inf
SAFE_CLIP_MIN = -1e8
SAFE_CLIP_MAX = 1e8

# ==============================================================================
# 3. Mechanisms (  Clip  )
# ==============================================================================

class MLPAdditive(MechanismModel):
    def __init__(self, n_layers, hidden_dim, activation, bias, noise, 
                 noise_scale=None, noise_scale_heteroscedastic=None, 
                 n_interv_vars=0, interv_dist=None):
        assert interv_dist is not None or n_interv_vars == 0
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias_dist = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _mlp_mechanism(x, z, is_parent, weights, biases, act_fn):
        inp = x[:, onp.where(is_parent)[0]]
        h = inp
        #  
        for W, b in zip(weights[:-1], biases[:-1]):
            h = act_fn(h @ W + b)
        #  
        out = h @ weights[-1] + biases[-1] + z
        # [FIX]  
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        
        n_layers_val = int(_resolve_param(rng, self.n_layers))
        hidden_dim_val = int(_resolve_param(rng, self.hidden_dim))
        
        act_val = _resolve_param(rng, self.activation)
        if _is_wrapper(act_val): act_val = getattr(act_val, 'name', 'relu')
        
        act_name = str(act_val).lower()
        if 'sigmoid' in act_name: act_fn = lambda x: 1 / (1 + onp.exp(-x))
        elif 'tanh' in act_name: act_fn = onp.tanh
        elif 'leaky' in act_name: act_fn = lambda x: onp.where(x > 0, x, 0.01 * x)
        else: act_fn = lambda x: onp.maximum(0, x)

        bias_dist_fn = _resolve_param(rng, self.bias_dist)
        if _is_wrapper(bias_dist_fn): bias_dist_fn = lambda r, s: r.normal(size=s)

        f = []
        for j in range(n_vars):
            n_parents = int(g[:, j].sum().item())
            dims = [n_parents] + [hidden_dim_val] * n_layers_val
            
            my_weights, my_biases = [], []
            for d_in, d_out in zip(dims[:-1], dims[1:]):
                limit = onp.sqrt(6 / (d_in + d_out)) if d_in > 0 else 1.0
                W = rng.uniform(-limit, limit, size=(d_in, d_out))
                b = bias_dist_fn(rng, shape=(d_out,))
                my_weights.append(W); my_biases.append(b)
            
            limit_out = onp.sqrt(6 / (dims[-1] + 1))
            W_out = rng.uniform(-limit_out, limit_out, size=(dims[-1],))
            b_out = bias_dist_fn(rng, shape=(1,)).item()
            my_weights.append(W_out); my_biases.append(b_out)

            f.append(functools.partial(self._mlp_mechanism, weights=my_weights, biases=my_biases, act_fn=act_fn))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            if _is_wrapper(noise_dist): noise_dist = lambda r, s: r.normal(size=s)
            
            noise_scale_dist = _resolve_param(rng, self.noise_scale)
            
            nse.append(init_noise_dist(rng=rng, dim=int(g[:, j].sum().item()),
                                       dist=noise_dist,
                                       noise_scale=noise_scale_dist,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class PolynomialAdditive(MechanismModel):
    def __init__(self, param, bias, noise, 
                 degree=2,
                 noise_scale=None, noise_scale_heteroscedastic=None, 
                 n_interv_vars=0, interv_dist=None, include_interaction=True):
        
        assert interv_dist is not None or n_interv_vars == 0
        self.degree = degree
        self.param = param
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist
        self.include_interaction = include_interaction

    @staticmethod
    def _poly_mechanism(x, z, is_parent, w_linear, w_quad, b):
        x_parents = x[:, onp.where(is_parent)[0]]
        linear_term = x_parents @ w_linear
        quad_term = 0.0
        if w_quad is not None and x_parents.shape[1] > 0:
            quad_term = onp.einsum('ni,nk,ik->n', x_parents, x_parents, w_quad)
        
        out = linear_term + quad_term + b + z
        # [FIX]  ， 
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        f = []
        
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        if _is_wrapper(param_dist): param_dist = lambda r, s: r.uniform(-1, 1, size=s)
        if _is_wrapper(bias_dist): bias_dist = lambda r, s: r.uniform(-1, 1, size=s)
        
        do_interact = _resolve_param(rng, self.include_interaction)

        for j in range(n_vars):
            n_parents = int(g[:, j].sum().item())
            w_linear = param_dist(rng, shape=(n_parents,))
            
            w_quad = None
            if do_interact:
                w_quad = param_dist(rng, shape=(n_parents, n_parents)) * 0.3 
                w_quad = (w_quad + w_quad.T) / 2
            
            b = bias_dist(rng, shape=(1,))
            f.append(functools.partial(self._poly_mechanism, w_linear=w_linear, w_quad=w_quad, b=b))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            if _is_wrapper(noise_dist): noise_dist = lambda r, s: r.normal(size=s)
            noise_scale_dist = _resolve_param(rng, self.noise_scale)

            nse.append(init_noise_dist(rng=rng, dim=int(g[:, j].sum().item()),
                                       dist=noise_dist,
                                       noise_scale=noise_scale_dist,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class MultiplicativeLinear(MechanismModel):
    def __init__(self, param, bias, noise, n_interv_vars=0, interv_dist=None):
        self.param = param
        self.bias = bias
        self.noise = noise
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _mult_mechanism(x, z, is_parent, w, b):
        val = ((x @ (w * is_parent)) + b) * z
        # [FIX]  ， 
        return onp.clip(val, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        if _is_wrapper(param_dist): param_dist = lambda r, s: r.uniform(0.5, 1.5, size=s)
        if _is_wrapper(bias_dist): bias_dist = lambda r, s: r.uniform(-1, 1, size=s)

        f = []
        for j in range(n_vars):
            w = param_dist(rng, shape=(n_vars,))
            b = bias_dist(rng, shape=(1,))
            f.append(functools.partial(self._mult_mechanism, w=w, b=b))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            if _is_wrapper(noise_dist): 
                from avici.synthetic import Uniform
                noise_dist = Uniform(low=0.5, high=1.5)
            
            nse.append(SimpleNoise(dist=noise_dist, scale=1.0))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class PostNonLinear(MechanismModel):
    def __init__(self, base_mechanism, nonlinearity, n_interv_vars=0, interv_dist=None):
        self.base_mech = base_mechanism
        
        if _is_wrapper(base_mechanism):
            kwargs = getattr(base_mechanism, 'kwargs', {})
            self.param = kwargs.get('param', None)
            self.bias = kwargs.get('bias', None)
            self.noise = kwargs.get('noise', None)
            self.noise_scale = kwargs.get('noise_scale', None)
            self.noise_scale_heteroscedastic = None
        else:
            self.param = getattr(base_mechanism, 'param', None)
            self.bias = getattr(base_mechanism, 'bias', None)
            self.noise = getattr(base_mechanism, 'noise', None)
            self.noise_scale = getattr(base_mechanism, 'noise_scale', None)
            self.noise_scale_heteroscedastic = getattr(base_mechanism, 'noise_scale_heteroscedastic', None)
        
        self.nonlinearity_type = nonlinearity
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _pnl_mechanism(x, z, is_parent, w, b, nonlin_fn):
        linear_out = (x @ (w * is_parent)) + b + z
        out = nonlin_fn(linear_out)
        # [FIX]   (  Sigmoid/Tanh  ，  LeakyReLU   Cube  )
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        
        nl = _resolve_param(rng, self.nonlinearity_type)
        if _is_wrapper(nl): nl = nl.name
        
        nl_str = str(nl).lower()
        if 'sigmoid' in nl_str: g_fn = lambda x: 1 / (1 + onp.exp(-x))
        elif 'tanh' in nl_str: g_fn = onp.tanh
        elif 'cube' in nl_str: g_fn = lambda x: x**3
        elif 'leaky' in nl_str: g_fn = lambda x: onp.where(x > 0, x, 0.1 * x)
        else: g_fn = lambda x: x 

        f = []
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        
        if _is_wrapper(param_dist): param_dist = lambda r, s: r.uniform(-1, 1, size=s)
        if _is_wrapper(bias_dist): bias_dist = lambda r, s: r.uniform(-1, 1, size=s)

        for j in range(n_vars):
            if param_dist: w = param_dist(rng, shape=(n_vars,))
            else: w = rng.uniform(-1, 1, size=(n_vars,))
            
            if bias_dist: b = bias_dist(rng, shape=(1,))
            else: b = onp.zeros(1)
            f.append(functools.partial(self._pnl_mechanism, w=w, b=b, nonlin_fn=g_fn))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            noise_scale_dist = _resolve_param(rng, self.noise_scale)
            
            if _is_wrapper(noise_dist): 
                from avici.synthetic import Gaussian
                noise_dist = Gaussian()

            nse.append(init_noise_dist(rng=rng, dim=int(g[:, j].sum().item()),
                                       dist=noise_dist,
                                       noise_scale=noise_scale_dist,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class SoftplusAdditive(MechanismModel):
    def __init__(self, param, bias, noise, noise_scale=None,
                 noise_scale_heteroscedastic=None, n_interv_vars=0, interv_dist=None):
        assert interv_dist is not None or n_interv_vars == 0
        self.param = param
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _softplus_mechanism(x, z, is_parent, w, b):
        h = (x @ (w * is_parent)) + b
        h = onp.clip(h, -30.0, 30.0)
        out = onp.log1p(onp.exp(h)) - onp.log(2.0) + z
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        if _is_wrapper(param_dist):
            param_dist = lambda r, s: r.uniform(-1.0, 1.0, size=s)
        if _is_wrapper(bias_dist):
            bias_dist = lambda r, s: r.uniform(-1.0, 1.0, size=s)

        f = []
        for _ in range(n_vars):
            w = param_dist(rng, shape=(n_vars,))
            b = bias_dist(rng, shape=(1,))
            f.append(functools.partial(self._softplus_mechanism, w=w, b=b))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            noise_scale_dist = _resolve_param(rng, self.noise_scale)
            if _is_wrapper(noise_dist):
                from avici.synthetic import Gaussian
                noise_dist = Gaussian()
            nse.append(init_noise_dist(
                rng=rng,
                dim=int(g[:, j].sum().item()),
                dist=noise_dist,
                noise_scale=noise_scale_dist,
                noise_scale_heteroscedastic=self.noise_scale_heteroscedastic,
            ))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class SigmoidAdditive(MechanismModel):
    def __init__(self, param, bias, noise, noise_scale=None,
                 noise_scale_heteroscedastic=None, n_interv_vars=0, interv_dist=None):
        assert interv_dist is not None or n_interv_vars == 0
        self.param = param
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _sigmoid_mechanism(x, z, is_parent, w, b):
        h = (x @ (w * is_parent)) + b
        h = onp.clip(h, -30.0, 30.0)
        out = (2.0 / (1.0 + onp.exp(-h))) - 1.0 + z
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        if _is_wrapper(param_dist):
            param_dist = lambda r, s: r.uniform(-1.0, 1.0, size=s)
        if _is_wrapper(bias_dist):
            bias_dist = lambda r, s: r.uniform(-1.0, 1.0, size=s)

        f = []
        for _ in range(n_vars):
            w = param_dist(rng, shape=(n_vars,))
            b = bias_dist(rng, shape=(1,))
            f.append(functools.partial(self._sigmoid_mechanism, w=w, b=b))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            noise_scale_dist = _resolve_param(rng, self.noise_scale)
            if _is_wrapper(noise_dist):
                from avici.synthetic import Gaussian
                noise_dist = Gaussian()
            nse.append(init_noise_dist(
                rng=rng,
                dim=int(g[:, j].sum().item()),
                dist=noise_dist,
                noise_scale=noise_scale_dist,
                noise_scale_heteroscedastic=self.noise_scale_heteroscedastic,
            ))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )


class ThresholdAdditive(MechanismModel):
    def __init__(self, param, bias, noise, threshold, noise_scale=None,
                 noise_scale_heteroscedastic=None, n_interv_vars=0, interv_dist=None):
        assert interv_dist is not None or n_interv_vars == 0
        self.param = param
        self.bias = bias
        self.noise = noise
        self.threshold = threshold
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def _threshold_mechanism(x, z, is_parent, w, b, threshold):
        h = (x @ (w * is_parent)) + b
        out = onp.maximum(h - threshold, 0.0) + z
        return onp.clip(out, SAFE_CLIP_MIN, SAFE_CLIP_MAX)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):
        n_vars = g.shape[-1]
        param_dist = _resolve_param(rng, self.param)
        bias_dist = _resolve_param(rng, self.bias)
        thr_dist = _resolve_param(rng, self.threshold)
        if _is_wrapper(param_dist):
            param_dist = lambda r, s: r.uniform(0.3, 1.5, size=s)
        if _is_wrapper(bias_dist):
            bias_dist = lambda r, s: r.uniform(-0.5, 0.5, size=s)
        if _is_wrapper(thr_dist):
            thr_dist = lambda r, s: r.uniform(0.1, 0.5, size=s)

        f = []
        for _ in range(n_vars):
            w = param_dist(rng, shape=(n_vars,))
            b = bias_dist(rng, shape=(1,))
            thr = thr_dist(rng, shape=(1,)).reshape(()).item()
            f.append(functools.partial(self._threshold_mechanism, w=w, b=b, threshold=thr))

        nse = []
        for j in range(n_vars):
            noise_dist = _resolve_param(rng, self.noise)
            noise_scale_dist = _resolve_param(rng, self.noise_scale)
            if _is_wrapper(noise_dist):
                from avici.synthetic import Gaussian
                noise_dist = Gaussian()
            nse.append(init_noise_dist(
                rng=rng,
                dim=int(g[:, j].sum().item()),
                dist=noise_dist,
                noise_scale=noise_scale_dist,
                noise_scale_heteroscedastic=self.noise_scale_heteroscedastic,
            ))

        return sample_recursive_scm(
            rng=rng, n_observations_obs=n_observations_obs, n_observations_int=n_observations_int,
            g=g, f=f, nse=nse, interv_dist=self.interv_dist, n_interv_vars=self.n_interv_vars,
        )
