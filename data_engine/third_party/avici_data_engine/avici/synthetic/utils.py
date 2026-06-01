import math
from collections import defaultdict

import numpy as onp

from avici.synthetic import Data
from avici.utils.graph import mat_to_toporder


def draw_rff_params(*, rng, d, length_scale, output_scale, n_rff):
    """Draws random instantiation of rffs"""
    # draw parameters
    ls = length_scale(rng, shape=(1,)).item() if callable(length_scale) else length_scale
    c = output_scale(rng, shape=(1,)).item() if callable(output_scale) else output_scale

    # draw rffs
    # [d, n_rff]
    omega_j = rng.normal(loc=0, scale=1.0 / ls, size=(d, n_rff))

    # [n_rff, ]
    b_j = rng.uniform(0, 2 * onp.pi, size=(n_rff,))

    # [n_rff, ]
    w_j = rng.normal(loc=0, scale=1.0, size=(n_rff,))

    return dict(
        c=c,
        omega=omega_j,
        b=b_j,
        w=w_j,
        n_rff=n_rff,
    )

def _check_finite(arr, name, extra=""):
    """ ：  NaN / Inf"""
    if not onp.isfinite(arr).all():
        print(
            f"[NaN DETECTED][{name}] {extra} | "
            f"min={onp.nanmin(arr)}, max={onp.nanmax(arr)}"
        )
        return False
    return True


def sample_recursive_scm(*,
                         rng,
                         n_observations_obs,
                         n_observations_int,
                         g,
                         f,
                         nse,
                         interv_dist,
                         n_interv_vars=0,):
    """Ancestral sampling over a DAG

    Args:
        rng:
        n_observations_obs: number of observational data rows to be sampled
        n_observations_int: number of interventional data rows to be sampled
        g: adjacency matrix of the DAG of shape [n_vars, n_vars]
        f: list of functions (mechanisms), one for each node.
            Each f[j] maps: observation matrix `x` [n_obs, n_vars], noise vector `z` [n_obs,], and
            parent indicator `is_parent` [n_vars,] to the observations observed for
            node j [n_obs,]
        nse: list of class instances representing the noise distributions for each node, subclassing `NoiseModel` ABC
        interv_dist: Subclass of `DistributionModel` ABC for sampling intervention values
        n_interv_vars (optional): number of variables intervened upon (default is 0). If -1 is passed, all variables are intervened
            upon. For other integers, a set of intervened variables is randomly selected and interventional data
            is generated in equal proportion for each node based on the total number of `n_observations_int` data points

    Returns:
        dict containing `g`, `x_obs`, `x_int`, `n_vars`, `n_observations_obs`, `n_observations_int`, `is_count_data`
    """

    # =================  =================
    #   G   ID (DAG   0)
    # 1.0 = Linear, 2.0 = RFF ( )
    # =================================================

    n_vars = g.shape[-1]
    toporder = mat_to_toporder(g)

    # sample target nodes for the interventions
    interv_targets = []

    simulate_observ_data = n_observations_obs > 0
    if simulate_observ_data:
        interv_targets += [None]

    simulate_interv_data = n_observations_int > 0
    if simulate_interv_data:
        assert n_interv_vars != 0, f"Need n_interv_vars != 0 to sample interventional data"
        if n_interv_vars == -1 or n_interv_vars == 1.0:
            n_interv_vars = n_vars
        elif not n_interv_vars.is_integer():
            n_interv_vars = math.ceil(n_interv_vars * n_vars)
        interv_targets += sorted(rng.choice(n_vars, size=min(n_vars, n_interv_vars), replace=False).tolist())

    assert (n_interv_vars == -1) or (0 <= n_interv_vars <= n_vars),\
        f"Got `n_interv_vars` = {n_interv_vars} for `n_vars` = {n_vars}, which is invalid."

    # simulate data for observational data and for each interventional target
    data = defaultdict(lambda: defaultdict(list))
    for interv_target in interv_targets:

        if interv_target is None:
            # observational
            data_type = "obs"
            is_intervened = onp.zeros(n_vars).astype(bool)
            n_obs = n_observations_obs

        else:
            # interventional
            data_type = "int"
            is_intervened = onp.eye(n_vars)[interv_target].astype(bool)
            n_obs = math.ceil(n_observations_int / n_interv_vars)

        # ancestral sampling in topological order
        x = onp.zeros((n_obs, n_vars))
        for j in toporder:
            # sample noise
            z_j = nse[j](rng=rng, x=x, is_parent=g[:, j])

            _check_finite(
                    z_j,
                    "noise",
                    extra=f"type={data_type}, node={j}, parents={int(g[:, j].sum())}"
                )


            # compute node given parents and noise or perform intervention state
            if is_intervened[j]:
                x[:, j] = interv_dist(rng, shape=z_j.shape)
            else:
                x[:, j] = f[j](x=x, z=z_j, is_parent=g[:, j])

            _check_finite(
                    x[:, j],
                    "mechanism",
                    extra=(
                        f"type={data_type}, node={j}, "
                        f"parents={onp.where(g[:, j])[0].tolist()}, "
                        f"x_range=({onp.nanmin(x)}, {onp.nanmax(x)}), "
                        f"z_range=({onp.nanmin(z_j)}, {onp.nanmax(z_j)})"
                    )
                )
        if not onp.isfinite(x).all():
            bad_ratio = 1.0 - onp.isfinite(x).mean()
            print(
                f"[NaN DETECTED][x] "
                f"type={data_type}, interv_target={interv_target}, "
                f"bad_ratio={bad_ratio:.6f}, "
                f"x_range=({onp.nanmin(x)}, {onp.nanmax(x)})"
            )
        # generate intervention mask
        # [n_obs, n_vars] with True/False depending on whether node was intervened upon
        interv_mask = onp.tile(is_intervened, (x.shape[0], 1)).astype(onp.float32)

        data[data_type]["x"].append(x)
        data[data_type]["interv_mask"].append(interv_mask)


    # concatenate interventional data, interweaving rows to have balanced observation counts when clipping the end
    if simulate_observ_data:
        x_obs = onp.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
        x_obs_msk = onp.stack(data["obs"]["interv_mask"]).reshape(-1, n_vars, order="F")
    else:
        x_obs = onp.zeros((0, n_vars))  # dummy
        x_obs_msk = onp.zeros((0, n_vars))  # dummy

    if simulate_interv_data:
        x_int = onp.stack(data["int"]["x"]).reshape(-1, n_vars, order="F")
        x_int_msk = onp.stack(data["int"]["interv_mask"]).reshape(-1, n_vars, order="F")
    else:
        x_int = onp.zeros((0, n_vars))  # dummy
        x_int_msk = onp.zeros((0, n_vars))  # dummy

    # clip number of observations to have invariant shape (in case n_obs doesn't evenly devide no. interv targets)
    # [n_observations, n_vars, 2]
    x_obs = onp.stack([x_obs, x_obs_msk], axis=-1)[:n_observations_obs, :, :]
    x_int = onp.stack([x_int, x_int_msk], axis=-1)[:n_observations_int, :, :]

    assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                               f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

    # collect data
    data = Data(
        x_obs=x_obs,
        x_int=x_int,
        is_count_data=False,
    )
    return data


import sys
import numpy as onp
import math
from collections import defaultdict
from avici.synthetic import Data
from avici.utils.graph import mat_to_toporder

#  ，  _check_finite

def sample_recursive_scm_test(*,
                         rng,
                         n_observations_obs,
                         n_observations_int,
                         g,
                         f,
                         nse,
                         interv_dist,
                         n_interv_vars=0,):
    
    # =======   1  =======
    #  ，  Worker  
    if not hasattr(sample_recursive_scm, "_debug_source_printed"):
        print(f"\n{'#'*60}")
        print(f"🔍 [DEBUG]   sample_recursive_scm")
        print(f"📂 [DEBUG]  : {__file__}")
        print(f"{'#'*60}\n")
        sys.stdout.flush()
        sample_recursive_scm._debug_source_printed = True
    # ==========================================

    n_vars = g.shape[-1]
    toporder = mat_to_toporder(g)

    interv_targets = []
    simulate_observ_data = n_observations_obs > 0
    if simulate_observ_data:
        interv_targets += [None]

    simulate_interv_data = n_observations_int > 0
    if simulate_interv_data:
        assert n_interv_vars != 0, "Need n_interv_vars != 0"
        if n_interv_vars == -1 or n_interv_vars == 1.0:
            n_interv_vars = n_vars
        elif not n_interv_vars.is_integer():
            n_interv_vars = math.ceil(n_interv_vars * n_vars)
        
        #   interv_targets   [0, 1, ..., 59]
        interv_targets += sorted(rng.choice(n_vars, size=min(n_vars, n_interv_vars), replace=False).tolist())

    #   ( )
    data = defaultdict(lambda: defaultdict(list))
    for interv_target in interv_targets:
        if interv_target is None:
            data_type = "obs"
            is_intervened = onp.zeros(n_vars).astype(bool)
            n_obs = n_observations_obs
        else:
            data_type = "int"
            is_intervened = onp.eye(n_vars)[interv_target].astype(bool)
            #  
            n_obs = math.ceil(n_observations_int / n_interv_vars)

        x = onp.zeros((n_obs, n_vars))
        for j in toporder:
            z_j = nse[j](rng=rng, x=x, is_parent=g[:, j])
            _check_finite(z_j, "noise")
            
            if is_intervened[j]:
                x[:, j] = interv_dist(rng, shape=z_j.shape)
            else:
                x[:, j] = f[j](x=x, z=z_j, is_parent=g[:, j])
            _check_finite(x[:, j], "mechanism")
            
        interv_mask = onp.tile(is_intervened, (x.shape[0], 1)).astype(onp.float32)
        data[data_type]["x"].append(x)
        data[data_type]["interv_mask"].append(interv_mask)

    # obs   ( )
    if simulate_observ_data:
        x_obs = onp.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
        x_obs_msk = onp.stack(data["obs"]["interv_mask"]).reshape(-1, n_vars, order="F")
    else:
        x_obs = onp.zeros((0, n_vars))
        x_obs_msk = onp.zeros((0, n_vars))

    # =======   &   2  =======
    if simulate_interv_data:
        # data["int"]["x"]   list，  n_targets (60)
        #   shape: (n_obs_per_target, n_vars) -> (2, 60)
        
        # 1. Stack
        raw_x = onp.stack(data["int"]["x"], axis=0) # (60, 2, 60)
        raw_msk = onp.stack(data["int"]["interv_mask"], axis=0)
        
        # 2. Transpose ( )
        #   ( ,  ,  ) -> (2, 60, 60)
        raw_x = raw_x.transpose(1, 0, 2)
        raw_msk = raw_msk.transpose(1, 0, 2)
        
        # 3. Reshape
        #   (120, 60)
        #  : [T0_S0, T1_S0, ..., T59_S0, T0_S1, ...]
        x_int = raw_x.reshape(-1, n_vars)
        x_int_msk = raw_msk.reshape(-1, n_vars)
        
        # ---  ：  5   ---
        #  ， 
        if not hasattr(sample_recursive_scm, "_debug_sort_checked"):
            print(f"\n🧐 [DEBUG]   (d={n_vars}):")
            print(f"  - Raw Stack Shape: {onp.stack(data['int']['x'], axis=0).shape}")
            print(f"  - Transposed Shape: {raw_x.shape} (Expect: (samples, targets, vars))")
            print(f"  - Final Reshape: {x_int.shape}")
            
            #   5  ，  (0, 1, 2, 3, 4)
            print("  -   5  :")
            for i in range(min(5, x_int_msk.shape[0])):
                #   mask   1  
                interv_idx = onp.where(x_int_msk[i] > 0)[0]
                print(f"    Row {i}: Intervened on {interv_idx}")
            
            #  
            row0_target = onp.where(x_int_msk[0] > 0)[0]
            row1_target = onp.where(x_int_msk[1] > 0)[0]
            if len(row0_target) > 0 and len(row1_target) > 0:
                if row0_target[0] == row1_target[0]:
                    print("❌ [DEBUG]  ！！Row 0   Row 1  ！")
                else:
                    print("✅ [DEBUG] Row 0   Row 1  ")
            
            sample_recursive_scm._debug_sort_checked = True
            sys.stdout.flush()
        # -----------------------------------------------
        
    else:
        x_int = onp.zeros((0, n_vars))
        x_int_msk = onp.zeros((0, n_vars))

    #  
    x_obs = onp.stack([x_obs, x_obs_msk], axis=-1)[:n_observations_obs, :, :]
    x_int = onp.stack([x_int, x_int_msk], axis=-1)[:n_observations_int, :, :] #   100

    assert x_obs.size != 0 or x_int.size != 0, "No observations sampled"

    data = Data(x_obs=x_obs, x_int=x_int, is_count_data=False)
    return data