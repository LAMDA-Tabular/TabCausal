#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Haoyue
@file: main.py
@time: 3/11/2025
@desc: main entrance for the CDIS algorithm.
"""

import copy
from functools import partial
import numpy as np
import networkx as nx
import causallearn.utils.cit as cit
from FCI_tools import (
    get_skeleton_and_sepsets,
    get_PAG_from_skeleton_and_sepsets,
    DASH, AROW, CIRC,
    translate_PAG_dict_format)


def cdis_core(
    nodenum,
    ci_tester_list,
    max_cond_set_size=None,
    max_skeleton_refinement_length=None,
    verbose=False
):
    '''
    :param ci_tester_list:
        a list of 1+num_of_intervs callable objects, each of which is a conditional independence tester that returns a boolean.
        the first one is on observational data only, callable to variables in {0,1,...,nodenum-1}.
        fot the rest num_of_intervs ones, each k-th tester is on the stacked data of observational and k-th interventional data,
            callable to variables in {0,1,...,nodenum-1} + {ZETA}, where ZETA=nodenum, as the intervention indicator.
        a list of np.ndarray, each of shape (n_samples, n_vars).
        the first one is pure observational data; the rest are interventional data.
        `n_samples' of each dataset can be different; the sample individuals among datasets need not be matched.
    :param max_cond_set_size:
        int or None, the maximum size of conditioning set to be enumerated in learning the adjacencies in FCI.
        default: None, i.e., no constraint (correct, but may be slow).
        for the purpose of speedup in real data and when the graph is dense, you may want to set it to a small number like 3.
    :param max_skeleton_refinement_length:
        int or None, the maximum length of paths to be enumerated in refining the adjacencies in FCI.
            check _Causation, Prediction, and Search (pp. 144–145)_ for more on this refinement step.
        default: None, i.e., no constraint (correct, but may be super slow).
        for empirical speedup, we recommend setting it to -1 (as default in the above cdis_from_data),
            i.e., dont run the skeleton refinement at all,
            since usually, the skeleton can already be learnt correctly with PC's search from neighbors.
    :param verbose:
        bool, whether to print out warnings and messages.

    :return:
        a dictionary containing the following key-value pairs:
            'final_PAG': the final PAG returned by the CDIS algorithm, which maximally uses all observational and interventional data.
            'pure_observational_PAG': the PAG one can maximally get by only using observational data. this is for record and comparison only.
        each PAG is represented as a dictionary in forms like {'->': set(), '--': set(), ...}
    '''
    ZETA = nodenum
    num_of_intervs = len(ci_tester_list) - 1

    def update_edge(CURREDGES, node1, node2, type1, type2):
        new_type_1, new_type_2 = None, None
        curr_type_1, curr_type_2 = CURREDGES.get((node1, node2), (CIRC, CIRC))
        if type1 is not None: # intend to update type1
            if curr_type_1 != CIRC and curr_type_1 != type1:
                if verbose:
                    # in real data, due to test errors, the CI results may be inconsistent with each other and with the graph, causing conflicts.
                    print(f"[WARNING] [OUTER REFINE] Conflict detected: Attempt to change '{curr_type_1}' to '{type1}' was not successful.")
            elif curr_type_2 == AROW and type1 == AROW:
                if verbose:
                    print(f"[WARNING] [OUTER REFINE] Conflict with sure_no_latents: Attempt to orient an <-> edge was not successful.")
            elif curr_type_1 == CIRC:
                new_type_1 = type1 # finally safe, directly orient.
        if new_type_1 is not None: curr_type_1 = new_type_1
        if type2 is not None:
            if curr_type_2 != CIRC and curr_type_2 != type2:
                if verbose:
                    print(f"[WARNING] [OUTER REFINE] Conflict detected: Attempt to change '{curr_type_2}' to '{type2}' was not successful.")
            elif curr_type_1 == AROW and type2 == AROW:
                if verbose:
                    print(f"[WARNING] [OUTER REFINE] Conflict with sure_no_latents: Attempt to orient an <-> edge was not successful.")
            elif curr_type_2 == CIRC:
                new_type_2 = type2 # finally safe, directly orient.
        if new_type_1 is not None or new_type_2 is not None:
            if new_type_1 is None: new_type_1 = curr_type_1
            if new_type_2 is None: new_type_2 = curr_type_2
            CURREDGES[(node1, node2)] = (new_type_1, new_type_2)
            CURREDGES[(node2, node1)] = (new_type_2, new_type_1)

    # step 1. get skeleton and sepsets
    p0_skeleton, p0_Sepsets, p0_Dependencies = get_skeleton_and_sepsets(
        list(range(nodenum)), ci_tester_list[0],
        max_cond_set_size=max_cond_set_size, max_skeleton_refinement_length=max_skeleton_refinement_length)
    pk_skeletons, pk_Sepsets = {}, {}
    varIDs_changed_by_interventions = {}
    for k in range(1, 1 + num_of_intervs):
        boolean_tester = ci_tester_list[k]
        sure_adjacencies = set(p0_skeleton)
        sure_dependencies = set(p0_Dependencies) | {(i, j, frozenset(S.union({ZETA}))) for i, j, S in p0_Dependencies}
        assert all(i < j for i, j in sure_adjacencies)
        assert all(i < j for i, j, _ in sure_dependencies)
        pk_skeleton, pk_Sepset, _ = get_skeleton_and_sepsets(
            list(range(nodenum + 1)), boolean_tester, sure_adjacencies, sure_dependencies,
            max_cond_set_size=max_cond_set_size, max_skeleton_refinement_length=max_skeleton_refinement_length)
        assert set(sure_adjacencies) <= set(pk_skeleton)
        varIDs_changed_by_interventions[k] = {i for i in range(nodenum) if not boolean_tester(i, ZETA, [])}
        pk_skeletons[k] = pk_skeleton
        pk_Sepsets[k] = pk_Sepset

    # step 2. get PAG from skeleton and sepsets
    last_p0_PAG, _ = get_PAG_from_skeleton_and_sepsets(
        nodelist=list(range(nodenum)),
        skeleton_edges=p0_skeleton,
        sepsets=p0_Sepsets,
        sure_no_latents=True,
        verbose=verbose)
    pure_observational_p0_PAG = copy.deepcopy(last_p0_PAG)
    last_pk_PAGs = {k: {} for k in range(1, 1 + num_of_intervs)}

    current_round = 1
    while True:
        directed_edges_from_observational = {(x, y) for (x, y), (t1, t2) in last_p0_PAG.items() if t1 == DASH and t2 == AROW}

        for k in range(1, 1 + num_of_intervs):
            directed_edges_from_I_to_X = {(ZETA, xid) for xid, node2 in pk_skeletons[k] if node2 == ZETA}
            background_knowledge_edges = copy.deepcopy(last_pk_PAGs[k])
            for x, y in directed_edges_from_observational | directed_edges_from_I_to_X:
                update_edge(background_knowledge_edges, x, y, DASH, AROW)

            last_pk_PAGs[k], _ = get_PAG_from_skeleton_and_sepsets(
                nodelist=list(range(nodenum + 1)),
                skeleton_edges=pk_skeletons[k],
                sepsets=pk_Sepsets[k],
                background_knowledge_edges=background_knowledge_edges,
                sure_no_latents=True,
                verbose=verbose
            )

        this_p0_PAG = copy.deepcopy(last_p0_PAG)
        for (n1, n2), (p0_type1, p0_type2) in last_p0_PAG.items():
            if p0_type1 != CIRC and p0_type2 != CIRC: continue
            for k in range(1, 1 + num_of_intervs):
                pk_type1, pk_type2 = last_pk_PAGs[k][(n1, n2)]
                if pk_type1 == DASH and pk_type2 == DASH:
                    update_edge(this_p0_PAG, n1, n2, DASH, DASH)
                    break
                if pk_type1 == DASH and pk_type2 == AROW and p0_type1 == CIRC and p0_type1 == DASH:
                    update_edge(this_p0_PAG, n1, n2, DASH, DASH)
                    break
                if pk_type1 == DASH and pk_type2 == AROW and n1 in varIDs_changed_by_interventions[k]:
                    update_edge(this_p0_PAG, n1, n2, DASH, AROW)
                    break
            all_pk_types = set()
            for k in range(1, 1 + num_of_intervs):
                pk_type1, pk_type2 = last_pk_PAGs[k][(n1, n2)]
                all_pk_types.add((pk_type1, pk_type2))
            if (DASH, AROW) in all_pk_types and (AROW, DASH) in all_pk_types:
                update_edge(this_p0_PAG, n1, n2, DASH, DASH)

        if this_p0_PAG == last_p0_PAG:
            return {
                'final_PAG': translate_PAG_dict_format(this_p0_PAG),
                'pure_observational_PAG': translate_PAG_dict_format(pure_observational_p0_PAG),
            }
        else:
            last_p0_PAG = this_p0_PAG
            current_round += 1



def cdis_from_oracle_graph(
    nodenum,
    dag_edgelist,
    interv_targets,
    selection_parents
):
    '''
    this is to get the oracle CDIS results from oracle DAG, intervention targets, and selection parents.
    :param nodenum:
        int, the number of substantial variables in the original DAG.
    :param dag_edgelist:
        list of (i, j) tuples for directed edges;   0 <= i, j <= nodenum-1
    :param interv_targets:
        list of subsets of {0,1,...,nodenum-1}; each subset is a target for intervention.
    :param selection_parents:
        list of subsets of {0,1,...,nodenum-1}; each subset is a set of parents for a selection mechanism.
    :return:
        see description in cdis_core.
    '''
    original_G = nx.DiGraph()
    original_G.add_nodes_from(list(range(nodenum)) + [f'S*{i}' for i in range(len(selection_parents))])
    original_G.add_edges_from(dag_edgelist + [(j, f'S*{i}') for i, parents in enumerate(selection_parents) for j in parents])
    original_G_descendants = {i: nx.descendants(original_G, i) | {i} for i in original_G.nodes}

    boolean_tester_func = lambda graph, x, y, Z: nx.is_d_separator(graph, {x}, {y}, set(Z) | {f'S*{i}' for i in range(len(selection_parents))})
    ci_tester_list = [partial(boolean_tester_func, original_G)]
    for k_target in interv_targets:
        ZETA = nodenum
        varIDs_changed_by_intervention = {int(_) for _ in set().union(*[original_G_descendants[ikt] for ikt in k_target]) if not isinstance(_, str)}
        twin_G = nx.DiGraph()
        twin_G.add_nodes_from(
            [i for i in range(nodenum)] +  # the f'X{i}'s in the paper
            [f'X*{i}' for i in varIDs_changed_by_intervention] +
            [f'S*{i}' for i in range(len(selection_parents))] +
            [f'E{i}' for i in varIDs_changed_by_intervention] +
            [ZETA]
        )
        for i, j in dag_edgelist:
            twin_G.add_edge(i, j)
            if i in varIDs_changed_by_intervention:
                twin_G.add_edge(f'X*{i}', f'X*{j}')
            elif j in varIDs_changed_by_intervention:
                twin_G.add_edge(i, f'X*{j}')
            else:
                assert i not in varIDs_changed_by_intervention and j not in varIDs_changed_by_intervention
        for i, parents in enumerate(selection_parents):
            for j in parents:
                if j in varIDs_changed_by_intervention:
                    twin_G.add_edge(f'X*{j}', f'S*{i}')
                else:
                    twin_G.add_edge(j, f'S*{i}')
        for j in varIDs_changed_by_intervention:
            twin_G.add_edge(f'E{j}', f'X*{j}')
            twin_G.add_edge(f'E{j}', j)
        twin_G.add_edges_from([(ZETA, j) for j in k_target])
        ci_tester_list.append(partial(boolean_tester_func, twin_G))

    return cdis_core(
        nodenum,
        ci_tester_list,
    )


def cdis_from_data(
    data_list,
    citest_method='fisherz',
    citest_alpha=0.05,
    max_cond_set_size=None,
    max_skeleton_refinement_length=-1,
    verbose=False
):
    '''
    :param data_list:
        a list of np.ndarray, each of shape (n_samples, n_vars).
        the first one is pure observational data; the rest are interventional data.
        `n_samples' of each dataset can be different; the sample individuals among datasets need not be matched.
    :param citest_method:
        str, the method to use in conditional independence test.
        default: 'fisherz'
        options: e.g., 'fisherz', 'kci'. for more, check causallearn.utils.cit.py
    :param citest_alpha:
        float, the significance level for conditional independence test.
    :param max_cond_set_size, max_skeleton_refinement_length, verbose:
        see description in cdis_core.

    :return:
        see description in cdis_core.
    '''
    boolean_tester_func = lambda pval_tester, alpha, x, y, Z: pval_tester(x, y, Z) > alpha
    pval_tester = cit.CIT(data_list[0], method=citest_method)
    ci_tester_list = [partial(boolean_tester_func, pval_tester, citest_alpha)]
    for k in range(1, len(data_list)):
        data_stacked = np.vstack([np.hstack([data_list[0], np.zeros((data_list[0].shape[0], 1))]),
                                  np.hstack([data_list[k], np.ones((data_list[k].shape[0], 1))])])
        pval_tester = cit.CIT(data_stacked, method=citest_method)
        ci_tester_list.append(partial(boolean_tester_func, pval_tester, citest_alpha))

    return cdis_core(
        data_list[0].shape[1],
        ci_tester_list,
        max_cond_set_size=max_cond_set_size,
        max_skeleton_refinement_length=max_skeleton_refinement_length,
        verbose=verbose
    )


if __name__ == '__main__':
    spsz = 5000

    print('In what follows we show several examples of <DAG, intervention, selection> configurations,\n'
          '    and see what causal realtions and selection mechanisms can be identified, from both oracle setting and real data.\n'
          '    One may check if the results from oracle and real data are consistent.\n')


    print('Eg1, I have two originally independent variables, 0 and 1, selected; if we only intervene on 0:')
    results = cdis_from_oracle_graph(nodenum=2, dag_edgelist=[], interv_targets=[(0,)], selection_parents=[(0, 1)])
    print('  [oracle] pure observational PAG:', results['pure_observational_PAG'])
    print('  [oracle] final PAG:', results['final_PAG'])
    X0_base = np.random.uniform(-1, 1, (spsz,))
    X1_base = np.random.uniform(-1, 1, (spsz,))
    selection = X0_base + X1_base > 0
    X0_star = X0_base[selection]
    X1_star = X1_base[selection]
    data_0 = np.vstack([X0_star, X1_star]).T
    X0 = X0_star + np.random.uniform(0.5, 1, (X0_star.shape[0],)) # intervention on 0
    data_1 = np.vstack([X0, X1_star]).T # X1 is unchanged
    results = cdis_from_data(data_list=[data_0, data_1], citest_method='fisherz', citest_alpha=0.05)
    print('  [data] pure observational PAG:', results['pure_observational_PAG'])
    print('  [data] final PAG:', results['final_PAG'])
    print('=> The relation between 0 and 1 is not identifiable; e.g., 0->S<-1 and 1->0 are both possible to produce the data.')

    print('\nEg2, I have two originally independent variables, 0 and 1, selected; if we have two datasets intervening on 0 and 1 respectively:')
    results = cdis_from_oracle_graph(nodenum=2, dag_edgelist=[], interv_targets=[(0,), (1,)], selection_parents=[(0, 1)])
    print('  [oracle] pure observational PAG:', results['pure_observational_PAG'])
    print('  [oracle] final PAG:', results['final_PAG'])
    X1 = X1_star + np.random.uniform(0.5, 1, (X1_star.shape[0],)) # intervention on 1
    data_2 = np.vstack([X0_star, X1]).T
    results = cdis_from_data(data_list=[data_0, data_1, data_2], citest_method='fisherz', citest_alpha=0.05)
    print('  [data] pure observational PAG:', results['pure_observational_PAG'])
    print('  [data] final PAG:', results['final_PAG'])
    print('=> This time, we can be sure about the existence of selection bias on 0 and 1.')

    print('\nEg3, I have 0->1->2 without selection, if we intervene on 0:')
    results = cdis_from_oracle_graph(nodenum=3, dag_edgelist=[(0,1), (1,2)], interv_targets=[(0,),], selection_parents=[])
    print('  [oracle] pure observational PAG:', results['pure_observational_PAG'])
    print('  [oracle] final PAG:', results['final_PAG'])
    E0 = np.random.uniform(-1, 1, (spsz,))
    E1 = np.random.uniform(-1, 1, (spsz,))
    E2 = np.random.uniform(-1, 1, (spsz,))
    X0 = E0
    X1 = 2 * X0 + E1
    X2 = 2 * X1 + E2
    data_0 = np.vstack([X0, X1, X2]).T
    X0 = X0 + np.random.uniform(0.5, 1, (X0.shape[0],)) # intervention on 0
    X1 = 2 * X0 + E1
    X2 = 2 * X1 + E2
    data_1 = np.vstack([X0, X1, X2]).T
    results = cdis_from_data(data_list=[data_0, data_1], citest_method='fisherz', citest_alpha=0.05)
    print('  [data] pure observational PAG:', results['pure_observational_PAG'])
    print('  [data] final PAG:', results['final_PAG'])
    print('=> Without selection (though we dont know apriori), we can identify the causal relations using only one intervention.')