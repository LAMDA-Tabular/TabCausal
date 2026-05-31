#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Haoyue
@file: FCI_tools.py
@time: 3/11/2025
@desc:
"""

import networkx as nx
from itertools import chain, combinations, permutations, product
import copy
from collections import deque


AROW, DASH, CIRC = 'AROW', 'DASH', 'CIRC'
LEFT, RIGHT = 'LEFT', 'RIGHT'


def translate_PAG_dict_format(CURREDGES):
    pag_edges = {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
    for (node1, node2), (type1, type2) in CURREDGES.items():
        if type1 == DASH and type2 == AROW:
            pag_edges['->'].add((node1, node2))
        elif type1 == AROW and type2 == AROW:
            pag_edges['<->'].add((node1, node2))  # has symmetric repeats
        elif type1 == DASH and type2 == DASH:
            pag_edges['--'].add((node1, node2))  # has symmetric repeats
        elif type1 == CIRC and type2 == DASH:
            pag_edges['⚬--'].add((node1, node2))
        elif type1 == CIRC and type2 == AROW:
            pag_edges['⚬->'].add((node1, node2))
        elif type1 == CIRC and type2 == CIRC:
            pag_edges['⚬-⚬'].add((node1, node2))  # has symmetric repeats
    return pag_edges


def get_PAG_from_skeleton_and_sepsets(
    nodelist,
    skeleton_edges,
    sepsets,
    background_knowledge_edges=None,
    sure_no_latents=False,
    sure_no_selections=False,
    rules_to_use=None,
    verbose=False
):
    '''
    :param nodelist: enumerate of nodes
    :param skeleton_edges: enumerate of tuples of nodes; no need to be symmetric
    :param sepsets: dict{(i, j): S}; assert all (i, j) not in skeleton_edges should have a sepset
    :param background_knowledge_edges: a dictionary like {(i, j): (DASH, AROW)}
    :param sure_no_latents: boolean; if True, <-> edge is forbidden, e.g., all (CIRC, AROW) is oriented as (DASH, AROW)
    :param sure_no_selections: boolean; if True, -- edge is forbidden, e.g., all (DASH, CIRC) is oriented as (DASH, AROW)
    :param rules_to_use: None, of subset of [1, .., 10]
    :return:
    '''
    assert set().union(*skeleton_edges) <= set(nodelist)
    assert all(set(k) | set(v) <= set(nodelist) for k, v in sepsets.items())
    ALLNODES = set(nodelist)
    CURREDGES, SEPSETS = {}, {}
    UPDATEREASONS = {}
    for x, y in skeleton_edges: CURREDGES[(x, y)] = CURREDGES[(y, x)] = (CIRC, CIRC)
    for (node1, node2), Z in sepsets.items(): SEPSETS[(node1, node2)] = SEPSETS[(node2, node1)] = set(Z)
    assert len(set(CURREDGES.keys()) & set(SEPSETS.keys())) == 0
    assert set(CURREDGES.keys()) | set(SEPSETS.keys()) == {(x, y) for x, y in product(nodelist, nodelist) if x != y}
    if background_knowledge_edges is not None:
        assert set(background_knowledge_edges.keys()) <= set(CURREDGES.keys())
    UNSHIELDED_TRIPLES = set()
    UNSHIELDED_TRIPLE_EDGES = set()
    for x, y in combinations(nodelist, 2):
        for z in ALLNODES - {x, y}:
            if (x, y) not in CURREDGES.keys() and {(x, z), (y, z)} <= set(CURREDGES.keys()):
                UNSHIELDED_TRIPLES |= {(x, y, z), (y, x, z)}
                UNSHIELDED_TRIPLE_EDGES |= {(x, z), (y, z), (z, x), (z, y)}

    def get_curr_edge_type(node1, node2, end=LEFT):
        if (node1, node2) not in CURREDGES: return False
        if end == LEFT:
            return CURREDGES[(node1, node2)][0]
        elif end == RIGHT:
            return CURREDGES[(node1, node2)][1]
        assert False

    def update_edge(node1, node2, type1, type2, reason=None):
        new_type_1, new_type_2 = None, None
        curr_type_1, curr_type_2 = CURREDGES[(node1, node2)]
        if type1 is not None: # intend to update type1
            if curr_type_1 != CIRC and curr_type_1 != type1:
                if verbose:
                    # in real data, due to test errors, the CI results may be inconsistent with each other and with the graph, causing conflicts.
                    print(f"[WARNING] [INNER FCI] Conflict detected: Attempt to change '{curr_type_1}' to '{type1}' was not successful.")
            elif sure_no_latents and curr_type_2 == AROW and type1 == AROW:
                if verbose:
                    print(f"[WARNING] [INNER FCI] Conflict with sure_no_latents: Attempt to orient an <-> edge was not successful.")
            elif sure_no_selections and curr_type_2 == DASH and type1 == DASH:
                if verbose:
                    print(f"[WARNING] [INNER FCI] Conflict with sure_no_latents: Attempt to orient an -- edge was not successful.")
            elif curr_type_1 == CIRC:
                new_type_1 = type1 # finally safe, directly orient.

        if new_type_1 is not None: curr_type_1 = new_type_1
        if type2 is not None:
            if curr_type_2 != CIRC and curr_type_2 != type2:
                if verbose:
                    print(f"[WARNING] [INNER FCI] Conflict detected: Attempt to change '{curr_type_2}' to '{type2}' was not successful.")
            elif sure_no_latents and curr_type_1 == AROW and type2 == AROW:
                if verbose:
                    print(f"[WARNING] [INNER FCI] Conflict with sure_no_latents: Attempt to orient an <-> edge was not successful.")
            elif sure_no_selections and curr_type_1 == DASH and type2 == DASH:
                if verbose:
                    print(f"[WARNING] [INNER FCI] Conflict with sure_no_latents: Attempt to orient an -- edge was not successful.")
            elif curr_type_2 == CIRC:
                new_type_2 = type2 # finally safe, directly orient.

        if new_type_1 is not None or new_type_2 is not None:
            if new_type_1 is None: new_type_1 = curr_type_1
            if new_type_2 is None: new_type_2 = curr_type_2
            CURREDGES[(node1, node2)] = (new_type_1, new_type_2)
            CURREDGES[(node2, node1)] = (new_type_2, new_type_1)
            UPDATEREASONS[(node1, node2, new_type_1, new_type_2)] = UPDATEREASONS[(node2, node1, new_type_2, new_type_1)] = reason
            return True
        return False

    def _RO():
        for alpha, beta in combinations(ALLNODES, 2):  # safe here; it's symmetric
            if (alpha, beta) in CURREDGES: continue
            for gamma in ALLNODES - {alpha, beta}:
                if (alpha, gamma) in CURREDGES and (beta, gamma) in CURREDGES:
                    gamma_not_in_sepset = gamma not in SEPSETS[(alpha, beta)]
                    if gamma_not_in_sepset:
                        update_edge(alpha, gamma, None, AROW, reason='RO')
                        update_edge(beta, gamma, None, AROW, reason='RO')

    def _R1():
        # If α∗→ β◦−−∗ γ, and α and γ are not adjacent, then orient the triple as α∗→ β →γ
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if get_curr_edge_type(alpha, bt, RIGHT) == AROW]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, LEFT) == CIRC and (alpha, gm) not in CURREDGES]:
                    changed_something |= update_edge(beta, gamma, DASH, AROW, reason='R1')
        return changed_something

    def _R2():
        # If α→β∗→ γ or α∗→ β →γ, and α ∗−◦ γ,then orient α ∗−◦ γ as α∗→γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, RIGHT) == AROW]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, RIGHT) == AROW]:
                    if get_curr_edge_type(alpha, gamma, RIGHT) == CIRC:
                        if get_curr_edge_type(alpha, beta, LEFT) == DASH or \
                           get_curr_edge_type(beta, gamma, LEFT) == DASH:
                            changed_something |= update_edge(alpha, gamma, None, AROW, reason='R2')
        return changed_something

    def _R3():
        # If α∗→ β ←∗γ, α ∗−◦ θ ◦−∗ γ, α and γ are not adjacent, and θ ∗−◦ β, then orient θ ∗−◦ β as θ∗→ β.
        changed_something = False
        for alpha, gamma in combinations(ALLNODES, 2):  # safe here; it's symmetric
            if (alpha, gamma) in CURREDGES: continue
            for beta in [bt for bt in ALLNODES - {alpha, gamma} if
                         get_curr_edge_type(alpha, bt, RIGHT) == AROW and \
                         get_curr_edge_type(bt, gamma, LEFT) == AROW]:
                for theta in [th for th in ALLNODES - {alpha, beta, gamma} if
                         get_curr_edge_type(alpha, th, RIGHT) == CIRC and \
                         get_curr_edge_type(th, gamma, LEFT) == CIRC]:
                    if get_curr_edge_type(theta, beta, RIGHT) == CIRC:
                        changed_something |= update_edge(theta, beta, None, AROW, reason='R3')
        return changed_something

    def _R4():
        # If u = <θ, ...,α,β,γ> is a discriminating path between θ and γ for β, and β◦−−∗γ;
        # then if β ∈ Sepset(θ,γ), orient β◦−−∗ γ as β →γ; otherwise orient the triple <α,β,γ> as α ↔β ↔γ.
        changed_something = False
        for theta in ALLNODES:
            for gamma in ALLNODES - {theta}:
                if (theta, gamma) in CURREDGES: continue
                for beta in {bt for bt in ALLNODES - {theta, gamma} if
                             get_curr_edge_type(bt, gamma, LEFT) == CIRC}:
                    gamma_parents = {af for af in ALLNODES - {theta, gamma, beta} if \
                                     get_curr_edge_type(af, gamma, LEFT) == DASH and
                                     get_curr_edge_type(af, gamma, RIGHT) == AROW}
                    if len(gamma_parents) < 1: continue
                    # to prevent from nx.all_simple_paths(self.mag_undirected_graph, ..) (too slow)
                    # we use subgraph to only allow paths through gamma_parents
                    subgraph = nx.Graph()   # undirected
                    subgraph.add_nodes_from(gamma_parents | {theta, beta})
                    subgraph.add_edges_from([(x, y) for x, y in combinations(gamma_parents | {theta, beta}, 2) if (x, y) in CURREDGES])
                    for theta_beta_path in nx.all_simple_paths(subgraph, theta, beta):
                        if len(theta_beta_path) < 3: continue
                        path = theta_beta_path + [gamma]
                        if all(
                                get_curr_edge_type(path[i - 1], path[i], RIGHT) == AROW and
                                get_curr_edge_type(path[i], path[i + 1], LEFT) == AROW
                                for i in range(1, len(path) - 2)
                        ):
                            if beta in SEPSETS[(theta, gamma)]:
                                changed_something |= update_edge(beta, gamma, DASH, AROW, reason='R4')
                            else:
                                changed_something |= update_edge(path[-3], beta, AROW, AROW, reason='R4')
                                changed_something |= update_edge(beta, gamma, AROW, AROW, reason='R4')
        return changed_something

    def _R5():
        # For every (remaining) α◦−−◦β, if there is an uncovered circle path p =?α,γ,...,θ,β? between α and β s.t. α,θ are
        # not adjacent and β,γ are not adjacent, then orient α◦−−◦β and every edge on p as undirected edges (--)
        #  i.e., to ensure the graph remains chordal, no colliders allowed.
        changed_something = False
        current_circ_circ_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if types12 == (CIRC, CIRC) and x < y} & UNSHIELDED_TRIPLE_EDGES
        subgraph = nx.Graph()  # undirected
        subgraph.add_nodes_from(set().union(*current_circ_circ_edges_that_also_belongs_to_UTs))
        subgraph.add_edges_from(current_circ_circ_edges_that_also_belongs_to_UTs)
        for cycle in nx.cycle_basis(subgraph):
            if len(cycle) < 4: continue
            cycle_is_uncovered = all((cycle[nid - 1], cycle[((nid + 1) % len(cycle))]) not in CURREDGES for nid in range(len(cycle)))
            if cycle_is_uncovered:
                for nid in range(len(cycle)):
                    changed_something |= update_edge(cycle[nid - 1], cycle[nid], DASH, DASH, reason='R5')
        return changed_something

    def _R6():
        # If α—β◦−−∗ γ (α and γ may or may not be adjacent), then orient β◦−−∗ γ as β −−∗ γ.
        #  (this is because for any α—-β, both α and β must be ancestors of S)
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) == DASH]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, LEFT) == CIRC]:
                    changed_something |= update_edge(beta, gamma, DASH, None, reason='R6')
        return changed_something

    def _R7():
        # If α −−◦ β◦−−∗ γ, and α, γ are not adjacent, then orient β◦−−∗ γ as β −−∗ γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) == CIRC]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              (alpha, gm) not in CURREDGES and
                              get_curr_edge_type(beta, gm, LEFT) == CIRC]:
                    changed_something |= update_edge(beta, gamma, DASH, None, reason='R7')
        return changed_something

    def _R8():
        # If α→β →γ or α−−◦β →γ, and α◦→γ,orient α◦→γ as α→γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) in [AROW, CIRC]]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                        get_curr_edge_type(beta, gm, LEFT) == DASH and get_curr_edge_type(beta, gm, RIGHT) == AROW]:
                    if get_curr_edge_type(alpha, gamma, LEFT) == CIRC and get_curr_edge_type(alpha, gamma, RIGHT) == AROW:
                        changed_something |= update_edge(alpha, gamma, DASH, None, reason='R8')
        return changed_something

    def _R9():
        # If α→β →γ or α−−◦β →γ, and α◦−−∗γ,orient α◦−−∗γ as α−−∗γ.
        changed_something = False
        circ_arrow_edges_exists = any(types == (CIRC, AROW) for types in CURREDGES.values())
        if not circ_arrow_edges_exists: return False
        current_semi_directed_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if
             types12 not in [(AROW, AROW), (DASH, DASH)] and x < y} & UNSHIELDED_TRIPLE_EDGES
        pretend_directed_edges = []
        for x, y in current_semi_directed_edges_that_also_belongs_to_UTs:
            type1, type2 = CURREDGES[(x, y)]
            if type1 == DASH or type2 == AROW:
                pretend_directed_edges.append((x, y))
            elif type1 == AROW or type2 == DASH:
                pretend_directed_edges.append((y, x))
            else:
                assert type1 == CIRC and type2 == CIRC
                pretend_directed_edges.extend([(x, y), (y, x)])
        sub_directed_graph = nx.DiGraph()
        sub_directed_graph.add_edges_from(pretend_directed_edges)
        for alpha in sub_directed_graph.nodes():
            for gamma in [gm for gm in set(sub_directed_graph.nodes()) - {alpha} if
                          get_curr_edge_type(alpha, gm, LEFT) == CIRC and
                          get_curr_edge_type(alpha, gm, RIGHT) == AROW]:
                for path in nx.all_simple_paths(sub_directed_graph, alpha, gamma):
                    if len(path) < 4: continue
                    cycle_is_uncovered = all(
                        (path[nid - 1], path[((nid + 1) % len(path))]) not in CURREDGES for nid in range(len(path)))
                    if cycle_is_uncovered:
                        changed_something |= update_edge(alpha, gamma, DASH, None, reason='R9')
                        break
        return changed_something


    def _R10():
        # Suppose α◦→γ, β →γ ←θ, p1 is an uncovered p.d. path from α to β, and p2 is an uncovered p.d. path from α to
        # θ.Let μ be the vertex adjacent to α on p1 (μ could be β), and ω be the vertex adjacent to α on p2 (ω could be θ).
        # If μ and ω are distinct, and are not adjacent, then orient α◦→γ as α→γ.
        changed_something = False
        circ_arrow_edges_exists = any(types == (CIRC, AROW) for types in CURREDGES.values())
        dash_arrow_edges_exists = any(types == (DASH, AROW) for types in CURREDGES.values())
        if not (circ_arrow_edges_exists and dash_arrow_edges_exists): return False
        current_semi_directed_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if
             types12 not in [(AROW, AROW), (DASH, DASH)] and x < y} & UNSHIELDED_TRIPLE_EDGES
        pretend_directed_edges = []
        for x, y in current_semi_directed_edges_that_also_belongs_to_UTs:
            type1, type2 = CURREDGES[(x, y)]
            if type1 == DASH or type2 == AROW:
                pretend_directed_edges.append((x, y))
            elif type1 == AROW or type2 == DASH:
                pretend_directed_edges.append((y, x))
            else:
                assert type1 == CIRC and type2 == CIRC
                pretend_directed_edges.extend([(x, y), (y, x)])
        sub_directed_graph = nx.DiGraph()
        sub_directed_graph.add_edges_from(pretend_directed_edges)
        for alpha in sub_directed_graph.nodes():
            for gamma in [gm for gm in set(sub_directed_graph.nodes()) - {alpha} if
                          get_curr_edge_type(alpha, gm, LEFT) == CIRC and
                          get_curr_edge_type(alpha, gm, RIGHT) == AROW]:
                gamma_parents = {p for p in sub_directed_graph.nodes() if
                                 get_curr_edge_type(p, gamma, LEFT) == DASH and
                                 get_curr_edge_type(p, gamma, RIGHT) == AROW}
                already_done_orientation = False
                for beta, theta in combinations(gamma_parents, 2):
                    if already_done_orientation: break
                    for path1 in nx.all_simple_paths(sub_directed_graph, alpha, beta):
                        if already_done_orientation: break
                        for path2 in nx.all_simple_paths(sub_directed_graph, alpha, theta):
                            mu, omega = path1[1], path2[1]
                            if mu != omega and (mu, omega) not in CURREDGES:
                                already_done_orientation = True
                                changed_something |= update_edge(alpha, gamma, DASH, None, reason='R10')
                                break
        return changed_something

    def _R_no_latents():
        # when we are sure that there are no latents, we confirm all ⚬-> as ->
        changed_something = False
        for (node1, node2), (type1, type2) in CURREDGES.items():
            if type1 == CIRC and type2 == AROW:
                changed_something |= update_edge(node1, node2, DASH, AROW, reason='no_latents')
        return changed_something

    def _R_no_selections():
        # when we are sure that there are no selection, we confirm all ⚬-- as <-
        changed_something = False
        for (node1, node2), (type1, type2) in CURREDGES.items():
            if type1 == CIRC and type2 == DASH:
                changed_something |= update_edge(node1, node2, AROW, DASH, reason='no_selections')
        return changed_something


    # ============================= main part ======================================
    # first apply background knowledge (for now we dont do consistency check; just trust it)
    if background_knowledge_edges is not None:
        for (node1, node2), (type1, type2) in background_knowledge_edges.items():
            update_edge(node1, node2, type1, type2, reason='background')
            update_edge(node2, node1, type2, type1, reason='background')

    # then fix the unshielded triples using observed CIs
    _RO()

    # then iteratively apply the rules until no more changes
    rule_id_to_func = {1: _R1, 2: _R2, 3: _R3, 4: _R4, 5: _R5, 6: _R6, 7: _R7, 8: _R8, 9: _R9, 10: _R10}
    if rules_to_use is None: rules_to_use = list(range(1, 11))
    RULES = [rule_id_to_func[rule_id] for rule_id in rules_to_use]
    if sure_no_latents: RULES.append(_R_no_latents)
    if sure_no_selections: RULES.append(_R_no_selections)
    while True:
        changed_something = False
        for rule in RULES:
            changed_something |= rule()
        if not changed_something:
            break

    return CURREDGES, UPDATEREASONS


def get_skeleton_and_sepsets(
    nodelist,
    CI_tester,
    sure_adjacencies=None,
    sure_dependencies=None,
    max_cond_set_size=None,
    max_skeleton_refinement_length=None,
):
    '''
    :param nodelist: a list of nodes
    :param CI_tester: a function that takes in (i, j, S) and returns True or False; i, j in nodelist; S subset of nodelist
    :param sure_adjacencies: list of tuples of nodes that are known to be adjacent; skip tests on them; always i < j
    :param sure_dependencies: list of (i, j, frozenset(S)) tuples that are known to be dependent, always i < j
    :param max_cond_set_size: used for speeding up real data; maximum size of conditioning set to consider
    :param max_skeleton_refinement_length: used for speeding up real data.
            note: this skeleton refinement is needed for correctness, but is very time consuming.
                  the current implementation is still slow; can be improved a lot.
                  if in real data you want to forbid it, plz set max_skeleton_refinement_length=-1
    :return:
        skeleton: a list of tuples (i, j) representing edges in the skeleton; i < j always
        sepsets: a dictionary of the form {(i, j): S} where i indep j | S; i < j always
        dependencies: a set of (i, j, frozenset(S)) tuples of dependencies found; i < j always
    '''
    ALLNODES = sorted(nodelist)
    curr_skeleton = list(combinations(ALLNODES, 2))
    curr_neighbors = {i: set(ALLNODES) - {i} for i in ALLNODES}
    sure_adjacencies = {tuple(sorted(e)) for e in sure_adjacencies} if sure_adjacencies is not None else set()
    assert all(x in ALLNODES and y in ALLNODES for x, y in sure_adjacencies)
    sure_dependencies = {(min(x,y), max(x,y), S) for x,y,S in sure_dependencies} if sure_dependencies is not None else set()
    assert all(x in ALLNODES and y in ALLNODES and set(S) <= set(ALLNODES) for x,y,S in sure_dependencies)

    Sepsets = {}
    Dependencies = set(sure_dependencies)
    if max_cond_set_size is None: max_cond_set_size = len(ALLNODES) - 2
    l = -1
    while True:
        l += 1
        if l > max_cond_set_size: break
        found_something = False
        visited_pairs = set()
        while True:
            this_i, this_j = None, None
            for i, j in curr_skeleton:
                if (i, j) in visited_pairs: continue
                if (i, j) in sure_adjacencies: continue
                assert j in curr_neighbors[i]
                if len(curr_neighbors[i]) - 1 >= l or len(curr_neighbors[j]) - 1 >= l:
                    this_i, this_j = i, j
                    found_something = True
                    break
            if this_i is None: break
            visited_pairs.add((this_i, this_j))
            choose_subset_from = set(map(frozenset, combinations(curr_neighbors[this_i] - {this_j}, l))) | \
                                 set(map(frozenset, combinations(curr_neighbors[this_j] - {this_i}, l)))
            for subset in choose_subset_from:
                if (this_i, this_j, frozenset(subset)) in Dependencies: continue
                if CI_tester(this_i, this_j, subset):
                    curr_skeleton.remove((this_i, this_j))
                    curr_neighbors[this_i].remove(this_j)
                    curr_neighbors[this_j].remove(this_i)
                    Sepsets[(this_i, this_j)] = set(subset)
                    break
                else:
                    Dependencies.add((this_i, this_j, frozenset(subset)))
        if not found_something: break
    if max_skeleton_refinement_length == -1:
        return curr_skeleton, Sepsets, Dependencies

    ## so far it is done for PC (causal sufficiency case); however for FCI and MAG/PAG,
    ## we have to do skeleton refinement (some nonadjacencies may not have sepsets from just neighbors)
    ## - Step 0 (Initialization): Run PC's adjacency search to obtain
    ##          an undirected graph G (a supergraph of the true skeleton),
    ##          the adjacencies Adj0 derived from this initial graph,
    ##          and the recorded Sepsets.
    ## - Step 1 (Orient v-structures): Orient v-structure edges in G based on Sepsets, making G a partial DAG.  (note that other Meek orientations are not done here).
    ## - Step 2 (Initialize Possible-D-Sep):  For each node i, initialize Possible-D-Sep(i) as Adj0(i).
    ## - Step 3 (Adding more to Possible-D-Sep): For each pair of nonadjacent nodes i, j in G,
    ##          if {  there exists a path (w_1, w_2, ..., w_n) (n>=3, w_1=i, w_n=j) in G such that for every w_k (k = 2, ..., n-1),
    ##                  either w_k is a collider on path, or w_{k-1} and w_{k+1} are adjacent in G  },
    ##          then {  add j to Possible-D-Sep(i), and add i to Possible-D-Sep(j)  }.
    ## - Step 4 (Edge removal): For each pair of adjacent nodes i, j in G,
    ##          if {  there exists nodes S ∈ [powerset(Possible-D-Sep(i)\{j}) ∪ powerset(Possible-D-Sep(j)\{i})] \ [powerset(Adj0(i)\{j}) ∪ powerset(Adj0(j)\{i})] such that i⊥j|S  },
    ##          then {  remove the adjacency i--j from G, and let Sepsets(i,j)=S  }.
    to_further_remove = set()
    CURREDGES, _ = get_PAG_from_skeleton_and_sepsets(
        ALLNODES,
        curr_skeleton,
        Sepsets,
        background_knowledge_edges=None,
        sure_no_latents=False,
        sure_no_selections=False,
        rules_to_use=[],  # no R1 to R10 at all; instead, just R0 for vstrucs orientations
    )
    Adj_by_CURREDGES = {i: {j for j in ALLNODES if (i, j) in CURREDGES.keys()} for i in ALLNODES}
    Possible_Dsep = {i: Adj_by_CURREDGES[i].copy() for i in ALLNODES}
    for x in ALLNODES:
        # we only store valid (current_node, path_prefix) into this queue.
        queue = deque([(n, [x, n]) for n in Adj_by_CURREDGES[x]])
        while queue:
            current, path = queue.popleft()
            for neighbor in Adj_by_CURREDGES[current] - set(path):
                new_path = path + [neighbor] # since `path` prefix is valid, we only need to check the last 3 nodes
                a, b, c = new_path[-3:]
                if (CURREDGES[(a, b)][1] == AROW and CURREDGES[(b, c)][0] == AROW) or (c in Adj_by_CURREDGES[a]):
                    Possible_Dsep[x].add(neighbor)
                    Possible_Dsep[neighbor].add(x)
                    if max_skeleton_refinement_length is None or len(new_path) <= max_skeleton_refinement_length:
                        queue.append((neighbor, new_path))

    for this_i, this_j in curr_skeleton:
        if (this_i, this_j) in sure_adjacencies: continue  # we don't touch these forced adjacencies.
        upper_size = min(max_cond_set_size, max(len(Possible_Dsep[this_i]), len(Possible_Dsep[this_j])) - 1)
        already_removed = False
        for r in range(1, upper_size + 1):
            if already_removed: break
            choose_from_1 = set(map(frozenset, combinations(Possible_Dsep[this_i] - {this_j}, r))) if r < len(Possible_Dsep[this_i]) else set()
            choose_from_2 = set(map(frozenset, combinations(Possible_Dsep[this_j] - {this_i}, r))) if r < len(Possible_Dsep[this_j]) else set()
            for subset in choose_from_1 | choose_from_2:
                if (this_i, this_j, frozenset(subset)) in Dependencies: continue  # including those sure_dependencies
                if CI_tester(this_i, this_j, subset):
                    to_further_remove.add((this_i, this_j,))
                    curr_neighbors[this_i].remove(this_j)
                    curr_neighbors[this_j].remove(this_i)
                    Sepsets[(this_i, this_j)] = set(subset)
                    already_removed = True
                    break
                else:
                    Dependencies.add((this_i, this_j, frozenset(subset)))
    curr_skeleton = set(curr_skeleton) - to_further_remove
    return curr_skeleton, Sepsets, Dependencies