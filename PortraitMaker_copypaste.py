import os, sys
import numpy as np
import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Dima's code for portrait making

def network_portrait(G, trim_lengths=True, trim_numbers=False):
	# a = nx.algorithms.shortest_path_length(G)
	# all_pairs_shortest_path_length
	# all_pairs_dijkstra_path_length
	# all_pairs_bellman_ford_path_length
	lengths = dict(nx.all_pairs_shortest_path_length(G))
	# B_{ℓ,k} ≡ the number of nodes who have k nodes at distance ℓ
	res = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=np.int32)
	# print(lengths)
	for key, data in lengths.items():
		# print(key, data.values())
		# print(key, collections.Counter(data.values()))
		counters = collections.Counter(data.values())
		for dist in counters:
			if dist == 0:
				continue
			res[dist, counters[dist]] += 1
	if trim_lengths:
		res = res[~np.all(res == 0, axis=1)]
	# res = network_portrait(graph)
	# res = res[~np.all(res == 0, axis=0)]
	if trim_numbers:
		emptys = np.all(res == 0, axis=0)
		i = 0
		for e in emptys[::-1]:
			if not e:
				break
			i += 1
		res = res[:, :-i]
	return res

# Nikita's code for random

import copy,random

def get_single_double_edges_lists(g):
    L1 = []
    L2 = []
    h = nx.to_undirected(g).copy()
    for e in h.edges():
        if g.has_edge(e[1],e[0]):
            if g.has_edge(e[0],e[1]):
                L2.append((e[0],e[1]))
            else:
                L1.append((e[1],e[0]))
        else:
            L1.append((e[0],e[1]))
    return [L1,L2]

def random_rewiring_IOM_preserving(G, r=10):
# параметры: сеть, отношение числа пересоединенных пар к числу связей
    [L1, L2] = get_single_double_edges_lists(G) #лист обычных связей, лист mutual связей
    Number_of_single_edges = len(L1)
    Number_of_double_edges = len(L2)
    Number_of_rewired_1_edge_pairs = Number_of_single_edges*r
    Number_of_rewired_2_edge_pairs = Number_of_double_edges*r
    #Number_of_rewired_2_edge_pairs = 20
    i=0
    count = 0
    previous_text = ''
    #print len(set(L).intersection(List_of_edges))
    print ('Rewiring double connections...')
    while i < Number_of_rewired_2_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_double_edges-1)
        Edge_index_2 = random.randint(0, Number_of_double_edges-1)
        Edge_1 = L2[Edge_index_1]
        Edge_2 = L2[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (Node_A == Node_C) or (Node_A == Node_D) or (Node_B == Node_C) or (Node_B == Node_D):
            Edge_index_1 = random.randint(0, Number_of_double_edges-1)
            Edge_index_2 = random.randint(0, Number_of_double_edges-1)
            Edge_1 = L2[Edge_index_1]
            Edge_2 = L2[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2
        #print ('Edges:',Node_A, Node_B, ';',Node_C, Node_D)
        #print G.has_edge(Node_A, Node_B), G.has_edge(Node_B, Node_A), G.has_edge(Node_C, Node_D), G.has_edge(Node_D, Node_C)
        if G.has_edge(Node_A, Node_D) == 0 and G.has_edge(Node_D, Node_A) == 0 and G.has_edge(Node_C, Node_B) == 0 and G.has_edge(Node_B, Node_C) == 0:
            #try:
            try:
                w_ab = G.get_edge_data(Node_A, Node_B)['weight']
            except:
                pass
            G.remove_edge(Node_A, Node_B)
            G.remove_edge(Node_B, Node_A)
            '''
            except nx.NetworkXError:
                pass
                #print('fuck')
            '''
            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)['weight']
                except:
                    pass
                G.remove_edge(Node_C, Node_D)
                G.remove_edge(Node_D, Node_C)
            except nx.NetworkXError:
                pass
                #print('fuck')
            try:
                G.add_edge(Node_A, Node_D, weight = w_ab)
                G.add_edge(Node_D, Node_A, weight = w_ab)
            except:
                G.add_edge(Node_A, Node_D)
                G.add_edge(Node_D, Node_A)
            try:
                G.add_edge(Node_C, Node_B, weight = w_cd)
                G.add_edge(Node_B, Node_C, weight = w_cd)
            except:
                G.add_edge(Node_C, Node_B)
                G.add_edge(Node_B, Node_C)
            #print L2[Edge_index_1]
            L2[Edge_index_1] = (Node_A, Node_D)
            #print L2[Edge_index_1]
            #L2[Edge_index_1+1] = (Node_D, Node_A)
            L2[Edge_index_2] = (Node_C, Node_B)
            #L2[Edge_index_2+1] = (Node_B, Node_C)
            i += 1
        if (i != 0) and (i % (Number_of_double_edges//1)) == 0:
            text = str(round(100.0*i/Number_of_rewired_2_edge_pairs, 0)) + "%"
            if text != previous_text:
                #print text
                pass
            previous_text = text
    i = 0
    print ('Rewiring single connections...')
    while i < Number_of_rewired_1_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_single_edges-1)
        Edge_index_2 = random.randint(0, Number_of_single_edges-1)
        Edge_1 = L1[Edge_index_1]
        Edge_2 = L1[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (Node_A == Node_C) or (Node_A == Node_D) or (Node_B == Node_C) or (Node_B == Node_D):
            Edge_index_1 = random.randint(0, Number_of_single_edges-1)
            Edge_index_2 = random.randint(0, Number_of_single_edges-1)
            Edge_1 = L1[Edge_index_1]
            Edge_2 = L1[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2
        if G.has_edge(Node_A, Node_D) == 0 and G.has_edge(Node_D, Node_A) == 0 and G.has_edge(Node_C, Node_B) == 0 and G.has_edge(Node_B, Node_C) == 0:
            try:
                try:
                    w_ab = G.get_edge_data(Node_A, Node_B)['weight']
                except:
                    pass
                G.remove_edge(Node_A, Node_B)
            except nx.NetworkXError:
                print('fuck')
            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)['weight']
                except:
                    pass
                G.remove_edge(Node_C, Node_D)
            except nx.NetworkXError:
                print('fuck')
            try:
                G.add_edge(Node_A, Node_D, weight = w_ab)
            except:
                G.add_edge(Node_A, Node_D)
            try:
                G.add_edge(Node_C, Node_B, weight = w_cd)
            except:
                G.add_edge(Node_C, Node_B)
            L1[Edge_index_1] = (Node_A, Node_D)
            L1[Edge_index_2] = (Node_C, Node_B)
            i += 1
    G_rewired = copy.deepcopy(G)
    return G_rewired
