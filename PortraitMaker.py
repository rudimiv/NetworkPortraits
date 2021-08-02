import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def getDmatrixByGraph(G):
	'''
	Make D matrix for networkx graph object
	Input: networkx.Graph object
	Output: NumPy 2D-array with (N,N) shape
	'''
	Nodes=list(G.nodes())
	lengths=dict(nx.all_pairs_shortest_path_length(G))
	D=np.zeros((len(Nodes),len(Nodes)),dtype=np.int32) # До кого какое расстояние
	for V1 in range(len(Nodes)): # Заполняем справочник
		for V2 in range(V1+1,len(Nodes)):
			D[V1,V2]=lengths[Nodes[V1]][Nodes[V2]] # Обобщение на случай направленного графа
			D[V2,V1]=lengths[Nodes[V2]][Nodes[V1]]
	return D

def getPortraitByGraph(G,enable_stubs=False,force_max_l=0):
	'''
	Make graph portrait for networkx graph object
	Input: networkx.Graph object
	enable_stubs=False -- delete l=0,k=1 and l=d,k=0 components (not truth!)
	force_max_l -- force setup for max l
	Output: NumPy 2D-array with (d+1,N) shape
	'''
	Nodes=list(G.nodes())
	lengths=dict(nx.all_pairs_shortest_path_length(G))
	D=np.zeros((len(Nodes),len(Nodes)),dtype=np.int32) # До кого какое расстояние
	Fucknum=0
	for V1 in range(len(Nodes)): # Заполняем справочник
		for V2 in range(V1+1,len(Nodes)):
			try:D[V1,V2]=lengths[Nodes[V1]][Nodes[V2]] # Обобщение на случай направленного графа
			except KeyError as e:
				if(Fucknum<5):print("Fuck",e)
				Fucknum+=1
				D[V1,V2]=-1
			try:D[V2,V1]=lengths[Nodes[V2]][Nodes[V1]]
			except KeyError as e:
				if(Fucknum<5):print("Fuck",e)
				Fucknum+=1
				D[V2,V1]=-1
	if(Fucknum>5):print(Fucknum,"fuckups happened!")
	if(not nx.is_connected(G)):
		largest_cc = max(nx.connected_components(G), key=len)
		G=G.subgraph(largest_cc)
		print("Fuck: graph is not connected. The diameter is the diameter of largest part")
	d=nx.distance_measures.diameter(G) # 0≤l≤d
	res=np.zeros((max(force_max_l,d+1),len(Nodes)),dtype=np.int32) # 0≤k≤N-1
	for l in range(int(enable_stubs),d+1): # 0≤l≤d,0≤k≤N-1
		# Вырезаем ячейки с расстоянием l и считаем, сколько таких расстояний нашлось для каждого узла
		tmp=(D==l).astype(np.int32).sum(axis=0) # axis=1 ? Надо тестить
		for k in tmp:res[l,k]+=1
	if(enable_stubs):res[d,0]=0 # Для красоты: убирает особо крупные значения
	return res

import os,GraphMixer

from PIL import Image

def concatImg(inp,outp):
	images = [Image.open(x) for x in inp]
	widths, heights = zip(*(i.size for i in images))
	total_width = max(widths)
	max_height = sum(heights)
	new_im = Image.new('RGB', (total_width, max_height))
	y_offset = 0
	for im in images:
		new_im.paste(im, (0,y_offset))
		y_offset += im.size[1]
	new_im.save(outp)

def unMulti(M):
	G = nx.Graph()
	for u,v,data in M.edges(data=True):
		w = data['weight'] if 'weight' in data else 1.0
		if G.has_edge(u,v):
			G[u][v]['weight'] += w
		else:
			G.add_edge(u, v, weight=w)
	return G

def multiRandomize(G,outname,enable_stubs=True,colorbar=True,llim=None,stat=100):
	tit=outname
	if(type(G)==str):
		tit=G
		G=nx.read_graphml(G)
	if("/" in tit):tit=tit[tit.rindex("/")+1:]
	if(isinstance(G,nx.MultiGraph)):
		G=unMulti(G)
		print("Graph is MultiGraph. It will be converted to simple graph")
	if(G.is_directed()):
		G=G.to_undirected()
		print("Graph is directed. It will be converted to undirected.")
	if(not nx.is_connected(G)):
		largest_cc = max(nx.connected_components(G), key=len)
		G=G.subgraph(largest_cc)
		print("Graph is not connected. The largest part will be analysed")
	B=getPortraitByGraph(G,enable_stubs=enable_stubs,force_max_l=G.number_of_nodes())
	maxl=(B.sum(axis=1)>0).astype(np.int).sum()+1
	plt.matshow(B[:maxl])
	plt.xlabel('k, num of nodes')
	plt.ylabel('l, distance')
	if(llim is not None):plt.ylim(llim)
	plt.title(tit+"\nOriginal graph\n")
	if(colorbar):plt.colorbar()
	plt.savefig("/tmp/orig"+tit+".png")
	plt.close()
	Br=np.zeros( (G.number_of_nodes(),G.number_of_nodes()))
	counter=0
	A=nx.adjacency_matrix(G)
	print("Run mainloop...")
	for i in range(stat):
		try:
			tmp=nx.from_scipy_sparse_matrix(GraphMixer.adj_random_rewiring_iom_preserving(A,False))
			while(not nx.is_connected(tmp)):
				print("Warning: randomized graph is not connected. Retry...")
				tmp=nx.from_scipy_sparse_matrix(GraphMixer.adj_random_rewiring_iom_preserving(A,False))
			tmp=getPortraitByGraph(tmp,enable_stubs=enable_stubs,force_max_l=G.number_of_nodes())
			Br+=tmp
			counter+=1
		except KeyboardInterrupt:
			print("Stopped by user")
			break
	Br/=counter
	maxl=(Br.sum(axis=1)>0).astype(np.int).sum()+1
	plt.matshow(Br[:maxl])
	plt.xlabel('k, num of nodes')
	plt.ylabel('l, distance')
	if(llim is not None):plt.ylim(llim)
	plt.title("Randomized graph ("+str(counter)+" tries)\n")
	if(colorbar):plt.colorbar()
	plt.savefig("/tmp/rand"+tit+".png")
	plt.close()
	Diff=Br-B
	maxl=max(maxl,nx.distance_measures.diameter(G))
	plt.matshow(Diff[:maxl])
	plt.xlabel('k, num of nodes')
	plt.ylabel('l, distance')
	if(llim is not None):plt.ylim(llim)
	plt.title("Difference\n")
	if(colorbar):plt.colorbar()
	plt.savefig("/tmp/diff"+tit+".png")
	plt.close()
	concatImg(["/tmp/orig"+tit+".png","/tmp/rand"+tit+".png","/tmp/diff"+tit+".png"],outname)

def multiRandomize2(G,outname,enable_stubs=True,colorbar=True,llim=None,stat=100):
	tit=outname
	if(type(G)==str):
		tit=G
		G=nx.read_graphml(G)
	if("/" in tit):tit=tit[tit.rindex("/")+1:]
	if(isinstance(G,nx.MultiGraph)):
		G=unMulti(G)
		print("Graph is MultiGraph. It will be converted to simple graph")
	if(G.is_directed()):
		G=G.to_undirected()
		print("Graph is directed. It will be converted to undirected.")
	if(not nx.is_connected(G)):
		largest_cc = max(nx.connected_components(G), key=len)
		G=G.subgraph(largest_cc)
		print("Graph is not connected. The largest part will be analysed")
	B=getPortraitByGraph(G,enable_stubs=enable_stubs,force_max_l=G.number_of_nodes())
	maxl=(B.sum(axis=1)>0).astype(np.int).sum()+1
	Br=np.zeros( (G.number_of_nodes(),G.number_of_nodes()))
	counter=0
	A=nx.adjacency_matrix(G)
	print("Run mainloop...")
	for i in range(stat):
		try:
			tmp=nx.from_scipy_sparse_matrix(GraphMixer.adj_random_rewiring_iom_preserving(A,False))
			while(not nx.is_connected(tmp)):
				print("Warning: randomized graph is not connected. Retry...")
				tmp=nx.from_scipy_sparse_matrix(GraphMixer.adj_random_rewiring_iom_preserving(A,False))
			tmp=getPortraitByGraph(tmp,enable_stubs=enable_stubs,force_max_l=G.number_of_nodes())
			Br+=tmp
			counter+=1
		except KeyboardInterrupt:
			print("Stopped by user")
			break
	Br/=counter
	maxl=max( (Br.sum(axis=1)>0).astype(np.int).sum()+1 , maxl )
	Diff=Br-B
	maxl=max(maxl,nx.distance_measures.diameter(G))
	fig=showPortraits(
		[B[:maxl],Br[:maxl],Diff[:maxl]],
		[tit+"\nOriginal graph","\nRandomized graph ("+str(counter)+" tries)","\nDifference"],
		colorbar=True,show=False)
	fig.savefig(outname)
	return B,Br

def restoreSetBySum(S):
	'''
	Consider S_k=\Sigma\limits_{i=0}^{N-1} \delta(t_i,k)
	What is t_i ? This function try to answer and restore t_i set.
	Input: S(NumPy 1D array)
	Output: NumPy 1D array of ints with length N (0...N-1)
	'''
	tmp_rest=[]
	for k in range(len(S)):tmp_rest+=[k]*S[k]
	return np.array(tmp_rest)

def restoreDmatrixByPortrait(B): # TODO
	'''
	Try to restore D matrix by
	'''
	N=B[0,1]
	print("Graph has",N,"nodes")
	d=len(B)-1 # Diameter of Graph
	protoD=np.zeros((d+1,N),dtype=np.int32)
	print("Diameter is",d)
	for l in range(d+1):
		protoD[l]=restoreSetBySum(B[l])
	return protoD # TODO

def showPortrait(A,colorbar=False):
	plt.matshow(A)
	plt.xlabel('k, num of nodes')
	plt.ylabel('l, distance')
	if(colorbar):plt.colorbar()
	plt.show()

def showGraph(G):
	nx.draw(G)
	plt.show()

def showPortraits(As,titles,colorbar=False,show=True,p1=20,p2=3):
	fig, ax = plt.subplots(len(As), figsize=(p1, len(As)*p2))
	for i in range(len(As)):
		ax[i].set_title(titles[i])
		im=ax[i].pcolormesh(As[i])#, norm=colors.LogNorm(vmin=As[i].min(), vmax=As[i].max()))
		ax[i].set_xlabel("k, nodes")
		ax[i].set_ylabel("l, distances")
		fig.colorbar(im, ax=ax[i])
	if(show):plt.show()
	return fig

def speedtest(foo):
	import time
	start_t=time.time()
	for i in range(10):
		tmp=foo(resus)
	return time.time()-start_t
