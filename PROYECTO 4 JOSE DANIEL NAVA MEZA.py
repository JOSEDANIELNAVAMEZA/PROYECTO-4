#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#CLASE NODO 

class Nodo(object):
    """
    Clase Nodo.
    Generación y manejo de Nodos para su uso en Grafos
    """
    def __init__(self, id):
        self.id = id
        self.attrs = dict()

    def __eq__(self, other):
        """
        Comparación de igualdad entre Nodos
        """
        return self.id == other.id

    def __lt__(self, other):
        """
        Checks if self.id is less than other.id
        """
        return self.id < other.id

    def __lt__(self, other):
        """
        Checks if self.id is greater than other.id
        """
        return self.id > other.id

    def __repr__(self):
        """
        Asigna representación repr a los Nodos
        """
        return repr(self.id)

    def __hash__(self):
        return hash(self.id)


# In[ ]:


#CLASE ARISTA

class Arista(object):
    """
    Clase Arista
    Generación y manejo de Aristas para su uso en Grafos
    Parametros
    """
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.id = (u.id, v.id)
        self.attrs = dict()

    def __eq__(self, other):
        """
        Comparación de igualdad entre Aristas
        """
        return self.u == other.u and self.v == other.v

    def __repr__(self):
        """
        Asigna representación repr a los Nodos
        """
        return repr(self.id)

    def __iter__(self):
      
       return Arista(self.u, self.v)

    def add_weight(self, weight):
        self.attrs['weight'] = weight


# In[ ]:


#CLASE GRAFO

import copy
import random
import sys

import heapdict
import numpy as np

from arista import Arista
from nodo import Nodo

class Grafo(object):

    def __init__(self, id='grafo', dirigido=False):
        self.id =       id
        self.dirigido = dirigido
        self.V =        dict()
        self.E =        dict()
        self.attr =     dict()

    def copy_grafo(self, id=f"copy", dirigido=False):
        """
        Regresa una copia deep del grafo
        """
        other = Grafo(id, dirigido)
        other.V = copy.deepcopy(self.V)
        other.E = copy.deepcopy(self.E)
        other.attr = copy.deepcopy(self.attr)

        return other

    def __repr__(self):
        """
        Asigna representación repr a los Grafos
        """
        return str("id: " + str(self.id) + '\n'
                   + 'nodos: ' + str(self.V.values()) + '\n'
                   + 'aristas: ' + str(self.E.values()))


    def add_nodo(self, nodo):
        """
        Agrega objeto nodo al grafo
        """
        self.V[nodo.id] = nodo


    def add_arista(self, arista):
        """
        Agrega arista al grafo si esta no existe de antemano en dicho grafo.
        """
        if self.get_arista(arista.id):
            return False

        self.E[arista.id] = arista
        return True


    def get_arista(self, arista_id):
        """
        Revisa si la arista ya existe en el grafo
        """
        if self.dirigido:
            return arista_id in self.E
        else:
            u, v = arista_id
            return (u, v) in self.E or (v, u) in self.E


    def random_weights(self):
        """
        Asigna un peso random a todas las aristas del nodo
        """
        for arista in self.E.values():
            arista.attrs['weight'] = random.randint(1, 100)

    def costo(self):
        """
        Calcula el costo del grafo. Suma del peso de las aristas
        """
        _costo = 0
        for edge in self.E.values():
            _costo += edge.attrs['weight']

        return _costo

    def to_graphviz(self, filename):
        """
        Exporta grafo a formato graphvizDOT
        """
        edge_connector = "--"
        graph_directive = "graph"
        if self.dirigido:
            edge_connector = "->"
            graph_directive = "digraph"

        with open(filename, 'w') as f:
            f.write(f"{graph_directive} {self.id} " + " {\n")
            for nodo in self.V:
                if "Dijkstra" in self.id:
                    f.write(f"\"{nodo} ({self.V[nodo].attrs['dist']})\";\n")
                else:
                    f.write(f"{nodo};\n")
            for arista in self.E.values():
                if "Dijkstra" in self.id:
                    weight = np.abs(self.V[arista.u.id].attrs['dist']
                                    - self.V[arista.v.id].attrs['dist'])
                    f.write(f"\"{arista.u} ({self.V[arista.u.id].attrs['dist']})\""
                            + f" {edge_connector} "
                            + f"\"{arista.v} ({self.V[arista.v.id].attrs['dist']})\""
                            # + f" [weight={weight}];\n")
                            + f";\n")
                else:
                    f.write(f"{arista.u} {edge_connector} {arista.v};\n")
            f.write("}")


    def BFS(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "breadth first
            search"
        """
        if not s.id in self.V:
            print("Error, node not in V", file=sys.stderr)
            exit(-1)

        bfs = Grafo(id=f"BFS_{self.id}", dirigido=self.dirigido)
        discovered = set()
        bfs.add_nodo(s)
        L0 = [s]
        discovered = set()
        added = [s.id]

        while True:
            L1 = []
            for node in L0:
                aristas = [ids_arista for ids_arista in self.E
                            if node.id in ids_arista]

                for arista in aristas:
                    v = arista[1] if node.id == arista[0] else arista[0]

                    if v in discovered:
                        continue

                    bfs.add_nodo(self.V[v])
                    bfs.add_arista(self.E[arista])
                    discovered.add(v)
                    L1.append(self.V[v])

            L0 = L1
            if not L0:
                break

        return bfs


    def DFS_R(self, u):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "depth first
            search".
        """
        dfs = Grafo(id=f"DFS_R_{self.id}", dirigido=self.dirigido)
        discovered = set()
        self.DFS_rec(u, dfs, discovered)

        return dfs

#6704
    def DFS_rec(self, u, dfs, discovered):
        """
        Función recursiva para agregar nodos y aristas al árbol DFS
        Parametros
        """
        dfs.add_nodo(u)
        discovered.add(u.id)
        aristas = (arista for arista in self.E if u.id in arista)

        for arista in aristas:
            v = arista[1]
            if not self.dirigido:
                v = arista[0] if u.id == arista[1] else arista[1]
            if v in discovered:
                continue
            dfs.add_arista(self.E[arista])
            self.DFS_rec(self.V[v], dfs, discovered)


    def DFS_I(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "depth first
            search"
        """
        dfs = Grafo(id=f"DFS_I_{self.id}", dirigido=self.dirigido)
        discovered = {s.id}
        dfs.add_nodo(s)
        u = s.id
        frontera = []
        while True:
            # añadir a frontera todos los nodos con arista a u
            aristas = (arista for arista in self.E if u in arista)
            for arista in aristas:
                v = arista[1] if u == arista[0] else arista[0]
                if v not in discovered:
                    frontera.append((u, v))

            # si la frontera está vacía, salir del loop
            if not frontera:
                break

            # sacar nodo de la frontera
            parent, child = frontera.pop()
            if child not in discovered:
                dfs.add_nodo(self.V[child])
                arista = Arista(self.V[parent], self.V[child])
                dfs.add_arista(arista)
                discovered.add(child)

            u = child

        return dfs


    def Dijkstra(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Dijkstra,
        que encuentra el grafo del camino más corto entre nodo
        """
        tree = Grafo(id=f"{self.id}_Dijkstra")
        line = heapdict.heapdict()
        parents = dict()
        in_tree = set()


        """
        asignar valores infinitos a los nodos
        """
        line[s] = 0
        parents[s] = None
        for node in self.V:
            if node == s:
                continue
            line[node] = np.inf
            parents[node] = None

        while line:
            u, u_dist = line.popitem()
            if u_dist == np.inf:
                continue

            self.V[u].attrs['dist'] = u_dist
            tree.add_nodo(self.V[u])
            if parents[u] is not None:
                arista = Arista(self.V[parents[u]], self.V[u])
                tree.add_arista(arista)
            in_tree.add(u)

            neigh = []
            for arista in self.E:
                if self.V[u].id in arista:
                    v = arista[0] if self.V[u].id == arista[1] else arista[1]
                    if v not in in_tree:
                        neigh.append(v)

            for v in neigh:
                arista = (u, v) if (u, v) in self.E else (v, u)
                if line[v] > u_dist + self.E[arista].attrs['weight']:
                    line[v] = u_dist + self.E[arista].attrs['weight']
                    parents[v] = u

        return tree

    def KruskalD(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Kruskal
        directo, que encuentra el árbol de expansión mínima
        """

        mst = Grafo(id=f"{self.id}_KruskalD")

        # sort edges by weight
        edges_sorted = list(self.E.values())
        edges_sorted.sort(key = lambda edge: edge.attrs['weight'])

        # connected component
        connected_comp = dict()
        for nodo in self.V:
            connected_comp[nodo] = nodo

        # add edges, iterating by weight
        for edge in edges_sorted:
            u, v = edge.u, edge.v
            if connected_comp[u.id] != connected_comp[v.id]:
                # add nodes and edge to mst
                mst.add_nodo(u)
                mst.add_nodo(v)
                mst.add_arista(edge)

                # change the connected component of v to be the same as u
                for comp in connected_comp:
                    if connected_comp[comp] == connected_comp[v.id]:
                        other_comp = connected_comp[v.id]
                        connected_comp[comp] = connected_comp[u.id]

                        # if we change the connected comp of one node,
                        # change it for the whole connected comp
                        iterator = (key for key in connected_comp                                     if connected_comp[key] == other_comp)
                        for item in iterator:
                            connected_comp[item] = connected_comp[u.id]

        return mst


    def KruskalI(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Kruskal
        inverso, que encuentra el árbol de expansión mínima
        """
        mst = self.copy_grafo(id=f"{self.id}_KruskalI", dirigido=self.dirigido)

        edges_sorted = list(self.E.values())
        edges_sorted.sort(key = lambda edge: edge.attrs['weight'], reverse=True)

        for edge in edges_sorted:
            u, v = edge.u.id, edge.v.id
            key, value = (u, v), edge
            del(mst.E[(u, v)])

            if len(mst.BFS(edge.u).V) != len(mst.V):
                mst.E[(u, v)] = edge

        return mst


    def Prim(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Prim,
        que encuentra el árbol de expansión mínima
        """
        mst = Grafo(id=f"{self.id}_Prim")
        line = heapdict.heapdict()
        parents = dict()
        in_tree = set()

        s = random.choice(list(self.V.values()))

        """
        asignar valores infinitos a los nodos
        """
        line[s.id] = 0
        parents[s.id] = None
        for node in self.V:
            if node == s.id:
                continue
            line[node] = np.inf
            parents[node] = None

        while line:
            u, u_dist = line.popitem()
            if u_dist == np.inf:
                continue

            self.V[u].attrs['dist'] = u_dist
            mst.add_nodo(self.V[u])
            if parents[u] is not None:
                arista = Arista(self.V[parents[u]], self.V[u])
                if (u, parents[u]) in self.E:
                    weight = self.E[(u, parents[u])].attrs['weight']
                else:
                    weight = self.E[(parents[u], u)].attrs['weight']
                arista.attrs['weight'] = weight
                mst.add_arista(arista)
            in_tree.add(u)

            neigh = []
            for arista in self.E:
                if self.V[u].id in arista:
                    v = arista[0] if self.V[u].id == arista[1] else arista[1]
                    if v not in in_tree:
                        neigh.append(v)

            for v in neigh:
                arista = (u, v) if (u, v) in self.E else (v, u)
                if line[v] > self.E[arista].attrs['weight']:
                    line[v] = self.E[arista].attrs['weight']
                    parents[v] = u

        return mst


# In[ ]:


import sys
import random

from grafo import Grafo
from arista import Arista
from nodo import Nodo

def grafoMalla(m, n, dirigido=False):
    """
    Genera grafo de malla
    """
    if m < 2 or n < 2:
        print("Error. m y n, deben ser mayores que 1", file=sys.stderr)
        exit(-1)

    total_nodes = m*n
    last_col = m - 1
    last_row = n - 1
    g = Grafo(id=f"grafoMalla_{m}_{n}", dirigido=dirigido)
    nodos = g.V

    # agregar nodos
    for id in range(total_nodes):
        g.add_nodo(Nodo(id))

    # agregar aristas
    # primera fila
    g.add_arista(Arista(nodos[0], nodos[1]))
    g.add_arista(Arista(nodos[0], nodos[m]))
    for node in range(1, m - 1):
        g.add_arista(Arista(nodos[node], nodos[node - 1]))
        g.add_arista(Arista(nodos[node], nodos[node + 1]))
        g.add_arista(Arista(nodos[node], nodos[node + m]))
    g.add_arista(Arista(nodos[m-1], nodos[m-2]))
    g.add_arista(Arista(nodos[m-1], nodos[m - 1 + m]))

    # filas [1 : n - 2]
    for node in range(m, total_nodes - m):
        col = node % m
        g.add_arista(Arista(nodos[node], nodos[node - m]))
        g.add_arista(Arista(nodos[node], nodos[node + m]))
        if col == 0:
            g.add_arista(Arista(nodos[node], nodos[node + 1]))
        elif col == last_col:
            g.add_arista(Arista(nodos[node], nodos[node - 1]))
        else:
            g.add_arista(Arista(nodos[node], nodos[node + 1]))
            g.add_arista(Arista(nodos[node], nodos[node - 1]))

    # última fila (n - 1)
    col_0 = total_nodes - m
    col_1 = col_0 + 1
    last_node = total_nodes - 1
    g.add_arista(Arista(nodos[col_0], nodos[col_1]))
    g.add_arista(Arista(nodos[col_0], nodos[col_0 - m]))
    for node in range(col_1, last_node):
        g.add_arista(Arista(nodos[node], nodos[node - 1]))
        g.add_arista(Arista(nodos[node], nodos[node + 1]))
        g.add_arista(Arista(nodos[node], nodos[node - m]))
    g.add_arista(Arista(nodos[last_node], nodos[last_node - m]))
    g.add_arista(Arista(nodos[last_node], nodos[last_node - 1]))

    return g

def grafoErdosRenyi(n, m, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Erdos-Renyi
    """
    if m < n-1 or n < 1:
        print("Error: n > 0 y m >= n - 1", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoErdos_Renyi_{n}_{m}")
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))

    # crear aristas
    rand_node = random.randrange
    for arista in range(m):
        while True:
            u = rand_node(n)
            v = rand_node(n)
            if u == v and not auto:
                continue
            if g.add_arista(Arista(nodos[u], nodos[v])):
                break

    return g

def grafoGilbert(n, p, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Gilbert
    """
    if p > 1 or p < 0 or n < 1:
        print("Error: 0 <= p <= 1 y n > 0", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoGilbert_{n}_{int(p * 100)}", dirigido=dirigido)
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))


    # crear pares de nodos, diferente generador dependiendo del parámetro auto
    if auto:
        pairs = ((u, v) for u in nodos.keys() for v in nodos.keys())
    else:
        pairs = ((u, v) for u in nodos.keys() for v in nodos.keys() if u != v)

    # crear aristas
    for u, v in pairs:
        add_prob = random.random()
        if add_prob <= p:
            g.add_arista(Arista(nodos[u], nodos[v]))

    return g

def grafoGeografico(n, r, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo geográfico simple
    """
    if r > 1 or r < 0 or n < 1:
        print("Error: 0 <= r <= 1 y n > 0", file=sys.stderr)
        exit(-1)

    coords = dict()
    g = Grafo(id=f"grafoGeografico_{n}_{int(r * 100)}", dirigido=dirigido)
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))
        x = round(random.random(), 3)
        y = round(random.random(), 3)
        coords[nodo] = (x, y)

    # crear aristas
    r **= 2
    for u in nodos:
        vs = (v for v in nodos if u != v)
        # si auto es true, se agrega la arista del nodo u a sí mismo
        if auto:
            g.add_arista(Arista(nodos[u], nodos[u]))
        # se agregan todos los nodos dentro de la distancia r
        for v in vs:
            dist = (coords[u][0] - coords[v][0]) ** 2                     + (coords[u][1] - coords[v][1]) ** 2
            if dist <= r:
                g.add_arista(Arista(nodos[u], nodos[v]))

    return g

def grafoBarabasiAlbert(n, d, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Barabasi-Albert
    """
    if n < 1 or d < 2:
        print("Error: n > 0 y d > 1", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoBarabasi_{n}_{d}", dirigido=dirigido)
    nodos = g.V
    nodos_deg = dict()

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))
        nodos_deg[nodo] = 0

    # agregar aristas al azar, con cierta probabilidad
    for nodo in nodos:
        for v in nodos:
            if nodos_deg[nodo] == d:
                break
            if nodos_deg[v] == d:
                continue
            p = random.random()
            equal_nodes = v == nodo
            if equal_nodes and not auto:
                continue

            if p <= 1 - nodos_deg[v] / d                and g.add_arista(Arista(nodos[nodo], nodos[v])):
                nodos_deg[nodo] += 1
                if not equal_nodes:
                        nodos_deg[v] += 1

    return g

def grafoDorogovtsevMendes(n, dirigido=False):
    """
    Genera grafo aleatorio con el modelo Barabasi-Albert
    """
    if n < 3:
        print("Error: n >= 3", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoDorogovtsev_{n}", dirigido=dirigido)
    nodos = g.V
    aristas = g.E

    # crear primeros tres nodos y sus correspondientes aristas
    for nodo in range(3):
        g.add_nodo(Nodo(nodo))
    pairs = ((u, v) for u in nodos for v in nodos if u != v)
    for u, v in pairs:
        g.add_arista(Arista(nodos[u], nodos[v]))

    # crear resto de nodos
    for nodo in range(3, n):
        g.add_nodo(Nodo(nodo))
        u, v = random.choice(list(aristas.keys()))
        g.add_arista(Arista(nodos[nodo], nodos[u]))
        g.add_arista(Arista(nodos[nodo], nodos[v]))

    return g


# In[ ]:


import random
from time import perf_counter

from grafo import Grafo
from arista import Arista
from nodo import Nodo
from generador_grafos import grafoMalla,                              grafoErdosRenyi,                              grafoGilbert,                              grafoGeografico,                              grafoBarabasiAlbert,                              grafoDorogovtsevMendes


def main():
    path = "/home/daniel/garbage/grafos/200/"

    nodos = 200
    nodos_malla = (20, 10)

    m_erdos = 1020
    p_gilbert = 0.15
    r_geografico = 0.3
    d_barabasi = 5

    print("\nMalla")
    g = grafoMalla(*nodos_malla)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

    print("\nErdos")
    g = grafoErdosRenyi(nodos, m_erdos)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

    print("\nGilbert")
    g = grafoGilbert(nodos, p_gilbert, dirigido=False, auto=False)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

    print("\nGeo")
    g = grafoGeografico(nodos, r_geografico)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

    print("\nBarabasi")
    g = grafoBarabasiAlbert(nodos, d_barabasi, auto=False)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

    print("\nDorog")
    g = grafoDorogovtsevMendes(nodos, dirigido=False)
    g.to_graphviz(path + g.id + ".gv")
    g.random_weights()
    kruskal = g.KruskalD()
    kruskal.to_graphviz(path + kruskal.id + ".gv")
    kruskalI = g.KruskalI()
    kruskalI.to_graphviz(path + kruskalI.id + ".gv")
    prim = g.Prim()
    prim.to_graphviz(path + prim.id + ".gv")
    print(f"costo kruskal: {kruskal.costo()}")
    print(f"costo kruskalI: {kruskalI.costo()}")
    print(f"costo prim: {prim.costo()}")

if __name__ == "__main__":
    main()

