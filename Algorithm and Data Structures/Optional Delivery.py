from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:
The situation is that we wish to travel from one city to another. However, travelling is costly and we want to get there
as cheaply as possible. There is a way to make money during the travel, we can opt to do delivery service where we must
pick up a parcel from a city and deliver it to a different city to earn some cash. Objective is to find the cheapest
way (including profit from delivery), and the path taken to get to the destination.

Data Format:


Methodology:
I first transform the given data into a graph format to implement one of the shortest-path algorithms. Each city will be
a vertex with the connecting roads being the edges, and the cost of travelling as their weightage. The way I've decided 
to solve this is to find the direct cheapest cost to the end destination, then compare it to a path with delivery. The 
algorithm involved is Dijkstra in this case as there are no negative edges and it's the most efficient algorithm for this
situation that I know of.
 
The only hard part in solving this problem is to get a functional min-heap to be used in the Dijkstra implementation. 
Dijkstra will be implemented to allow each city to record what the previous optimal city is. Thus at the end backtracking
can be done to construct the optimal path to destination from the starting city.

"""


class MinHeap:
    """A full working self-implemented Min-Heap"""

    def __init__(self, maxsize: int) -> None:
        """
        Initialise a heap of Certain Size

        :param maxsize: The maximum number of items in the heap allowed.
        :time_complexity: O(1)
        """
        self.maxsize = maxsize
        self.size = 0
        self.indexes = [None] * self.maxsize
        self.heap = [None] * (self.maxsize + 1)
        self.heap[0] = (-math.inf, -math.inf)


    def isLeaf(self, pos: int) -> bool:
        """
        Checks if a node is a leaf in the heap.

        :param pos: Which Node that you would like to look at
        :return: If it is a leaf or not
        :time_complexity: O(1)
        """
        if (self.size // 2) <= pos <= self.size:
            return True
        else:
            return False


    def swap(self, pos1: int, pos2: int) -> None:
        """
        Swaps the nodes in pos1 and pos2 in the heap.

        :param pos1: First node to be swapped
        :param pos2: Second node to be swapped
        :time_complexity: O(1)
        """
        self.indexes[self.heap[pos1][0]] = pos2
        self.indexes[self.heap[pos2][0]] = pos1
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]


    def sink(self, pos: int) -> None:
        """
        Moves an item down if it is larger than either of its child

        :param pos: Which node to start performing sink from
        :time_complexity: O(log n) where n is the number of nodes/items in the heap
        """
        # if leaf node or if there are no items then stop.
        if not self.isLeaf(pos) and self.size > 0:
            # Check if larger than child nodes and swap with the one that is smaller between the two
            if self.heap[pos][1] > self.heap[2*pos][1] or self.heap[pos][1] > self.heap[2*pos + 1][1]:
                if self.heap[2*pos][1] < self.heap[2*pos + 1][1]:
                    self.swap(pos, pos*2)
                    self.sink(2 * pos)
                else:
                    self.swap(pos, 2*pos + 1)
                    self.sink(2 * pos + 1)


    def rise(self, pos: int) -> None:
        """
        Continue moving an item upwards if it smaller than its parents

        :param pos: Node you would like to perform rise on
        :time_complexity: O(log n) where n is the number of nodes/items in the heap.
        """
        current = pos
        parent = current // 2
        while self.heap[current][1] < self.heap[parent][1]:
            self.swap(current, parent)
            current = parent
            parent = current // 2


    def insert(self, item: List[int]) -> None:
        """
        Inserts a new item in the heap and tries to bring it to its proper position.

        :param item: A item that contains two spots, comparisons will be made on the 2nd spot.
        :time_complexity: O(log n) where n is the number of nodes/items in the heap
        """
        self.size += 1
        self.heap[self.size] = item
        index = item[0]
        self.indexes[index] = self.size
        self.rise(self.size)


    def update(self, vertex, value) -> None:
        """
        Looks for a vertex in the heap, updates it with the new value and adjusts the minHeap.

        :param vertex: Vertex to be updated
        :param value: Value to be updated with
        :time_complexity: O(log n) where n is the number of nodes/items in the heap
        """
        index = self.indexes[vertex]
        self.heap[index][1] = value
        current = index
        self.rise(current)


    def getInfo(self, vertex: int) -> int:
        """
        For the context of this Assignment Q2, returns the distance from source to this vertex

        :param vertex: Vertex to be queried
        :return: Distance from source to vertex
        :time_complexity: O(1)
        """
        index = self.indexes[vertex]
        return self.heap[index][1]


    def serve(self) -> List[int]:
        """
        Returns the smallest item in the heap

        :return: The item with vertex and distance
        :time_complexity: O(log n) where n is the number of nodes/items in heap
        """
        serving = self.heap[1]
        self.swap(1, self.size)
        self.size -= 1
        self.sink(1)
        return serving


class Graph:
    """A Graph Class containing a Dijkstra method"""

    def __init__(self, num_vertices: int) -> None:
        """
        Creates a graph object with set number of vertices

        :param num_vertices: Number of vertices graph should have.
        :time_complexity: O(n) where n is the number of vertices
        """
        self.vertices = []
        for i in range(num_vertices):
            self.vertices += [Vertex(i)]


    def add_edges(self, edges: List[Tuple[int, int, int]]) -> None:
        """
        Adds in multiple edges into the graph with format for edge as (start, end, weight).

        :param edges: Edges to be placed in the graph, must contain valid vertices inside.
        :time_complexity: O(n) where n is the number of edges
        """
        for edge in edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            current = Edge(u, v, w)
            self.vertices[u].add_edge(current)


    def dijkstra(self, source: int) -> (List[int], List[int]):
        """
        Finds the cheapest path to each city from source

        :param source: Starting place
        :return: Distance to each city and record of previous city before entering city
        :time_complexity: O(E log V) where E is the number of edges and V is the number of vertices
        """
        min_heap = MinHeap(len(self.vertices))
        distance = [None] * len(self.vertices)
        previous_index = [None] * len(self.vertices)

        # Initialise Min Heap:
        for i in range(len(self.vertices)):
            self.vertices[i].visited = False    # Reset Vertices
            if i == source:
                min_heap.insert([i, 0])
            else:
                min_heap.insert([i, math.inf])

        # perform dijkstra
        while min_heap.size > 0:
            item = min_heap.serve()
            vertex_no = item[0]
            distance[vertex_no] = item[1]
            vertex = self.vertices[vertex_no]
            vertex.visited = True
            for edge in vertex.edges:
                destination = edge.v
                value = distance[vertex_no] + edge.w
                if not self.vertices[destination].visited and value < min_heap.getInfo(destination):
                    min_heap.update(destination, value)
                    previous_index[destination] = vertex.id
        return distance, previous_index


class Vertex:
    """Simple Vertex Class with basic vertex details"""

    def __init__(self, id: int) -> None:
        """
        Creates a vertex object

        :param id: Vertex's ID
        :time_complexity: O(1)
        """
        self.id = id
        self.visited = False
        self.edges = []


    def add_edge(self, edge):
        """
        Adds an edge to this vertex

        :param edge: Edge object with details
        :time_complexity: O(1)
        """
        self.edges.append(edge)


    def __str__(self):
        ret_str = str(self.id)
        return ret_str


class Edge:
    """Simple Edge Class with simple Edge Details"""

    def __init__(self, u: int, v: int, w: float):
        """
        Creates an Edge object with variables u = start, v = end and w = weight.

        :param u: start
        :param v: end
        :param w: weight
        :time_complexity: O(1)
        """
        self.u = u
        self.v = v
        self.w = w


    def __str__(self):
        ret_str = str([self.u, self.v, self.w])
        return ret_str


def opt_delivery(n: int, roads: List[Tuple[int, int, int]], start: int, end: int, delivery: Tuple[int, int, int]) -> Tuple[int, List[int]]:
    """
    Finds the cheapest route to get to the end destination from the start destination.

    :param n: Number of Cities available
    :param roads: Total number of roads between these cities with the format (city1, city2, cost)
    :param start: Starting City
    :param end: Ending City
    :param delivery: Delivery that can be done, in the format of (start, end, pay)
    :return: Returns the lowest cost in travelling to end city from start and the path taken to get there.
    time_complexity: O(R log N) where R is the total number of roads and N is the total number of cities.
    """
    # Local Variables Used
    delivered = False
    orig_tracks, start_delivery, end_delivery = None, None, None
    temp_path, final_path = [], []
    cost, cheapest = 0, 0
    graph = Graph(n)

    for road in roads:      # Add in roads, as it is undirected add the reversed in as well
        rev_path = [road[1], road[0], road[2]]
        graph.add_edges([road])
        graph.add_edges([rev_path])

    distance, orig_tracks = graph.dijkstra(start)
    cheapest = distance[end]    # Cost of without delivery

    # Try Delivering, Start Delivery
    if delivery[0] != start:        # if start of delivery is at a different city
        cost = distance[delivery[0]]
        distance, start_delivery = graph.dijkstra(delivery[0])
    cost += distance[delivery[1]]

    # Then finish delivery
    distance, end_delivery = graph.dijkstra(delivery[1])

    cost += distance[end]
    cost -= delivery[2]

    if cost < cheapest:  # Compares delivery vs no delivery
        cheapest = cost
        delivered = True

    # Backtracking, Constructing Path
    temp_path.append(end)
    if delivered:  # If delivery was done then multiple backtracking have to be done
        vertex = end_delivery[temp_path[-1]]
        while vertex is not None:   # Delivery's end city to ending destination
            temp_path.append(vertex)
            vertex = end_delivery[vertex]
        if start_delivery is not None:  # If delivery's start was not the starting city
            vertex = start_delivery[temp_path[-1]]
            while vertex is not None:
                temp_path.append(vertex)
                vertex = start_delivery[vertex]

    vertex = orig_tracks[temp_path[-1]]
    while vertex is not None:
        temp_path.append(vertex)
        vertex = orig_tracks[vertex]

    final_path = temp_path[::-1]     # Reverse the path to generate start:end path.
    return cheapest, final_path
