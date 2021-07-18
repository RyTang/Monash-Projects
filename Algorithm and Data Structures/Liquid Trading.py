import math
from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:
There is a trader who specialises in valuable liquids. The trader arrives in a new town, and they would like to trade
with the local people in order to maximise the liquid they currently have.

Each person in the village is willing to make various trades at various ratios. The trader knows the value of each type 
of liquid, thus being able to determine his current profit. The trader only has one container (with unlimited capacity),
that can only store one liquid at any given time. If liquids are mixed, they become worthless. Thus whenever a trade 
occurs, the trader must trade all their current liquid.

The trader is only in the village for a certain period of time, and trading takes time. Thus, there is a maximum number of
trades that can be done before the trader has to move on. The trader does not have to trade until they reach maximum number
of times. They just need to acquire the highest value they can with any liquid.

Data Format:
prices = [10, 5, 4, 3] -> There are 4 liquids available, liquid 1 has a price of 10 dollars per litre and so on.

townspeople = [[(1, 3, 0.5), (2, 3, 2)], [(3, 1, 1.5)]] -> Each nested list is a different villager with their own 
trade offerings. For example villager 1 is offering 2 trades, e.g one of his trade (1, 3, 0.5) means that he is willing
to trade between liquid 1 & 3 for a ratio of 0.5.

Methodology:
This whole situation can be mapped on to a graph and be solved with a graph algorithm, in this case Bellman-Ford. Each 
type of liquid will be a vertex on the graph and all the trades will be an edge between the 2 involved liquids with a 
weightage of the ratio. Thus starting at 1 litre of the starting liquid, Bellman-Ford can be adjusted to find the largest 
volume for each liquid types within max_trades. Then from the liquids with their largest volumes, iterate through them
and find the one with the highest value.
"""


class Graph:
    """Simple Graph Class with basic functionalities"""

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


def best_trades(prices: List[int], starting_liquid: int, max_trades: int, townspeople: List[List[Tuple[int, int, int]]]) -> float:
    """
    Finds the highest amount of profit within a certain number of trades within a group of traders.

    :param prices: Value of each available liquid
    :param starting_liquid: Liquid is available at beginning, have 1 litre at start.
    :param max_trades: Maximum number of trades available
    :param townspeople: Number of people offering their trades in the region
    :return: Returns the highest profit available
    :time_complexity: O(T * M) where T is the total number of trades available and M is max_trades
    """
    # Local Variables used
    old_volume, cur_volume = [], []
    trading = True
    num_trade = 0

    graph = Graph(len(prices))
    # Add Edges to graph
    for person in townspeople:
        for trade in person:
            graph.add_edges([trade])

    # Initialise starting value
    for i in range(len(prices)):
        if i == starting_liquid:
            old_volume += [1]
        else:
            old_volume += [0]
        cur_volume += [old_volume[-1]]

    while trading and num_trade < max_trades:   # Stop earlier if trading anymore doesn't make a difference
        trading = False
        for vertex in graph.vertices:           # Goes through every available trade
            for edge in vertex.edges:
                u, v, ratio = edge.u, edge.v, edge.w
                new_volume = old_volume[u] * ratio
                if new_volume > cur_volume[v]:
                    trading = True
                    cur_volume[v] = new_volume

        for i in range(len(old_volume)):    # Update Old Volume for next comparisons
            old_volume[i] = cur_volume[i]
        num_trade += 1

    # Get Highest Value in the list of volume/ Did it this way so you can know highest of each liquid.
    highest = 0
    for i in range(len(cur_volume)):
        price = cur_volume[i] * prices[i]
        if highest < price:
            highest = price
    return highest
