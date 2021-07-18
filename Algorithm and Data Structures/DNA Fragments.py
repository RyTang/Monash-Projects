from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief 1:
The task of this code was to maintain a database that was used in a study for drug resistances. The database is used to
store Bacterial DNA collected from patients. Each bacterial DNA contains a drug resistant gene sequence. Researchers will 
need to be able to add new sequences and query existing sequences in the database. 

Queries return a string that satisfies the following conditions:
    - Must contain the sequence queried as the prefix
    - Have a higher frequency in the database than any other sequences with q as prefix
    - Return lexicographically least if multiple strings are tied for query

DNA sequence is typically denoted using A, T, C and G. However for ease of coding and proving that the functionalities work.
A, B, C and D will be used instead due to adjacent ASCII Values.

Methodology:
As I need to be able to satisfy the requirements of the Query function. I would need to maintain a frequency count of each 
inserted genome string while effectively maintaining the structure. Thus, the best way to do that will be the use of a Trie 
data structure. Currently a trie is currently the fastest way that I know of to efficiently insert and query for a given string. 
Simply enough, insert a genome string and if it exists update the frequency at the terminal node. Then to have an efficient query, 
each node will store a reference to the highest frequency and least lexicographically string. Thus, if a prefix is given the 
query will only require to traverse len(prefix) nodes to obtain the answer. The references are updated after each insertion 
of a genome string. This is done by backtracking from the terminal node to the root node and updating the references if 
the inserted node now has a higher frequency than the current highest.

"""


class Node:
    """
    A node used to store data payloads and containing links to connected nodes
    """

    def __init__(self, frequency=0, size: int = 27) -> None:
        """
        Creates a node of certain size with certain characteristics.

        :param frequency: Number of appearance for this terminal node
        :param size: Number of links that should be created per node
        :time_complexity: O(1)
        """
        # Terminal $ at index 0
        self.links = [None] * size
        self.previous = None
        # data payload
        self.frequency = frequency
        self.highest_link = None
        self.highest_index = size + 1
        self.key = None


class Trie:
    """
    A Trie specialised to store Characters using ASCII indexes
    """

    TRAVERSE_CREATE = 0
    TRAVERSE_QUERY = 1

    def __init__(self, size: int = 26, ascii_start_index: int = 97) -> None:
        """
        Creation of Trie with specifications from arguments.

        :param size: The number of links a node should have.
        :param ascii_start_index: The position of the smallest character used in the ASCII table.
        :time_complexity: O(1)
        """
        self.start_index = ascii_start_index
        self.node_size = size + 1
        self.root = Node(size=self.node_size)
        self.root.index = 0


    def insert(self, key: str) -> None:
        """
        Traverses the path given by the key and updates the frequency of said key in terminal node.

        :param key: Key that you would like to access
        :time_complexity: O(n) where n is len(key).
        """
        current = self.root
        # Go through key's path
        current = self.__traverse(current, key)[0]

        if current.links[0] is not None:  # Go through terminal
            current = current.links[0]
        else:  # Create Terminal
            temp = current
            current.links[0] = Node(frequency=0, size=0)  # Terminal node should have frequency
            current = current.links[0]
            current.key = key  # Store Key at terminal
            current.previous = temp  # Keep track of previous node
            current.index = 0
            current.highest_index = None

        # Update Frequency
        current.frequency += 1

        # Backtrack to root and update data
        self.__backtrack(current.frequency, current, current.previous)
        return


    def __traverse(self, current: Node1, key: str, mode: int = TRAVERSE_CREATE) -> Tuple[Node1, int]:
        """
        Traverses the key's path in the trie, while performing operations depending on "mode".

        :param current: The node from which you would like to begin from.
        :param key: The path you would like to travel through in the trie.
        :param mode: The mode of operation you would like to do, use modes listed at top of class.
        :return: A Tuple containing the current node after traversing and the number of traversals to this node.
        :time_complexity: O(n) where n is len(key).
        """
        trie_level = 0  # Is used to keep track of the Level of the trie currently on
        for char in key:
            index = ord(char) - self.start_index + 1
            if current.links[index] is not None:
                trie_level += 1
                current = current.links[index]
            elif mode == self.TRAVERSE_CREATE:  # Generate node when there is no linking node in Create mode
                temp = current
                current.links[index] = Node1(size=self.node_size)
                current = current.links[index]
                current.previous = temp
                current.index = index
            elif mode == self.TRAVERSE_QUERY:  # Stop search as key does not exist in Searching Mode
                break
        return current, trie_level


    def __backtrack(self, freq: int, link: Node1, start: Node1) -> None:
        """
        Backtracks from terminal to root, updates data along the way.

        :param freq: Current frequency of terminal node.
        :param link: Link to terminal node.
        :param start: From which node would you like to backtrack from.
        :time_complexity: O(n) where n is the number of nodes to traverse back to root.
        """
        current = start
        last_index = 0    # As last node was a terminal Node

        while current is not None:
            state1 = current.frequency < freq
            state2 = current.frequency == freq and current.highest_index >= last_index
            if state1 or state2:    # If the previous node I was at has a better link than current
                current.frequency = freq
                current.highest_index = last_index
                current.highest_link = link
            link = current.highest_link
            last_index = current.index
            current = current.previous


    def query(self, prefix: str) -> str:
        """
        Looks for a valid key with the highest frequency in lexicographical order.

        :param prefix: The start of a valid key that you would like to query.
        :return: Key with the highest frequency or None if there is none.
        :time_complexity: O(n) where n is len(rev_prefix)
        """
        current = self.root
        ret_str = None
        current, counter = self.__traverse(current, prefix)
        # If found a similar prefix
        if counter == len(prefix) and current.highest_link is not None:
            current = current.highest_link
            ret_str = current.key
        return ret_str


class SequenceDatabase:
    def __init__(self) -> None:
        """
        Instantiate a Trie data structure with fixed size.

        :time_complexity: O(1)
        """
        self.trie = Trie(size=4, ascii_start_index=65)
        return


    def addSequence(self, s: str) -> None:
        """
        Traverses the path given by the key and updates the frequency of said key in terminal node.

        :param s: Key that you would like to access
        :time_complexity: O(n) where n is len(s).
        """
        self.trie.insert(key=s)
        return


    def query(self, q: str) -> str:
        """
        Looks for a valid key with the highest frequency in lexicographical order.

        :param q: The start of a valid key that you would like to query.
        :return: Key with the highest frequency or None if there is none.
        :time_complexity: O(n) where n is len(q)
        """
        return self.trie.query(q)
