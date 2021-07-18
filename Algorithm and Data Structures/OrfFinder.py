from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:
In Molecular Genetics, there is a notion of Open Reading Frame (ORF). An ORF is a portion of DNA that is used as the blueprint for a protein. 
All ORFs start with a particular sequence and end with a particular sequence. 

The task is to find all sections of a genome which start with a given sequence of characters and end with a possibly different given sequence 
of characters 

DNA sequence is typically denoted using A, T, C and G. However for ease of coding and proving that the functionalities work. A, B, C and D will
be used instead due to adjacent ASCII Values.

Methodology:
The idea is to maintain the genome in a trie data structure for easier searching. In particular a prefix trie of the genome is created, followed
by a suffix trie of the reversed genome (An Example is given below to illustrate how this looks). In each node of the trie, there will be a record
that contains parts of genomes that has the related prefix or suffixes of the remainder trie. Then what needs to be done is to conduct a search of 
the starting genome in the prefix trie, and a search of the ending genome in the reversed suffix trie to retrieve 2 records from the different tries.
Afterwards a check is done to compare if there are any available combinations between the 2 records. 

Example. If the genome given was "ABCD":
Prefix Trie: ["ABCD", "BCD" , "CD", "D"]
Reversed Suffix Trie: ["DCBA", "CBA", "BA", "A"]

"""


class Node:
    """ A Class used to store data in a trie, contains links to connecting nodes"""

    def __init__(self, size: int = 27):
        """
        Creates a node of certain size with certain characteristics.

        :param size: Number of links that should be created per node
        :time_complexity: O(1)
        """
        # Terminal $ at index 0
        self.links = [None] * size
        # data payload
        self.indexes = []
        self.last_added = None


class PrefixSuffixTrie:
    """ A Class that creates a Prefix or Suffix Trie of the given word depending on the mode chosen."""

    MODE_SUFFIX = 0
    MODE_PREFIX_REVERSED = 1


    def __init__(self, word: str, mode: int = MODE_PREFIX_REVERSED, size: int = 26, ascii_start_index: int = 97) -> None:
        """
        Creation of Trie with specifications from arguments.

        :param word: The word that is to be used in the generation
        :param mode: Either make a rev_prefix or suffix trie, Default = suffix.
        :param size: The number of links a node should have.
        :param ascii_start_index: The position of the smallest character used in the ASCII table.
        :time_complexity: O(N^2) where N is len(word).
        """
        self.word = word
        self.mode = mode
        self.start_index = ascii_start_index
        self.node_size = size + 1
        self.root = Node(size=self.node_size)

        # Create Trie based on mode
        for i in range(len(self.word)):
            self.__insert(i)


    def __insert(self, length: int) -> None:
        """
        Inserts a word into the trie based on length of the word

        :param length: How long the word should currently be
        :time_complexity: O(n) where n is the len(word)
        """
        current = self.root
        start = length

        if self.mode == self.MODE_SUFFIX:
            end = len(self.word)
            step = 1
        else:
            end = 0 - 1
            step = -1

        # Traverse word
        for char_pos in range(start, end, step):
            char = self.word[char_pos]
            index = ord(char) - self.start_index + 1
            # if next node not available, create one
            if current.links[index] is None:
                current.links[index] = Node(size=self.node_size)
            # move to next node
            current = current.links[index]
            # Update node data
            if current.last_added is None or current.last_added != char_pos:
                current.indexes += [char_pos]
                current.last_added = char_pos
        return


    def search(self, key: str) -> List[int]:
        """
        Searches for a key given and returns the indexes that have that substring in the word.

        :param key: word that is to be searched in the trie.
        :return: The list of indexes that have the key in the trie.
        :time_complexity: O(n) where n is len(key)
        """
        current = self.root

        for char in key:
            index = ord(char) - self.start_index + 1
            if current.links[index] is None:
                return None
            else:
                current = current.links[index]
        return current.indexes


class OrfFinder:
    """ Used to find all possible strands in the genome given that begins and ends in a certain pattern efficiently"""

    def __init__(self, genome: str) -> None:
        """
        Create a Suffix Trie of the reversed genome and a normal Prefix Trie of the genome.

        :param genome: Genome that is to be searched on.
        :time_complexity: O(n^2) where n is len(genome).
        """
        self.genome = genome
        self.suffix = PrefixSuffixTrie(genome, mode=PrefixSuffixTrie.MODE_SUFFIX, size=4, ascii_start_index=65)
        self.prefix_rev = PrefixSuffixTrie(genome, mode=PrefixSuffixTrie.MODE_PREFIX_REVERSED, size=4, ascii_start_index=65)
        return


    def find(self, start: str, end: str) -> List[str]:
        """
        Finds all valid choices that begins with "start" and ends with "end" within the genome.

        :param start: Series of characters that the choice must begin with
        :param end: Series of characters that the choice must end with
        :return: All valid choices within given genome
        :time_complexity: O(len(start) + len(end) + N) where N is the number of combinations of start and end.
        """
        ar = []
        valid = True

        # Search start in suffix trie and get the list of indexes
        front = self.suffix.search(start)
        # Search end reversed in prefix reversed trie and get the list of indexes
        end_rev = end[::-1]
        back = self.prefix_rev.search(end_rev)
        if front is None or back is None:
            valid = False

        # Get possible strings with no overlap
        if valid:
            i, j = 0, len(back) - 1
            while i < len(front):
                if front[i] >= back[j] or j < 0:
                    j = len(back) - 1
                    i += 1
                else:
                    start_index = front[i] - len(start) + 1
                    end_index = back[j] + len(end)
                    ar += [self.genome[start_index:end_index]]
                    j -= 1
        return ar

