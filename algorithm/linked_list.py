class Node:
    """
    Represents a node in a linked list.

    Attributes:
        number (any): The data stored in the node.
        next (Node, optional): The reference to the next node in the linked list. Default is None.

    Linked lists are a fundamental data structure used to store elements in a sequential manner.
    Unlike arrays, linked lists do not store elements in contiguous memory locations. Each element
    in a linked list is a separate object called a node, which stores the data and a reference to the
    next node in the list.

    Advantages of linked lists over arrays:
    1. Dynamic size: The size of a linked list can grow or shrink during the execution of the program.
    2. Efficient insertions/deletions: Adding or removing elements from a linked list is generally more
       efficient than in an array, especially at the beginning or in the middle.

    Disadvantages of linked lists compared to arrays:
    1. Memory usage: Each node in a linked list requires extra memory for storing the reference to the next node.
    2. No direct access: Linked lists do not allow direct access to elements by their position. Accessing an
       element requires traversing the list from the beginning.
    """

    def __init__(self, number, next=None):
        """
        Initializes a new Node.

        Args:
            number (any): The data to store in the node.
            next (Node, optional): The next node in the linked list. Default is None.
        """
        self.number = number
        self.next = next

    def addManyNodes(self, nodes):
        """
        Adds multiple nodes to the end of the linked list.

        Args:
            nodes (list): A list of values to create nodes with and add to the linked list.
        """
        while self.next != None:
            self = self.next
        for node in nodes:
            nextNode = Node(node)
            self.next = nextNode
            self = nextNode
        return
        
    def addNodeFinal(self, node):
        """
        Adds a single node to the end of the linked list.

        Args:
            node (Node): The node to add to the linked list.
        """
        while self.next != None:
            self = self.next
            continue
        self.next = node
        return
    
    def addNodeFirst(self, node):
        """
        Adds a node to the beginning of the linked list.

        Args:
            node (Node): The node to add at the start of the linked list.
        """
        node.next = self
        return
    
    def addAfter(self, number, node):
        """
        Adds a node after a specific node in the linked list.

        Args:
            number (any): The value of the node after which the new node is to be added.
            node (Node): The new node to add to the linked list.
        """
        while self.number != number:
            self = self.next
        node.next = self.next
        self.next = node
        
    def showList(self):
        """
        Prints the elements of the linked list.
        """
        print('The linked list is :')
        while self != None:
            print(self.number)
            self = self.next
        print('======================')
        return
    
    def getLength(self):
        """
        Calculates the length of the linked list.

        Returns:
            int: The number of nodes in the linked list.
        """
        c = 0
        while self != None:
            c += 1
            self = self.next
        return c

# Example of using the Node class to create and manipulate a linked list.
head = Node(1)
head.addManyNodes([2, 3, 4, 5, 6])
head.showList()
print(head.getLength())