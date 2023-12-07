"""
Linked List Implementation in Python

Linked lists are a fundamental data structure in programming. They consist of a sequence of nodes, where each node contains data and a reference (or link) to the next node in the sequence. This structure allows for efficient insertion and deletion of elements, as it does not require reorganizing the entire structure, unlike arrays.

Pros of Linked Lists:
1. Flexibility in Size: They can grow and shrink in size during the program's execution.
2. Efficiency in Insertions/Deletions: Especially at the head of the list, where these operations are constant time.

Cons of Linked Lists:
1. Sequential Access: Accessing a specific element is less efficient than in an array, as it requires traversing the list from the start.
2. Memory Usage: Each node requires additional memory to store the link to the next node.

Comparison with Arrays:
- Arrays offer fast access to elements via indices, but resizing them or inserting/deleting elements can be inefficient.
- Linked Lists offer efficient insertions and deletions but with slower access to specific elements.
"""

class Node:
    """
    A Node in a linked list.
    """
    def __init__(self, data, next=None):
        self.data = data  # The data stored in the node
        self.next = next  # The reference to the next node

class linkedList:
    """
    A class for the Linked List.
    """
    def __init__(self, head=None):
        self.head = head  # Head of the list

    def addNodeFinal(self, node):
        """
        Adds a node at the end of the list.
        """
        current = self.head
        while current.next != None:
            current = current.next
        current.next = node  # Assign the new node to the next of the last node

    def addManyNodes(self, nodes):
        """
        Adds multiple nodes to the end of the list.
        """
        for node in nodes:
            self.addNodeFinal(Node(node))

    def addNodeFirst(self, node):
        """
        Adds a node at the beginning of the list.
        """
        node.next = self.head
        self.head = node  # Make the new node as the head

    def addAfter(self, number, node):
        """
        Adds a node after a specific node in the list.
        """
        current = self.head
        while current.data != number:
            current = current.next
        node.next = current.next
        current.next = node  # Insert the new node after the current node

    def showList(self):
        """
        Prints the entire list.
        """
        current = self.head
        print('The linked list is:')
        while current != None:
            print(current.data)
            current = current.next
        print('======================')

    def deleteNode(self, node):
        current = self.head
        while current != None:
            if current.next.data == node.data:
                current.next = current.next.next
                current.next
            
            current = current.next


    def getLength(self):
        """
        Returns the length of the list.
        """
        current = self.head
        length = 0
        while current != None:
            length += 1
            current = current.next
        return length

# Usage Example
# Creating a linked list and adding nodes
head = Node(1)
lista = linkedList(head)
lista.addManyNodes([2, 3, 4])
print("List after adding multiple nodes:")
lista.showList()

# Adding a node to the end
lista.addNodeFinal(Node(5))
print("List after adding a node to the end:")
lista.showList()

# Adding a node at the beginning
lista.addNodeFirst(Node(0))
print("List after adding a node at the beginning:")
lista.showList()

# Adding a node after a specific node
lista.addAfter(2, Node(2.5))
print("List after adding a node after the node with value 2:")
lista.showList()

# Showing the length of the list
print("Length of the list:", lista.getLength())
