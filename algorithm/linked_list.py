class Node:
    def __init__(self, number, next=None):
        self.number = number
        self.next = next

    def addManyNodes(self, nodes):
        while self.next != None:
            self = self.next
        for node in nodes:
            nextNode = Node(node)
            self.next = nextNode
            self = nextNode
        return
        
    def addNodeFinal(self, node):
        while self.next != None:
            self = self.next
            continue
        self.next = node
        return
    
    def addNodeFirst(self, node):
        node.next = self
        return
    
    def addAfter(self, number, node):
        while(self.number != number):
            continue
        node.next = self.next
        self.next = node
        return
        
    def showList(self):
        print('The linked list is :')
        while self != None:
            print(self.number)
            self = self.next
        print('======================')
        return
    
    def getLength(self):
        c = 0
        while self != None:
            c += 1
            self = self.next
        return c

head = Node(1)
head.addManyNodes([2, 3, 4, 5, 6])
head.showList()
print(head.getLength())