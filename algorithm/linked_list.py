class Node:
    def __init__(self, number, next=None):
        self.number = number
        self.next = next
    
    def addNodeFinal(self, node):
        while self.next != None:
            self = self.next
            continue
        self.next = node
        return
        
    def showList(self):
        print('The linked list is :')
        while self != None:
            print(self.number)
            self = self.next
        print('======================')
        return

head = Node(1) 
head.showList()
child = Node(2)
head.addNodeFinal(child)
head.showList()
child2 = Node(3)
head.addNodeFinal(child2)
head.showList()