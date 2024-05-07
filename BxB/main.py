import copy
import math
import numpy as np
from queue import PriorityQueue
import csv
import timeit


class Node:
    def __init__(self, path=[], reducedMatrix=None, cost=0, vertex=None, level=None):
        self.path = path
        self.reducedMatrix = reducedMatrix
        self.cost = cost
        self.vertex = vertex
        self.level = level

        # Define a custom comparison method
    def __lt__(self, other):
            # Compare nodes based on their `cost` attribute
            return self.cost < other.cost

def newNode(parentMatrix, path, level, i, j):
    node = Node()
    node.path = copy.deepcopy(path)
    if level == 0:
        cost, node.reducedMatrix = reducMatrix(np.copy(parentMatrix), True)
        node.cost = cost
    else:
        node.path.append([i, j])
        parentMatrix = setAxisix(parentMatrix, i, j)
        parentMatrix[j][0] = math.inf
        cost, node.reducedMatrix = reducMatrix(np.copy(parentMatrix), True)
        node.cost = cost
    node.reducedMatrix[j][0] = math.inf
    node.level = level
    node.vertex = j
    return node

def reducMatrix(Matrix, flag=False):
    cost = 0
    for idx, row in enumerate(Matrix):
        if np.any(row == 0):
            continue
        elif np.all(row == math.inf):
            continue
        else:
            red_val = min(row)
            Matrix[idx] = row - red_val
            cost += red_val

    Matrix = Matrix.transpose()

    for idx, row in enumerate(Matrix):
        if np.any(row == 0):
            continue
        elif np.all(row == math.inf):
            continue
        else:
            red_val = min(row)
            Matrix[idx] = row - red_val
            cost += red_val
    Matrix = Matrix.transpose()
    if flag:
        return cost, np.copy(Matrix)  # Ensure you create a deep copy
    return cost

def setAxisix(matrix, row, column):
    matrix[row].fill(math.inf)
    matrix = matrix.transpose()
    matrix[column].fill(math.inf)
    matrix = matrix.transpose()
    return matrix

def BxB(costMatrix):
    N = len(costMatrix)
    q = PriorityQueue()
    v = []
    root = newNode(np.copy(costMatrix), v, 0, -1, 0)
    #root.cost = reducMatrix(np.copy(root.reducedMatrix))

    q.put(root)

    while not q.empty():
        min = q.get()

        i = min.vertex

        #print(min.level)

        if min.level == N - 1:
            min.path.append([i, 0])
            print(min.path)
            return min.cost

        for j in range(N):
            if min.reducedMatrix[i][j] != math.inf:
                child = newNode(np.copy(min.reducedMatrix), min.path, min.level + 1, i, j)
                #val1 = min.costand min.reducedMatrix[i][j] != 0:
                #val2 = min.reducedMatrix[i][j]
                #reducMatrix(np.copy(child.reducedMatrix))

                child.cost = min.cost + min.reducedMatrix[i][j] + child.cost
                q.put(child)

    return 0



#print(BxB(matrix))


x = input("Podaj plik inicjalizujacy: ")

with open(x, 'r') as f:
    list_of_instances = f.readlines()



for ind, row in enumerate(list_of_instances):
    list_of_instances[ind] = row.split(' ')

all_times = []

for instance in list_of_instances:
    matrix = []
    with open(instance[0], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # konwersja typu na int
        for row in spamreader:
            temp_list = []
            for value in row:
                temp_list.append(int(value))
            matrix.append(temp_list)

    matrix.pop(0)
    # ustaw nieskonczonosci na przekatnej
    #print(matrix)
    for i in range(len(matrix)):
        matrix[i][i] = math.inf

    matrix = np.array(matrix)
    times = []
    for i in range(int(instance[1])):
        timer = timeit.timeit(stmt='print(BxB(matrix))', globals=globals(), number=1)
        times.append(timer)
    all_times.append(times)
    #print("Jedna z instancji obliczona!")

f = open('output.csv', 'w', newline='')

writer = csv.writer(f)

#print(all_times)

for value in all_times:
    writer.writerows([value])
    '''''''''''''''
    print("___________________________________")
    print(f'wartość optymalna: {instance[1]}')
    print(f'Obliczone wartośći: {BxB(matrix)}')
    print("___________________________________")
    '''''''''
