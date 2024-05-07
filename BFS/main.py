# Python3 program to implement traveling salesman
# problem using naive approach.
import csv
from sys import maxsize
from itertools import permutations
import numpy as np
import timeit



#tworzenie grafow do testow
def graphFactory(n):
    graph =[]
    for i in range(n):
        graph.append(np.random.randint(20, 100, size=n))
        graph[i][i] = 0
    return graph





def travellingSalesmanProblem(graph, start, V):
    # dodanie wszystkich węzłów oprócz węzła startowego
    vertex = []
    for i in range(V):
        if i != start:
            vertex.append(i)

            # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    # znalezienie wszystkich permutacji
    next_permutation = permutations(vertex)
    #Obliczanie dlugosci kazdej z permutacji
    for sequence in next_permutation:
        # zachowaj current Path weight(cost)
        current_pathweight = 0

        # oblicz current path weight
        k = start
        for node in sequence:
            current_pathweight += graph[k][node]
            k = node
        current_pathweight += graph[k][start]

        # zaktualizuj minimum
        if current_pathweight < min_path:
            path = sequence
        min_path = min(min_path, current_pathweight)

    return min_path, path

all_times = []
#Mierzenie i zapisa czasy
'''''''''#
for nodeNum in range(12, 14):
    times = []
    V = nodeNum
    start = 0
    graph = graphFactory(nodeNum)
    file = open(f'graph{nodeNum}', 'w', newline='')
    writer = csv.writer(file)
    writer.writerows(graph)

    repeats = 10

    


    for i in range(50):
        timer = timeit.timeit(stmt='travellingSalesmanProblem(graph,start,V)', globals=globals(), number=repeats)
        times.append(timer/repeats)
    writer.writerow([travellingSalesmanProblem(graph, start, V)])
    file.close()
    all_times.append(times)

# matrix representation of graph
f = open('data2', 'w', newline='')

writer = csv.writer(f)

print(len(all_times))

for value in all_times:
    writer.writerows([value])


#print(travellingSalesmanProblem(graph,start))
'''''
#graph = graphFactory(6)
#min_path, sequence = travellingSalesmanProblem(graph, 0, 6)
#print(min_path, sequence)
print("Podaj nazwe pliku")
file_name =  input()
graph = []
#Wczytanie pliku typu csv
with open(file_name, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #konwersja typu na int
    for row in spamreader:
        temp_list = []
        for value in row:
           temp_list.append(int(value))
        graph.append(temp_list)


V = graph.pop(0)

min_path, sequence = travellingSalesmanProblem(graph, 0, int(V[0]))
sequence = list(sequence)

sequence.insert(len(sequence), 0)
sequence.insert(0, 0)

#printowanei wynikow
print(f'Minimalny dystans: {min_path}')
print("Trasa ", sequence)



