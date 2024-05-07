# Simulated Annealin Michał Pawlus
import math
import random
import csv
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from statistics import mean
from tqdm.auto import tqdm


def calculate_path_length(path, graph):
    length = 0
    for i in range(len(path) - 1):
        length += graph[path[i]][path[i+1]]
    length += graph[path[-1]][path[0]]  # Wrapping back to the starting city
    return length
def greedy_search(graph):
    temp_graph = graph.copy()
    path = []
    start = 0
    path.append(start)
    for i in range(len(graph)):
        temp_graph[i][start] = float('inf')
    for ind in range(len(graph)-1):

            next = temp_graph[start].argmin()
            for i in range(len(graph)):
                temp_graph[i][next] = float('inf')
            path.append(next)
            start=next
            # print(temp_graph)
    return path

def generate_neighbour_swap(solution):
    neighbour = solution.copy()
    i = random.randint(0, len(solution)-1)
    j = random.randint(0, len(solution)-1)
    neighbour[i], neighbour[j] = neighbour[j], neighbour[i]

    return neighbour


def generate_neighbour_invert(solution):
    # Kopiujemy rozwiązanie, aby nie zmieniać oryginalnego
    neighbour = solution.copy()

    # Losujemy indeksy i inwertujemy elementy w wybranym zakresie
    i = random.randint(0, len(solution) - 1)
    j = random.randint(0, len(solution) - 1)
    i, j = min(i, j), max(i, j)  # Upewniamy się, że i <= j
    neighbour[i:j + 1] = reversed(neighbour[i:j + 1])

    return neighbour

def init_test(graph):
    return list(range(len(graph)))

def accept_prob(curr_cost, neighbour_cost, temperature):
    if curr_cost > neighbour_cost:
        return 1.0
    return math.exp((curr_cost-neighbour_cost)/temperature)
def anneling(graph, n, initial_temperature=100000, cooling_rate=0.999, end_temp=0.1):

    initial_temperature=calculate_initial_temperature(graph)
    #print(f"temp początkowa: {initial_temperature}")
    current_solution = init_test(graph)
    best_solution = current_solution.copy()
    temperature = initial_temperature

    #iteration_count= 0
    rangeN = (math.ceil((n * (n - 1)) / 2))/10
    while temperature > end_temp:
        neighbour = generate_neighbour_invert(current_solution)
        current_cost = calculate_path_length(current_solution, graph)
        neighbour_cost = calculate_path_length(neighbour, graph)
        #print(f"current solution cost : {current_cost}")
        #print(f"neighbour solution cost : {neighbour_cost}")


        for i in range(int(rangeN)):
            if neighbour_cost < calculate_path_length(best_solution, graph):
                best_solution = neighbour
                #epochs_to_break = 0
                #print("dokonano zmiany")
                #break

            if accept_prob(current_cost, neighbour_cost, temperature) > random.random():
                #print(accept_prob(current_cost, neighbour_cost, temperature))
                #print("zmiana ze current sol")
                current_solution = neighbour
                #epochs_to_break = 0
                #break

        #temp geometryczna
        temperature *= cooling_rate
        #temp cauchy
        #temperature = initial_temperature/(1+(cooling_rate*iteration_count))

        #temperature = initial_temperature/(1+(cooling_rate*iteration_count))
        #print(f"c: {cooling_rate}, k: {iteration_count}")
        #print(f"temperatura po ochlodzeniu:{temperature}")
        #iteration_count += 1

        #if temperature < end_temp:
            #print("Doszlismy do temp .1")
            #break
        #print("__________________________________________")
    return best_solution, calculate_path_length(best_solution, graph)




def draw_plot(names, labels):
    y_axis_values = []
    x_axis_values = [17,21, 24, 70, 53, 202, 150, 318, 400 ,666]
    for name in names:
        with open(name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            # konwersja typu na int
            list_of_avrg = []
            for row in spamreader:

                temp_list = []
                for value in row:
                    temp_list.append(float(value))
                    #print(f"value: {value}, type: {type(float(value))}, flaot: {float(value)} ")

                list_of_avrg.append(mean(temp_list))

            y_axis_values.append(list_of_avrg)
    """
    if len(names) == 1:
        g = plt.figure(1, figsize=(9, 6))
        y_plot_list = []
        for j in range(len(y_axis_values[0])):
            if j % 2 == 1:
                #print(f"i={i}, j={j}")
                y_plot_list.append(y_axis_values[0][j])
        print(f"dlugosc: {len(y_plot_list)}")
        print(y_plot_list)
        plt.scatter(x_axis_values[:5], y_plot_list[:5]).set_label("TSP")
        plt.scatter(x_axis_values[5:], y_plot_list[5:]).set_label("ATSP")
        plt.title('Errors')
        plt.xlabel("n")
        plt.ylabel("error[%]")
        g.legend()
        g.show()

        f = plt.figure(2, figsize=(9, 6))
        y_plot_list = []
        for j in range(len(y_axis_values[0])):
            if j % 2 == 0:
                # print(f"i={i}, j={j}")
                y_plot_list.append(y_axis_values[0][j])
        print(f"dlugosc: {len(y_plot_list)}")
        print(y_plot_list)
        plt.scatter(x_axis_values[:5], y_plot_list[:5]).set_label("TSP")
        plt.scatter(x_axis_values[5:], y_plot_list[5:]).set_label("ATSP")
        plt.title('Time')
        plt.xlabel("n")
        plt.ylabel("T[s]")
        plt.legend()
        plt.show()

        return 0
    """
    g = plt.figure(1, figsize=(9, 6))
    print(f"len of x: {len(x_axis_values)} | len of y: {len(y_axis_values[0])}")
    print(f"y_axis_values: {y_axis_values}")
    for i in range(len(names)):
        y_plot_list = []
        for j in range(len(y_axis_values[i])):
            if j % 2 == 1:
                print(f"i={i}, j={j}")
                y_plot_list.append(y_axis_values[i][j])
        print(f"dlugosc: {len(y_plot_list)}")
        print(y_plot_list)
        plt.scatter(x_axis_values, y_plot_list).set_label(labels[i])

    plt.title('Errors')
    plt.xlabel("n")
    plt.ylabel("error[%]")
    g.legend()
    g.show()

    f = plt.figure(2, figsize=(9, 6))
    for i in range(len(names)):
        y_plot_list = []
        for j in range(len(y_axis_values[i])):
            if j % 2 == 0:
                print(f"i={i}, j={j}")
                y_plot_list.append(y_axis_values[i][j])
        print(f"dlugosc: {len(y_plot_list)}")
        print(y_plot_list)
        plt.scatter(x_axis_values, y_plot_list).set_label(labels[i])
    plt.title('Time')
    plt.xlabel("n")
    plt.ylabel("T[s]")
    plt.legend()
    plt.show()


def draw_plot_outputs(names, labels):
    y_axis_values = []
    x_axis_values = [[1,2,3,4,5,6,7,8,9,10,11,12,13], [2,3,4,5,6,7,8,9,10,12,13,14,15,18,19,20], [17,29,48,58,76,96,100,130,150, 34,45,171], [17,21,24,70,53,202,150,318,400,666]]
    for name in names:
        with open(name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            # konwersja typu na int
            list_of_avrg = []
            for row in spamreader:

                temp_list = []
                for value in row:
                    temp_list.append(float(value))
                    # print(f"value: {value}, type: {type(float(value))}, flaot: {float(value)} ")

                list_of_avrg.append(mean(temp_list))

            y_axis_values.append(list_of_avrg)



    g = plt.figure(1, figsize=(9, 6))
    print(f"len of x: {len(x_axis_values)} | len of y: {len(y_axis_values[0])}")
    print(f"y_axis_values: {y_axis_values}")
    for i in range(len(names)):
        y_plot_list = []
        if i == 0:
            plt.scatter(x_axis_values[i], y_axis_values[i]).set_label(labels[i])
        elif i == 1:
            plt.scatter(x_axis_values[i], y_axis_values[i]).set_label(labels[i])
        else:
            for j in range(len(y_axis_values[i])):
                if j % 2 == 0:
                    print(f"i={i}, j={j}")
                    y_plot_list.append(y_axis_values[i][j])
            print(f"dlugosc: {len(y_plot_list)}")
            print(y_plot_list)
            plt.scatter(x_axis_values[i], y_plot_list).set_label(labels[i])

    plt.title('Time')
    plt.xlabel("n")
    plt.ylabel("T[s]")
    g.legend()
    g.show()
    plt.show()






def calculate_initial_temperature(graph, iterations=100, initial_temperature=10000):
    total_diff = 0
    current_solution = greedy_search(graph)
    for _ in range(iterations):

        current_cost = calculate_path_length(current_solution, graph)

        neighbour = generate_neighbour_invert(current_solution)
        neighbour_cost = calculate_path_length(neighbour, graph)

        total_diff += abs(current_cost - neighbour_cost)
        current_solution = neighbour
    average_diff = total_diff / iterations
    initial_temperature = -average_diff / math.log(0.9)  # Wartość 0.9 można dostosować

    return initial_temperature



def convertTSP(tspfile):
    allStrings = []
    allStringsCorrect = []
    with open(tspfile, 'r') as f:
        allStrings = f.readlines()
        for i in range(len(allStrings)):
            string = allStrings[i]
            ind = 0
            while True:
                if string[ind] == " " and ind == 0:
                    string = string[1:]
                    #print(string)
                    ind = ind-1
                elif string[ind] == " " and string[ind+1] != ' ':
                    string = string[:ind] +',' + string[ind+1:]
                    #print(string)
                    ind = ind-1
                elif string[ind] == " " and string[ind+1] == ' ':
                    string = string[:ind] + string[ind+1:]
                    #print(string)
                    ind = ind -1
                elif string[ind] == "\n":
                    #string = string[:-1]
                    break;
                ind+=1
            allStringsCorrect.append(string)

    with open(f"{tspfile}_converted.csv", 'w') as f:
        f.writelines(allStringsCorrect)


    return allStringsCorrect

"""if __name__ == '__main__':
    graph = [[float('inf'), 30, 20, 31, 28, 40],
             [30, float('inf'), 10, 14, 20, 44],
             [40, 20, float('inf'), 10, 22, 50],
             [41, 24, 20, float('inf'), 14, 42],
             [38, 30, 32, 24, float('inf'), 28],
             [50, 54, 60, 52, 38, float('inf')]]

    graph = np.array(graph)
    path = greedy_search(graph)
    print(path)
    path2, cost = anneling(graph)
    print(path2, cost)

#print(convertTSP("tsp_45.txt"))
#print(accept_prob(132,304,10000));

"""

"""
x = input("Podaj plik inicjalizujacy: ")

with open(x, 'r') as f:
    list_of_instances = f.readlines()



for ind, row in enumerate(list_of_instances):
    list_of_instances[ind] = row.split(' ')

all_times = []
errors = []
instances = []
all_errors = []

for instance in tqdm(list_of_instances):
    matrix = []
    with open(instance[0], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # konwersja typu na int
        for row in spamreader:
            temp_list = []
            for value in row:
                temp_list.append(int(value))
            matrix.append(temp_list)

    n = matrix.pop(0)
    instances.append(n[0])
    # ustaw nieskonczonosci na przekatnej
    #print(matrix)
    for i in range(len(matrix)):
        matrix[i][i] = math.inf

    matrix = np.array(matrix)
    #print(n[0])
    #print(type(n[0]))
    #sol, sol_len = anneling(matrix, n[0])
    times_temp = []
    errors_temp = []
    avarage_error = 0
    for i in range(int(instance[1])):
        timer_start = timer()
        sol, sol_len = anneling(matrix, n[0])
        timer_end = timer()
        times_temp.append(timer_end-timer_start)
        error_temp = ((sol_len-int(instance[2]))/int(instance[2])) * 100
        errors_temp.append(error_temp)
        #print(f"error: {error_temp}")
        avarage_error += error_temp

    all_errors.append(errors_temp)
    all_times.append(times_temp)
    avarage_error /= int(instance[1])
    errors.append(avarage_error)
    print(f"Rozwiązanie problemu {instance[0]} | sciezka: {sol} | dlugosc: {sol_len} | error sredni: {avarage_error} %")

    print(f"len of all times: {len(all_times)}")
times_avaraged = []
for i in range(len(all_times)):
    times_avaraged.append(mean(all_times[i]))



with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    for time, error in zip(all_times, all_errors):
        writer.writerow(time)
        writer.writerow(error)

print(times_avaraged)
print(errors)
g = plt.figure(1, figsize=(9, 6))
plt.scatter(instances[:5], errors[:5])
plt.scatter(instances[5:], errors[5:])
plt.title('Errors')
plt.xlabel("n")
plt.ylabel("error[%]")
g.show()

f = plt.figure(2, figsize=(9, 6))
plt.scatter(instances[:5], times_avaraged[:5])
plt.scatter(instances[5:], times_avaraged[5:])
plt.title('Time')
plt.xlabel("n")
plt.ylabel("T[s]")
plt.show()
"""
draw_plot(["outout_CAS_GOODGOOD.csv"],["CAS"])



"""
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
"""
