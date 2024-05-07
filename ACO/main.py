import random
from itertools import accumulate
import math
import csv
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from statistics import mean
from tqdm.auto import tqdm

class Aco:
    def calculate(self, vert, dist):
        #random.seed(42)
        best_path = []
        best_cost = -1

        alpha = 1.0
        beta = 2.5
        evaporation = 0.5
        Cnn = self.calc_init_Cnn(vert, dist)
        init_pher = vert / Cnn
        if vert<25:
            iterations = 100
        elif vert < 75:
            iterations = 10
        elif vert < 449:
            iterations = 5
        else: iterations = 2
        pher = 100

        pheromones = [[init_pher] * vert for _ in range(vert)]
        pheromones_to_add = [[0.0] * vert for _ in range(vert)]


        for j in range(iterations):
            #print("Iteracja:", j + 1)
            for m in range(vert):
                start_node = m
                path = [start_node]
                left_vertices = [y for y in range(vert) if y != start_node]

                for x in range(vert - 1):
                    cn = path[-1]
                    sum_factor = sum(
                        (pheromones[cn][y] ** alpha) * ((1.0 / dist[cn][y]) ** beta)
                        for y in left_vertices
                    )

                    probabil = [
                        (pheromones[cn][ir] ** alpha)
                        * ((1.0 / dist[cn][ir]) ** beta)
                        / sum_factor
                        for ir in left_vertices
                    ]

                    partial_sum = list(accumulate(probabil))
                    p = random.uniform(0, 1)

                    k = 0
                    for ix in range(len(partial_sum)):
                        if p < partial_sum[ix]:
                            wybrany = left_vertices[k]
                            path.append(wybrany)
                            left_vertices.pop(k)
                            break
                        k += 1

                path.append(start_node)
                L = self.calc_cost(path, dist)
                ph = pher/L

                if L < best_cost or best_cost == -1:
                    best_cost = L
                    best_path = path


                for r in range(len(path) - 1):
                    pheromones_to_add[path[r]][path[r + 1]] += ph

            for i in range(vert):
                for k in range(vert):
                    pheromones[i][k] = (
                        pheromones[i][k] * evaporation + pheromones_to_add[i][k]
                    )
                    pheromones_to_add[i][k] = 0.0



        best_path.pop()
        itt = best_path.index(0)
        best_path = best_path[itt:] + best_path[:itt] + [0]
        solution_and_cost = best_path + [best_cost]

        return solution_and_cost

    def calc_cost(self, path, dist):
        cost = 0
        for i in range(len(path) - 1):
            cost += dist[path[i]][path[i + 1]]
        return cost

    def calc_init_Cnn(self, vert, dist):
        path = list(range(vert))
        path.append(0)
        sum_cost = 0
        N = 1000

        for _ in range(N):
            random.shuffle(path[1:-1])
            sum_cost += self.calc_cost(path, dist)

        avg_cost = sum_cost / N
        return avg_cost





    def calculate_DAS(self, vert, dist):
        #random.seed(42)
        best_path = []
        best_cost = -1

        alpha = 1.0
        beta = 5.0
        evaporation = 0.5
        Cnn = self.calc_init_Cnn(vert, dist)
        init_pher = vert / Cnn

        if vert < 25:
            iterations = 100
        elif vert < 75:
            iterations = 10
        elif vert < 449:
            iterations = 5
        else:
            iterations = 2

        pheromones = [[init_pher] * vert for _ in range(vert)]
        pher = 100

        for _ in range(iterations):

            for m in range(vert):
                start_node = m
                path = [start_node]
                left_vertices = [y for y in range(vert) if y != start_node]

                for x in range(vert - 1):
                    cn = path[-1]
                    sum_factor = sum(
                        (pheromones[cn][y] ** alpha) * ((1.0 / dist[cn][y]) ** beta)
                        for y in left_vertices
                    )

                    probabil = [
                        (pheromones[cn][ir] ** alpha)
                        * ((1.0 / dist[cn][ir]) ** beta)
                        / sum_factor
                        for ir in left_vertices
                    ]

                    partial_sum = list(accumulate(probabil))
                    p = random.uniform(0, 1)

                    k = 0
                    for ix in range(len(partial_sum)):
                        if p < partial_sum[ix]:
                            wybrany = left_vertices[k]
                            path.append(wybrany)
                            left_vertices.pop(k)
                            break
                        k += 1

                    # Update pheromones here, using a constant value 'pher'

                    pheromones[path[-2]][path[-1]] += pher
                path.append(start_node)
                L = self.calc_cost(path, dist)

                if L < best_cost or best_cost == -1:
                    best_cost = L
                    best_path = path

            # Evaporation and reset pheromones_to_add
            for i in range(vert):
                for k in range(vert):
                    pheromones[i][k] = pheromones[i][k] * evaporation



        best_path.pop()
        itt = best_path.index(0)
        best_path = best_path[itt:] + best_path[:itt] + [0]
        solution_and_cost = best_path + [best_cost]

        return solution_and_cost
##################################














#########################################################################################





graph = [[float('inf'), 30, 20, 31, 28, 40],
            [30, float('inf'), 10, 14, 20, 44],
            [40, 20, float('inf'), 10, 22, 50],
            [41, 24, 20, float('inf'), 14, 42],
            [38, 30, 32, 24, float('inf'), 28],
            [50, 54, 60, 52, 38, float('inf')]]
instance = Aco()
sol = instance.calculate_DAS( vert= 6, dist=graph)
print(sol)
#142 0-1-2-3-4-5-0

x = input("Podaj plik inicjalizujacy: ")

with open(x, 'r') as f:
    list_of_instances = f.readlines()



for ind, row in enumerate(list_of_instances):
    list_of_instances[ind] = row.split(' ')

all_times = []
errors = []
instances = []
all_errors = []
all_paths = []

for instance in tqdm(list_of_instances):
    matrix = []
    with open(instance[0], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # konwersja typu na int
        for row in spamreader:
            temp_list = []
            for value in row:
                if value != '':
                    temp_list.append(int(value))
            matrix.append(temp_list)

    n = matrix.pop(0)
    instances.append(n[0])
    # ustaw nieskonczonosci na przekatnej
    #print(matrix)
    for i in range(len(matrix)):
        matrix[i][i] = math.inf

    #matrix = np.array(matrix)
    #print(n[0])
    #print(type(n[0]))
    #sol, sol_len = anneling(matrix, n[0])
    times_temp = []
    errors_temp = []
    paths_temp = []
    avarage_error = 0
    aco_calculator = Aco()
    for i in range(int(instance[1])):
        timer_start = timer()

        sol_temp = aco_calculator.calculate(n[0], matrix)
        sol, sol_len = sol_temp[:-1], sol_temp[-1]
        timer_end = timer()
        times_temp.append(timer_end-timer_start)
        error_temp = ((sol_len-int(instance[2]))/int(instance[2])) * 100
        paths_temp.append(sol)
        errors_temp.append(error_temp)
        #print(f"error: {error_temp}")
        avarage_error += error_temp

    all_errors.append(errors_temp)
    all_times.append(times_temp)
    all_paths.append(paths_temp)
    avarage_error /= int(instance[1])
    errors.append(avarage_error)
    print(f"Rozwiązanie problemu {instance[0]} | sciezka: {sol} | dlugosc: {sol_len} | error sredni: {avarage_error} %")

    print(f"len of all times: {len(all_times)}")
times_avaraged = []
for i in range(len(all_times)):
    times_avaraged.append(mean(all_times[i]))



with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    for time, error, path in zip(all_times, all_errors, all_paths):
        writer.writerow(time)
        writer.writerow(error)
        writer.writerow(path)

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
