import random

# 设置遗传算法的参数
pop_size = 50  # 种群大小
max_gen = 100  # 最大迭代次数
mutation_rate = 0.01  # 变异概率

# 定义适应度函数
def fitness(w1, w2, w3):
    y = w1 * x1 + w2 * x2 + w3 * x3
    return y

# 初始化种群
def init_population():
    population = []
    for i in range(pop_size):
        w1 = random.uniform(0, 1)
        w2 = random.uniform(0, 1)
        w3 = random.uniform(0, 1)
        population.append((w1, w2, w3))
    return population

# 选择操作
def selection(population):
    fitness_list = [fitness(*individual) for individual in population]
    total_fitness = sum(fitness_list)
    selection_probs = [fitness / total_fitness for fitness in fitness_list]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

# 交叉操作
def crossover(parents):
    w1_1, w2_1, w3_1 = parents[0]
    w1_2, w2_2, w3_2 = parents[1]
    w1_new = (w1_1 + w1_2) / 2
    w2_new = (w2_1 + w2_2) / 2
    w3_new = (w3_1 + w3_2) / 2
    child = (w1_new, w2_new, w3_new)
    return child

# 变异操作
def mutation(child):
    w1, w2, w3 = child
    if random.random() < mutation_rate:
        w1 = random.uniform(0, 1)
    if random.random() < mutation_rate:
        w2 = random.uniform(0, 1)
    if random.random() < mutation_rate:
        w3 = random.uniform(0, 1)
    mutated_child = (w1, w2, w3)
    return mutated_child

# 进化操作
def evolution(population):
    for i in range(max_gen):
        new_population = []
        for j in range(pop_size // 2):
            parents = selection(population)
            child = crossover(parents)
            mutated_child = mutation(child)
            new_population.append(child)
            new_population.append(mutated_child)
        best_individual = max(population, key=lambda x: fitness(*x))
        new_population.append(best_individual)
        population = new_population
    return population

# 执行遗传算法
def genetic_algorithm(sim1,sim2,sim3):
    global x1,x2,x3
    x1,x2,x3=sim1,sim2,sim3
    if x1==0 and x2==0 and x3==0:
        return 0,0,0
    population = init_population()
    population = evolution(population)

    # 找到适应度最好的个体，并输出结果
    best_individual = max(population, key=lambda x: fitness(*x))
    best_w1, best_w2, best_w3 = best_individual
    best_y = fitness(best_w1, best_w2, best_w3)
    # print("Best solution: w1={}, w2={}, w3={}, y={}".format(best_w1, best_w2, best_w3, best_y))
    total_w=best_w1+best_w2+best_w3
    best_w1=best_w1/total_w
    best_w2=best_w2/total_w
    best_w3=best_w3/total_w
    return best_w1,best_w2,best_w3
