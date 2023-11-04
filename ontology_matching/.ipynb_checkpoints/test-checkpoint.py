import numpy as np
import random
import matplotlib.pyplot as plt

# 定义适应度函数
def fitness(w):
    y = w[0]*x1 + w[1]*x2 + w[2]*x3
    return y

# 初始化种群
def init_population(pop_size, w_size):
    population = []
    for i in range(pop_size):
        w = [random.uniform(-1, 1) for j in range(w_size)]
        population.append(w)
    return population

# 选择
def selection(population, fitness):
    total_fitness = np.sum(fitness)
    if total_fitness == 0:
        n = len(population) // 2
        sorted_population = sorted(population, key=lambda x: fitness(x), reverse=True)
        selected = sorted_population[:n]
    else:
        p = fitness / total_fitness
        cum_p = np.cumsum(p)
        selected = []
        for i in range(len(population)):
            r = random.random()
            for j in range(len(cum_p)):
                if r < cum_p[j]:
                    selected.append(population[j])
                    break
    return selected


# 交叉
def crossover(parents, offspring_size):
    offspring = []
    while len(offspring) < offspring_size:
        parent1 = random.randint(0, len(parents)-1)
        parent2 = random.randint(0, len(parents)-1)
        if parent1 != parent2:
            cross_point = random.randint(0, len(parents[0])-1)
            offspring1 = parents[parent1][:cross_point] + parents[parent2][cross_point:]
            offspring2 = parents[parent2][:cross_point] + parents[parent1][cross_point:]
            offspring.append(offspring1)
            offspring.append(offspring2)
    return offspring

# 变异
def mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            r = random.random()
            if r < mutation_rate:
                offspring[i][j] = offspring[i][j] + random.uniform(-1, 1)
    return offspring

# 运行遗传算法
def genetic_algorithm(pop_size, w_size, max_iter, mutation_rate):
    global x1, x2, x3
    # 生成样本数据
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    x3 = np.random.normal(0, 1, 100)
    y = x1 + 2*x2 + 3*x3
    # 初始化种群
    population = init_population(pop_size, w_size)
    best_fitness = []
    best_w = None
    # 迭代
    for i in range(max_iter):
        # 计算适应度
        fitness_values = np.array([fitness(w) for w in population])
        # 选择
        parents = selection(population, fitness_values)
        # 交叉
        offspring = crossover(parents, pop_size-len(parents))
        # 变异
        offspring = mutation(offspring, mutation_rate)
        # 合并父代和
        population = parents + offspring
        # 计算适应度
        fitness_values = np.array([fitness(w) for w in population])
        # 找到最优解
        best_index = np.argmax(fitness_values)
        best_fitness.append(fitness_values[best_index])
        best_w = population[best_index]
        # 输出结果
        print("iter:", i, "best fitness:", best_fitness[-1], "best w:", best_w)
        # 绘制结果图像
        plt.plot(best_fitness)
        plt.title("Fitness Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
        return best_w

best_w = genetic_algorithm(pop_size=100, w_size=3, max_iter=100, mutation_rate=0.1)
print("best w:", best_w)