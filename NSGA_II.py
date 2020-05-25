import copy
import numpy as np

class NSGA():
    def __init__(
        self,
        problem="SCH",
        numIterations=1e+2,
        crossOverSize=5,
        populationSize=10,
        multiObjects=2,
        mutationRate=0.8,
        ):
        self.problem = problem
        self.bound = self.initBound()
        self.numIterations = numIterations
        self.crossOverSize = crossOverSize
        self.populationSize = populationSize
        self.geneParameters = self.initParameters()
        self.multiObjects = multiObjects
        self.mutationRate = mutationRate

    def initParameters(self):
        if self.problem == "SCH": return 1
        elif self.problem == "FON": return 3
        elif self.problem == "ZDT2": return 30

    def initBound(self):
        if self.problem == "SCH": return [-1e+3, 1e+3]
        elif self.problem == "FON": return [-4, 4]
        elif self.problem == "ZDT2": return [0, 1]

    def initPopulation(self):
        return np.random.uniform(self.bound[0], self.bound[1], (self.populationSize * 2, self.geneParameters))

    def f(self, x, objective):
        if objective == 1:
            if self.problem == "SCH": return x ** 2
            elif self.problem == "FON": return (1 - np.exp(-np.sum((x - (1 / np.sqrt(3))) ** 2)))
            elif self.problem == "ZDT2": return x[0]
        elif objective == 2:
            if self.problem == "SCH": return (x - 2) ** 2
            elif self.problem == "FON": return (1 - np.exp(-np.sum((x + (1 / np.sqrt(3))) ** 2)))
            elif self.problem == "ZDT2":
                s = np.sum(x[1:]) / (len(x) - 1)
                gx = 1 + 9 * s
                return gx * (1 - (x[0]/gx) ** 2)

    def nonDominatedSorting(self, doubleParents):
        F = []
        fronts = []
        table = []
        for i, p in enumerate(doubleParents):
            info = [i]
            sp = []
            n = 0
            for j, q in enumerate(doubleParents):
                if i == j: continue
                if self.compare(p, q) == 2: sp.append(j)
                elif self.compare(p, q) == 0: n += 1
            if n == 0: F.append(i)
            info.append(n)
            info.append(sp)
            table.append(info)
        while len(F) != 0:
            fronts.append(F)
            Q = []
            for i in F:
                sp = table[i][2]
                for j in sp:
                    table[j][1] = table[j][1] - 1
                    if table[j][1] == 0: Q.append(j)
            F = Q
        return fronts

    def compare(self, p, q):
        """
        Compare method
            if p is smaller than q:
                p is dominate q
                len(evaluate) == 2
        """
        evaluateP = np.array([self.f(p, 1), self.f(p, 2)])
        evaluateQ = np.array([self.f(q, 1), self.f(q, 2)])
        evaluate  = evaluateP - evaluateQ
        dominate = evaluate[evaluate <= 0]
        return len(dominate)

    def calcCrowdingDistance(self, doubleParents, fronts):
        dists = []
        for front in fronts:
            total = 0
            for objective in range(1, self.multiObjects+1):
                sortedFront = self.sorting(doubleParents, front, objective)
                dist = self.calcDistanceBySortedFront(doubleParents, front, sortedFront, objective)
                temp = np.array(dist)
                total += temp
            dists.append(list(total))
        return dists

    def sorting(self, doubleParents, front, objective):
        m = [self.f(doubleParents[dataIndex], objective) for dataIndex in front]
        sortedIndex = np.argsort(m)
        if self.problem == "SCH": sortedIndex = sortedIndex[0]
        new = [front[index] for index in sortedIndex]
        return new

    def calcDistanceBySortedFront(self, doubleParents, front, sortedFront, objective):
        dists = np.zeros(len(front))
        left = self.findIndex(front, sortedFront[0])
        right = self.findIndex(front, sortedFront[-1])
        dists[left] = np.inf
        dists[right] = np.inf
        fmin = self.calcDistance(doubleParents, sortedFront[0], objective)
        fmax = self.calcDistance(doubleParents, sortedFront[-1], objective)
        for i in range(1, len(sortedFront) - 1):
            if (fmax - fmin) == 0:
                fmax += 1e-5
                fmin -= 1e-5
            da = self.calcDistance(doubleParents, sortedFront[i+1], objective)
            db = self.calcDistance(doubleParents, sortedFront[i-1], objective)
            dist = (da - db) / (fmax - fmin)
            index = self.findIndex(front, sortedFront[i])
            dists[index] = dist
        return dists

    def calcDistance(self, doubleParents, index, objective):
        data = doubleParents[index]
        return self.f(data, objective)

    def findIndex(self, front, index):
        for i in range(len(front)):
            if front[i] == index: return i

    def selectPopulations(self, doubleParents, fronts, distance):
        parents = np.zeros((self.populationSize, self.geneParameters))
        add = 0
        selectedNotComplete = True
        while selectedNotComplete:
            for i, front in enumerate(fronts):
                sortedIndex = np.argsort(distance[i])
                sortedFront = [front[index] for index in sortedIndex]
                for index in sortedFront:
                    parents[add] = doubleParents[index]
                    add += 1
                    if add >= self.populationSize:
                        selectedNotComplete = False
                        break
                if selectedNotComplete == False: break
        return parents

    def crossOverAndMutation(self, parents):
        if self.crossOverSize > self.populationSize: raise Exception("Cross over size need to lower than gene dimension")
        childs = np.zeros_like(parents)
        for i in range(self.populationSize):
            child = self.crossOver(parents)
            child = self.mutation(parents, child)
            childs[i] = copy.deepcopy(child)
        return childs

    def mutation(self, parents, child):
        return child if np.random.random() > self.mutationRate else np.random.uniform(self.bound[0], self.bound[1], self.geneParameters)

    def crossOver(self, parents):
        weights = self.generateWeights(self.populationSize)
        sortedWeights = np.argsort(weights)
        newParents = np.array([parents[index] for index in sortedWeights])[::-1]
        bestParents = newParents[:self.crossOverSize]
        total = 0
        weights = self.generateWeights(self.crossOverSize)
        for i in range(self.crossOverSize):
            total += bestParents[i] * weights[i]
        return total

    def generateWeights(self, length):
        weights = np.array([np.random.random() for i in range(length)])
        weights = weights / np.sum(weights)
        return weights

    def combineByElitismStrategy(self, parents, childs):
        return np.concatenate((parents, childs), axis=0)
    
    def lossFunction(self, parents):
        loss = 0
        for i in range(len(parents)):
            loss += self.f(parents[i], 1) * self.f(parents[i], 2)
        return loss

    def bestSolution(self, parents):
        best = np.inf
        p = None
        for parent in parents:
            sol = self.f(parent, 1) + self.f(parent, 2)
            if sol < best:
                best = sol
                p = parent
        return best, p