import copy
import numpy as np

class NSGA():
    def __init__(
        self,
        problem="SCH",
        numIterations=1e+2,
        crossOverSize=2,
        populationSize=10,
        multiObjects=2,
        mutationRate=1,
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
        elif self.problem == "POL": return 2
        elif self.problem == "FON": return 3
        elif self.problem == "KUR": return 3
        elif self.problem == "ZDT2": return 30
        elif self.problem == "ZDT6": return 10

    def initBound(self):
        if self.problem == "SCH": return [-1e+3, 1e+3]
        elif self.problem == "POL": return [-np.pi, np.pi]
        elif self.problem == "FON": return [-4, 4]
        elif self.problem == "KUR": return [-5, 5]
        elif self.problem == "ZDT2": return [0, 1]
        elif self.problem == "ZDT6": return [0, 1]

    def initPopulation(self):
        return np.random.uniform(self.bound[0], self.bound[1], (self.populationSize * 2, self.geneParameters))

    def f(self, x, objective):
        if objective == 1:
            if self.problem == "SCH": return x ** 2
            elif self.problem == "POL":
                A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + 1 * np.sin(2) - 1.5 * np.cos(2)
                A2 = 1.5 * np.sin(1) - 1 * np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
                B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + 1 * np.sin(x[1]) - 1.5 * np.cos(x[1])
                B2 = 1.5 * np.sin(x[0]) - 1 * np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
                return 1 + (A1 - B1) ** 2 + (A2 - B2) ** 2
            elif self.problem == "FON": return (1 - np.exp(-np.sum((x - (1 / np.sqrt(3))) ** 2)))
            elif self.problem == "KUR": return np.sum(-10 * (np.exp(-0.2 * ((x[:self.geneParameters-1] ** 2 + x[1:] ** 2) ** 0.5))))
            elif self.problem == "ZDT2": return x[0]
            elif self.problem == "ZDT6": return 1 - np.exp( -4*x[0] * (np.sin(6 * np.pi * x[0]) ** 6) )

        elif objective == 2:
            if self.problem == "SCH": return (x - 2) ** 2
            elif self.problem == "POL": return (x[0] + 3) ** 2 + (x[1] + 1) ** 2
            elif self.problem == "FON": return (1 - np.exp(-np.sum((x + (1 / np.sqrt(3))) ** 2)))
            elif self.problem == "KUR": return np.sum((abs(x)**0.8) + 5 * np.sin((x**3)))
            elif self.problem == "ZDT2":
                gx = 1 + 9 * np.sum(x[1:]) / (self.geneParameters - 1)
                return gx * (1 - (x[0]/gx) ** 2)
            elif self.problem == "ZDT6":
                gx = 1 + 9 * ( np.sum(x[1:]) / (self.geneParameters - 1) ) ** 0.25
                return gx * (1 - (self.f(x, 1)/gx) ** 2 )

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

    def calcDistance(self, doubleParentsIndex, objective):
        return self.f(doubleParentsIndex, objective)

    def findIndex(self, front, index):
        for i in range(len(front)):
            if front[i] == index: return i

    def sorting(self, doubleParents, front, objective):
        m = np.array([self.f(parents, objective) for parents in doubleParents[front]])
        m = m.reshape(-1)
        sortedIndex = np.argsort(m)
        new = copy.deepcopy(front)
        new = np.array(new)
        new = new[sortedIndex]
        return new

    def calcDistanceBySortedFront(self, doubleParents, front, sortedFront, objective):
        dists = np.zeros(len(front))
        left = self.findIndex(front, sortedFront[0])
        right = self.findIndex(front, sortedFront[-1])
        dists[left] = np.inf
        dists[right] = np.inf
        fmin = self.calcDistance(doubleParents[sortedFront[0 ]], objective)
        fmax = self.calcDistance(doubleParents[sortedFront[-1]], objective)
        for i in range(1, len(sortedFront) - 1):
            if (fmax - fmin) == 0:
                fmax += 1e-5
                fmin -= 1e-5
            da = self.calcDistance(doubleParents[sortedFront[i+1]], objective)
            db = self.calcDistance(doubleParents[sortedFront[i-1]], objective)
            dist = (da - db) / (fmax - fmin)
            index = self.findIndex(front, sortedFront[i])
            dists[index] = dist
        return dists

    def calcCrowdingDistance(self, doubleParents, front):
        total = 0
        for objective in range(1, self.multiObjects+1):
            sortedFront = self.sorting(doubleParents, front, objective)
            distList = self.calcDistanceBySortedFront(doubleParents, front, sortedFront, objective)
            temp = copy.deepcopy(distList)
            distArray = np.array(temp)
            total += distArray
        return total

    def selectPopulations(self, doubleParents, fronts):
        best = []
        parents = []
        for i, front in enumerate(fronts):
            if (len(parents) + len(front)) > self.populationSize:
                dist = self.calcCrowdingDistance(doubleParents, front)
                sortedIndex = np.argsort(dist)
                temp = copy.deepcopy(front)
                temp = np.array(temp)
                temp = list(temp[sortedIndex])
                temp = temp[::-1]
                size = self.populationSize - len(parents)
                for index in temp[:size]:
                    parents.append(index)
                break
            else:
                for index in front:
                    parents.append(index)
        return parents

    def crossOverAndMutation(self, doubleParents, parents, bestParents):
        if self.crossOverSize > self.populationSize: raise Exception("Cross over size need to lower than gene dimension")
        newParents = copy.deepcopy(doubleParents[parents])
        childs = np.zeros((self.populationSize, self.geneParameters))
        for i in range(self.populationSize):
            child = self.crossOver(doubleParents, parents[i], bestParents)
            child = self.mutation(child)
            child = np.clip(child, self.bound[0], self.bound[1])
            childs[i] = copy.deepcopy(child)
        return newParents, childs

    def mutation(self, child):
        return child + np.random.normal(0, 0.01, self.geneParameters)

    def crossOver(self, doubleParents, parents, bestParents):
        # indexA = np.random.choice(len(parents))
        # indexB = np.random.choice(len(parents))
        # a = doubleParents[parents[indexA]]
        # b = doubleParents[parents[indexB]]
        p = np.random.random()
        return doubleParents[parents] * p + bestParents * (1 - p)
        # return (doubleParents[parents] + bestParents) / 2

    def combineByElitismStrategy(self, parents, childs):
        return np.concatenate((parents, childs), axis=0)
    
    def lossFunction(self, parents):
        loss = 0
        best = np.inf
        bestParents = None
        for i in range(len(parents)):
            y = (self.f(parents[i], 1) + self.f(parents[i], 2)) / 2
            if y < best:
                best = y
                bestParents = parents[i]
            loss += float(y)
        return loss / len(parents)

    def findBestParameters(self, doubleParents, fronts):
        # best = np.inf
        # bestP = None
        # for index in fronts[0]:
        #     y = (abs(self.f(doubleParents[index], 1)) + abs(self.f(doubleParents[index], 2))) / 2
        #     if y <= best:
        #         best = y
        #         bestP = doubleParents[index]
        # return bestP
        index = self.calcCrowdingDistance(doubleParents, fronts[0])
        sortedIndex = np.argsort(index)
        temp = copy.deepcopy(fronts[0])
        temp = np.array(temp)
        temp = temp[sortedIndex]
        return doubleParents[temp[0]]