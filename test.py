import copy
import numpy as np

class NSGA():
    def __init__(
        self,
        problem="SCH",
        numIterations=5,
        crossOverSize=5,
        populationSize=10,
        geneParameters=30,
        multiObjects=2,
        mutationRate=0.3,
        ):
        self.problem = problem
        self.bound = self.initBound()
        self.numIterations = numIterations
        self.crossOverSize = crossOverSize
        self.populationSize = populationSize
        self.geneParameters = geneParameters
        self.multiObjects = multiObjects
        self.mutationRate = mutationRate

    def initBound(self):
        if self.problem == "SCH": return [-1e+3, 1e+3]
        elif self.problem == "FON": return [-4, 4]

    def initPopulation(self):
        return np.random.uniform(self.bound[0], self.bound[1], (self.populationSize * 2, self.geneParameters))

    def f1(self, x):
        if self.problem == "SCH": return x ** 2
        elif self.problem == "FON":
            a = (x - 1 / np.sqrt(3)) ** 2
            b = np.sum(a, axis=1)
            return (1 - np.exp(-b)).reshape(-1)

    def f2(self, x):
        if self.problem == "SCH": return (x - 2) ** 2
        elif self.problem == "FON":
            if x.shape[0] == self.multiObjects:
                x = x.reshape(1, 3)
            a = (x + 1 / np.sqrt(3)) ** 2
            b = np.sum(a, axis=1)
            return (1 - np.exp(-b)).reshape(-1)

    def nonDominatedSorting(self, doubleParents):
        """ minimum case """
        print(doubleParents)
        evaluation = np.array([self.f1(doubleParents), self.f2(doubleParents)])
        print(evaluation)
        npArray = np.zeros((len(doubleParents)))
        for i in range(len(doubleParents)):
            for j in range(len(doubleParents)):
                if i == j: continue
                diff = evaluation[0,i] - evaluation[1,j]
                calcRank = len(diff[diff > 0])
                if calcRank > 1: npArray[i] += 1
        doubleParents, rankList = self.sortingByRank(doubleParents, npArray)
        return doubleParents, rankList

    def sortingByRank(self, doubleParents, npArray):
        doubleParents = doubleParents[np.argsort(npArray)]
        npArray = np.sort(npArray)
        rankList = []
        isNotCheckAll = True
        while isNotCheckAll:
            rank = []
            for i in range(len(doubleParents)):
                if npArray[i] == 0:
                    print(self.f1(doubleParents[i]))
                    rank.append([self.f1(doubleParents[i]), self.f2(doubleParents[i]), doubleParents[i]])
                elif npArray[i] > 0: break
                elif npArray[i] < 0 and i == (len(doubleParents) - 1): isNotCheckAll = False
            if len(rank) > 0:
                rank = sorted(rank)
                rankList.append(rank)
            npArray -= 1
        return doubleParents, rankList
    
    def calcCrowdingDistance(self, rankArray):
        distance = []
        for i in range(len(rankArray)):
            dist = [np.inf]
            for j in range(1, len(rankArray[i]) - 1):
                normalizeDist = 0
                for m in range(self.multiObjects):
                    fmin = min(rankArray[i][:, m])
                    fmax = max(rankArray[i][:, m])
                    diff = abs(rankArray[i][j+1, m] - rankArray[i][j-1, m])
                    if fmax == fmin: print("Error")
                    diff = diff / (fmax - fmin)
                    normalizeDist += diff
                dist.append(normalizeDist)
            if len(rankArray[i]) > 1: dist.append(np.inf)
            distance.append(dist)
        return distance
    
    def convertListToArray(self, lists):
        array = []
        for i in range(len(lists)):
            temp = np.zeros((len(lists[i]), self.multiObjects + 1))
            for j in range(len(lists[i])):
                for m in range(len(lists[i][j])):
                    temp[j, m] = float(lists[i][j][m])
            array.append(temp)
        return array

    def selectPopulations(self, rankArray, distance):
        index = 0
        parents = np.zeros((self.populationSize, self.geneParameters))
        for i in range(len(rankArray)):
            argsort = np.argsort(distance[i])
            rank = copy.deepcopy(rankArray[i])
            rank = rank[argsort][::-1]
            for j in range(len(rank)):
                parents[index] = rank[j, -1]
                index += 1
                if index >= self.populationSize: return parents

    def crossOverAndMutation(self, parents):
        if self.crossOverSize > self.populationSize: raise Exception("Cross over size need to lower than gene dimension")
        childs = np.zeros_like(parents)
        for i in range(self.populationSize):
            child = self.crossOver(parents)
            child = self.mutation(parents, child)
            childs[i] = child
        return childs

    def mutation(self, parents, child):
        if np.random.random() > self.mutationRate:
            rand = np.random.uniform(min(parents), max(parents), 1)
            child = rand
        return child

    def crossOver(self, parents):
        weights = self.generateWeights(parents)
        parents = np.sort(parents)
        bestParents = parents[:self.crossOverSize]
        total = 0
        for i in range(self.crossOverSize):
            total += bestParents[i] * weights[i]
        return total

    def generateWeights(self, parents):
        weights = np.array([np.random.random() for i in range(len(parents))])
        weights = weights / np.sum(weights)
        return weights

    def combineByElitismStrategy(self, parents, childs):
        return np.concatenate((parents, childs), axis=0)
    
    def lossFunction(self, doubleParents):
        return np.sum(abs(doubleParents))
