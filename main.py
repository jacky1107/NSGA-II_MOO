import numpy as np
import matplotlib.pyplot as plt

from NEW_NSGA_II import NSGA

functionName = ["SCH", "FON", "POL", "KUR", "ZDT2", "ZDT6"]

for time in range(50):
    ga = NSGA(
        problem=functionName[-2],
        numIterations=20,
        populationSize=50,
        )

    doubleParents = ga.initPopulation()

    losses = []
    for iterations in range(int(ga.numIterations)):
        #=====main function=====#
        fronts = ga.nonDominatedSorting(doubleParents)
        parentsIndex = ga.selectPopulations(doubleParents, fronts)
        bestParents = ga.findBestParameters(doubleParents, fronts)
        parents, childs = ga.crossOverAndMutation(doubleParents, parentsIndex, bestParents)
        doubleParents = ga.combineByElitismStrategy(parents, childs)

        #=====loss function=====#
        loss = ga.lossFunction(parents)
        losses.append(loss)
        print(f"Time: {time}")
        print(f"loss: {loss}")
        print(f"iterations: {iterations}")
        print(f"Best Parameters: {bestParents}")

    plt.scatter(ga.f(bestParents, 1), ga.f(bestParents, 2))
    # plt.plot(np.arange(len(losses)), losses)
plt.show()