import numpy as np
import matplotlib.pyplot as plt

from NSGA_II import NSGA

evaluation = False
if evaluation: testTime = 100
else: testTime = 1

average = []
for _ in range(testTime):

    show = False
    ga = NSGA(
        problem="ZDT2",
        numIterations=200,
        crossOverSize=5,
        populationSize=20,
        )

    doubleParents = ga.initPopulation()

    losses = []
    for iterations in range(int(ga.numIterations)):
        #=====main function=====#
        fronts = ga.nonDominatedSorting(doubleParents)
        distance = ga.calcCrowdingDistance(doubleParents, fronts)
        parents = ga.selectPopulations(doubleParents, fronts, distance)
        childs = ga.crossOverAndMutation(parents)
        doubleParents = ga.combineByElitismStrategy(parents, childs)
        best, parent = ga.bestSolution(parents)

        #=====loss function=====#
        loss = ga.lossFunction(parents)
        losses.append(loss)
        print(f"loss: {loss}")
        print(f"iterations: {iterations}")
        print(f"Best Parameters: {parent}")

        if show:
            bound = ga.bound
            x = np.linspace(bound[0], bound[1], 1000)
            y1 = ga.f1(x)
            y2 = ga.f2(x)
            y = y1 + y2
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.plot(x, y)
            if 'sca' in globals(): sca.remove()
            sca = plt.scatter(parents, ga.f1(parents), s=40, lw=0, c='b', alpha=0.5)
            if 'sca' in globals(): sca.remove()
            sca = plt.scatter(parents, ga.f2(parents), s=40, lw=0, c='b', alpha=0.5)
            plt.pause(0.05)

    plt.plot(np.arange(len(losses)), losses)
    plt.show()

if evaluation:
    s = 0
    count = 0
    failed = 0
    for i in range(len(average)):
        if average[i] != (ga.numIterations - 1):
            s += average[i]
            count += 1
        else:
            failed += 1
    print(f"Average iteration: {s / count}")
    print(f"Success: {len(average) - failed}")
    print(f"Failed: {failed}")