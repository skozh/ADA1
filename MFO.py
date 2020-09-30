Original Code: https://github.com/IbraDje/MFO-Python/blob/master/MFO.ipynb

import numpy as np

def MFO(nsa, dim, ub, lb, max_iter, fobj):
  ''' Main function
  Parameters :
  - nsa : Number of Search Agents
  - dim : Dimension of Search Space
  - ub : Upper Bound
  - lb : Lower Bound
  - max_iter : Number of Iterations
  - fobj : Objective Function (Fitness Function)
  Returns :
  - bFlameScore : Best Flame Score
  - bFlamePos : Best Flame Position
  - ConvergenceCurve : Evolution of the best Flame Score on every iteration
  '''

  # Initialize the positions of moths
  mothPos = np.random.uniform(low=lb, high=ub, size=(nsa, dim))

  convergenceCurve = np.zeros(shape=(max_iter))

  # print("Optimizing  \"" + fobj.__name__ + "\"")

  for iteration in range(max_iter):  # Main loop
    # Number of flames Eq. (3.14) in the paper
    flameNo = int(np.ceil(nsa-(iteration+1)*((nsa-1)/max_iter)))

    # Check if moths go out of the search space and bring them back
    mothPos = np.clip(mothPos, lb, ub)

    # Calculate the fitness of moths
    mothFit = fobj(mothPos)

    if iteration == 0:
      # Sort the first population of moths
      order = mothFit.argsort()
      mothFit = mothFit[order]
      mothPos = mothPos[order, :]

      # Update the flames
      bFlames = np.copy(mothPos)
      bFlamesFit = np.copy(mothFit)

    else:
      # Sort the moths
      doublePop = np.vstack((bFlames, mothPos))
      doubleFit = np.hstack((bFlamesFit, mothFit))

      order = doubleFit.argsort()
      doubleFit = doubleFit[order]
      doublePop = doublePop[order, :]

      # Update the flames
      bFlames = doublePop[:nsa, :]
      bFlamesFit = doubleFit[:nsa]

    # Update the position best flame obtained so far
    bFlameScore = bFlamesFit[0]
    bFlamesPos = bFlames[0, :]

    # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a = -1 + (iteration+1) * ((-1)/max_iter)

    # D in Eq. (3.13)
    distanceToFlames = np.abs(bFlames - mothPos)

    b = 1
    t = (a-1)*np.random.rand(nsa, dim) + 1
    ''' Update the position of the moth with respect to its corresponding
    flame if the moth position is less than the number of flames
    calculated, otherwise update the position of the moth with respect
    to the last flame '''
    temp1 = bFlames[:flameNo, :]
    temp2 = bFlames[flameNo-1, :]*np.ones(shape=(nsa-flameNo, dim))
    temp2 = np.vstack((temp1, temp2))
    mothPos = distanceToFlames*np.exp(b*t)*np.cos(t*2*np.pi) + temp2

    convergenceCurve[iteration] = bFlameScore

  return bFlameScore, bFlamesPos, convergenceCurve
