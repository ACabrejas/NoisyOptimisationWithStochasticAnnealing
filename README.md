# NoisyOptimisationWithStochasticAnnealing
Noisy Optimization with Optimal Stochastic Annealing.

This is an implementation and proof of concept of the algorithm "Optimal Sampling for Simulated Annealing Under Noise".
This algorithm was developed by Juergen Branke and Robin C. Ball.
The published paper can be found in https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0774?journalCode=ijoc .

The comparison presented here was performed in the scope of the independent final report for the MSc in Mathematics for Real-World Systems, MathSys CDT (University of Warwick), in collaboration with Lanner Group, developer of the commercial optimiser for industrial processes WITNESS Horizon (https://www.lanner.com/en-us/technology/witness-simulation-software.html).

The algorithm is a simulated annealing variant for optimization problems in which the solution quality can only be estimated by sampling from a random distribution. The aim is to find the solution with the best expected performance, as, e.g., is typical for problems where solutions are evaluated using a stochastic simulation. 

Assuming Gaussian noise with known standard deviation, a fully sequential sampling procedure and decision rule are introduced. The procedure starts with a single sample of the value of a proposed move to a neighboring solution and then continues to draw more samples until it is able to make a decision to accept or reject the move. Under constraints of equilibrium detailed balance at each draw, there is a decoupling between the acceptance criterion and the choice of the rejection criterion. 

A universally optimal acceptance criterion is presented, in the sense of maximizing the acceptance probability per sample and thus the efficiency of the optimization process. A simple and practical (albeit more empirical) solution that preserves detailed balance is used. This empirical evaluation shows that the resulting approach is indeed more efficient than several previously proposed simulated annealing variants.
