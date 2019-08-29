# NoisyOptimisationWithStochasticAnnealing
Noisy Optimization with Optimal Stochastic Annealing.

This is an implementation and proof of concept of the algorithm "Optimal Sampling for Simulated Annealing Under Noise".
This algorithm was developed by Juergen Branke and Robin C. Ball, and compared against other commercial algorithms by me.
The comparison was performed in the scope of the independent Final report for the MSc in Mathematics for Real-World Systems, Centre for Complexity Science, University of Warwick.

The published paper can be found in https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0774?journalCode=ijoc .

This algorithm is a simulated annealing variant for optimization problems in which the solution quality can only be estimated by sampling from a random distribution. The aim is to find the solution with the best expected performance, as, e.g., is typical for problems where solutions are evaluated using a stochastic simulation. 
Assuming Gaussian noise with known standard deviation, we derive a fully sequential sampling procedure and decision rule. The procedure starts with a single sample of the value of a proposed move to a neighboring solution and then continues to draw more samples until it is able to make a decision to accept or reject the move. Under constraints of equilibrium detailed balance at each draw, we find a decoupling between the acceptance criterion and the choice of the rejection criterion. 
We derive a universally optimal acceptance criterion in the sense of maximizing the acceptance probability per sample and thus the efficiency of the optimization process. We show that the choice of the move rejection criterion depends on expectations of possible alternative moves and propose a simple and practical (albeit more empirical) solution that preserves detailed balance. An empirical evaluation shows that the resulting approach is indeed more efficient than several previously proposed simulated annealing variants.
