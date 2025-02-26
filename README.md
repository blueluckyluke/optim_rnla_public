# optim_rnla
Optimisation environment to test and implement a new class of optimisation algorithms. It centres around reformulating the trust-region subproblem. This is a not necessarily convex, constrained quadratic optimisation problem which needs to be solved many times for the trust region algorithm. Here, we reformulate the trust-region subproblem as an eigenvalue problem and solve it via novel fast randomised numerical linear algebra methods.
The jupyter notebook examplecode.ipynb contains simple code snippets executing the main parts of the code and is a good place to start.
