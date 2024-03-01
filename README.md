# CBOSS - Combinatorial Bayesian Optimization for Structure Selection

As part of the submission at the Transactions in Machine Learning Research (TMLR) journal:

[Discovering Model Structure of Dynamical Systems with Combinatorial Bayesian Optimization](https://openreview.net/forum?id=2iOOvQmJBK)

This repository contains the code for the CBOSS optimizer and the experiments presented in the paper.
The main optimizer function can be found at [optimizers/CBOSS.py](optimizers/CBOSS.py).


## Optimization Problem

**CBOSS** can be used to for the following problem setup:

$$
\begin{align}
    \boldsymbol x^* = \arg\min_{\boldsymbol x\in\mathcal X} & \quad f (\boldsymbol x) 
    \\
    s.t. & \quad g_j(\boldsymbol x) \leq 0 \quad  \forall j \in \{1, \dots, M\} \\
         & \quad h(\boldsymbol x) = 1
\end{align}
$$

where 
- $x$ is a vector of categorical decision variables denoted as $\boldsymbol x \in \mathcal X$, defined over the combinatorial domain $\mathcal X = \mathcal X_1 \times \mathcal X_2 \times\dots \times \mathcal X_d$ with $d$ categorical decision variables with respective cardinalities $k_1, \dots, k_d$.
- $f: \mathcal X \to \mathbb R$ is the objective function.
- $g_j: \mathcal X \to \mathbb R$ are the inequality constraints.
- $h: \mathcal X \to \{0,1\}$ are the binary equality constraints.


## Acquisition function

It implements surrogate models designed for combinatorial spaces and the acquisition function `FRCHEI` that can handle inequality and crash constraints.

![notebooks/images/FRCHEI.png](notebooks/images/FRCHEI.png)

*(generated with `notebooks/FRCHEI.ipynb`)


Note that although we consider only categorical variables, this optimizer can be adapted for continuous or mixed inputs spaces.

## Installation

To install the package, activate your python environment and run:

```sh
make build
make install
```


## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more details.