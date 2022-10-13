# Probabilistic Discriminative Models
For the two-class classification problem, we have seen that the posterior probability
of class $C_1$ can be written as a logistic sigmoid acting on a linear function of *x*, for a
wide choice of class-conditional distributions $p(x|C_k)$. Similarly, for the multiclass
case, the posterior probability of class $C_k$ is given by a softmax transformation of a
linear function of *x*. For specific choices of the class-conditional densities $p(x|C_k)$,
we have used maximum likelihood to determine the parameters of the densities as
well as the class priors $p(C_k)$ and then used Bayes’ theorem to find the posterior class probabilities.

However, an alternative approach is to use the functional form of the generalized
linear model explicitly and to determine its parameters directly by using maximum
likelihood. We shall see that there is an efficient algorithm finding such solutions
known as [[iterative reweighted least squares]], or [[IRLS]].

The indirect approach to finding the parameters of a generalized linear model,
by fitting class-conditional densities and class priors separately and then applying Bayes’ theorem, represents an example of [[generative]] modelling, because we could
take such a model and generate synthetic data by drawing values of *x* from the
marginal distribution $p(x)$. In the direct approach, we are maximizing a likelihood
function defined through the conditional distribution $p(C_k|x)$, which represents a
form of [[discriminative]] training. One advantage of the discriminative approach is
that there will typically be fewer adaptive parameters to be determined, as we shall
see shortly. It may also lead to improved predictive performance, particularly when
the class-conditional density assumptions give a poor approximation to the true distributions.

## Index
- [[1. Fixed basis functions]]
- [[2. Logistic regression]]
- [[3. Iterative reweighted least squares]]
- [[4. Multiclass logistic regression]]
- [[5. Probit regression]]
- [[6. Canonical link functions]]