# Network Training
So far, we have viewed neural networks as a general class of parametric nonlinear
functions from a vector x of input variables to a vector y of output variables. A
simple approach to the problem of determining the network parameters is to make an analogy with the discussion of polynomial curve fitting in [[Example - Polynomial Curve Fitting| Section 1.1]], and therefore to minimize a sum-of-squares error function. Given a training set comprising a set of input vectors $\{x_n\}$, where $n = 1, . . . , N$, together with a corresponding set of target vectors $\{t_n\}$, we minimize the error function
$$
E(w) = \frac{1}{2}\sum^N_{n=1}||y(x_n,w) - t_n||^2
\tag{5.11}
$$
However, we can provide a much more general view of network training by first
giving a probabilistic interpretation to the network outputs. We have already seen
many advantages of using probabilistic predictions in [[4. Inference and decision|Section 1.5.4]]. Here it will also
provide us with a clearer motivation both for the choice of output unit nonlinearity
and the choice of error function.

We start by discussing regression problems, and for the moment we consider
a single target variable *t* that can take any real value. Following the discussions
in [[Section 1.2.5]] and [[Linear Basis Function Models|3.1]], we assume that t has a Gaussian distribution with an 
x-dependent mean, which is given by the output of the neural network, so that
$$
p(t|x,w) = N(t|y(x,w),\beta^{-1})
\tag{5.12}
$$
where $\beta$ is the precision (inverse variance) of the Gaussian noise. Of course this
is a somewhat restrictive assumption, and in [[Section 5.6]] we shall see how to extend
this approach to allow for more general conditional distributions. For the conditional distribution given by (5.12), it is sufficient to take the output unit activation function to be the identity, because such a network can approximate any continuous function from x to *y*. Given a data set of *N* independent, identically distributed observations $X = \{x_1, . . . , x_N\}$, along with corresponding target values $t = {t_1, . . . , t_N}$, we can construct the corresponding likelihood function
$$
p(t|X,w,\beta)=\prod^N_{n=1}p(t_n|x_n,w,\beta)
$$
Taking the negative logarithm, we obtain the error function
$$
\frac{\beta}{2}\sum^N_{n=1}\{y(x_n,w)-t_n\}^2-\frac{N}ln(2\pi){2}ln(\beta + \frac{N}{2})
\tag{5.13}
$$
which can be used to learn the parameters w and $\beta$. In [[Section 5.7]], we shall discuss
the Bayesian treatment of neural networks, while here we consider a maximum
likelihood approach. Note that in the neural networks literature, it is usual to consider the minimization of an error function rather than the maximization of the (log) likelihood, and so here we shall follow this convention. Consider first the determination of **w**. Maximizing the likelihood function is equivalent to minimizing the sum-of-squares error function given by
$$
E(w) = \frac{1}{2}\sum^N_{n=1}\{y(x_n,w)-t_n\}^2
\tag{5.14}
$$
where we have discarded additive and multiplicative constants. The value of **w** found by minimizing $E(w)$ will be denoted $w_{ML}$ because it corresponds to the maximum likelihood solution. In practice, the nonlinearity of the network function $y(x_n,w)$ causes the error $E(w)$ to be nonconvex, and so in practice local maxima of the likelihood may be found, corresponding to local minima of the error function, as discussed in [[Section 5.2.1]].

Having found $w_{ML}$, the value of $\beta$ can be found by minimizing the negative log
likelihood to give
$$
\frac{1}{\beta_{ML}}=\frac{1}{N}\sum^N_{n=1}]\{y(x_n,w_{ML})-t_n\}^2
\tag{5.15}
$$
Note that this can be evaluated once the iterative optimization required to find $w_{ML}$ is completed. If we have multiple target variables, and we assume that they are independent conditional on x and w with shared noise precision $\beta$, then the conditional distribution of the target values is given by
$$
p(t|x,w) = n(t|y(x,w),\beta^{-1}I)
\tag{5.16}
$$
Following the same argument as for a single target variable, we see that the maximum likelihood weights are determined by minimizing the sum-of-squares error function (5.11). The noise precision is then given by
$$
\frac{1}{\beta_{ML}} = \frac{1}{NK}\sum^N_{n=1}||y(x_n,w_{ML})-t_n||^2
\tag{5.17}
$$
where *K* is the number of target variables. The assumption of independence can be dropped at the expense of a slightly more complex optimization problem.

Recall from [[Section 4.3.6]] that there is a natural pairing of the error function
(given by the negative log likelihood) and the output unit activation function. In the
regression case, we can view the network as having an output activation function that is the identity, so that $y_k = a_k$. The corresponding sum-of-squares error function has the property
$$
\frac{\partial E}{\partial a_k} = y_k - t_k
\tag{5.18}
$$
which we shall make use of when discussing error backpropagation in [[Section 5.3]].

Now consider the case of binary classification in which we have a single target
variable *t* such that $t = 1$ denotes class $C_1$ $and *t* = 0 denotes class $C_2$. Following
the discussion of canonical link functions in [[Section 4.3.6]], we consider a network
having a single output whose activation function is a logistic sigmoid
$$
y = \sigma(a) \equiv \frac{1}{1 + exp(-a)}
\tag{5.19}
$$
so that $0 \leq y(x,w) \leq 1$. We can interpret $y(x,w)$ as the conditional probability
$p(C_1|x)$, with $p(C_2|x)$ given by $1 − y(x,w)$. The conditional distribution of targets
given inputs is then a Bernoulli distribution of the form
$$
p(t|x,w) = y(x,w)^t \{1 − y(x,w)\}^{1−t}
\tag{5.20}
$$
If we consider a training set of independent observations, then the error function,
which is given by the negative log likelihood, is then a [[cross-entropy]] error function
of the form
$$
E(w) = -\sum^N_{n=1}\{t_nln(y_n)+(1-t_n)ln(1-)\}
\tag{5.21}
$$
where $y_n$ denotes $y(x_n,w)$. Note that there is no analogue of the noise precision $\beta$
because the target values are assumed to be correctly labelled. However, the model
is easily extended to allow for labelling errors. Simard et al. (2003) found that using the cross-entropy error function instead of the sum-of-squares for a classification problem leads to faster training as well as improved generalization.

If we have *K* separate binary classifications to perform, then we can use a network
having *K* outputs each of which has a logistic sigmoid activation function.
Associated with each output is a binary class label $t_k \in \{0, 1\}$, where $k = 1, . . . , K$.
If we assume that the class labels are independent, given the input vector, then the
conditional distribution of the targets is
$$
p(t|x,w) = \prod^K_{k=1}y_k(x,w)^{}
\tag{5.22}t_k[1 - y_k(x,w)]^{1-t_k}
$$
Taking the negative logarithm of the corresponding likelihood function then gives the following error function
$$
E(w) = -\sum^N_{n=1}\sum^K_{k=1}\{t_{nk}ln(y_{nk})+(1-t_{nk})ln(1-y_nk)\}
\tag{5.23}
$$
where $y_{nk}$ denotes $y_k(x_n,w)$. Again, the derivative of the error function with respect to the activation for a particular output unit takes the form (5.18) just as in the
regression case.

It is interesting to contrast the neural network solution to this problem with the
corresponding approach based on a linear classification model of the kind discussed in [[Chapter 4]]. Suppose that we are using a standard two-layer network of the kind shown in [[Figure 5.1.png|Figure 5.1]]. We see that the weight parameters in the first layer of the network are shared between the various outputs, whereas in the linear model each classification problem is solved independently. The first layer of the network can be viewed as performing a nonlinear feature extraction, and the sharing of features between the different outputs can save on computation and can also lead to improved generalization.

Finally, we consider the standard multiclass classification problem in which each
input is assigned to one of *K* mutually exclusive classes. The binary target variables
$t_k \in \{0, 1\}$ have a 1-of-*K* coding scheme indicating the class, and the network
outputs are interpreted as $y_k(x,w) = p(t_k = 1|x)$, leading to the following error
function
$$
E(w) = -\sum^N_{n=1}\sum^K_{k=1}t_{kn}ln\ y_k(x_n,w)
\tag{5.24}
$$
Following the discussion of [[Section 4.3.4]], we see that the output unit activation
function, which corresponds to the canonical link, is given by the softmax function
$$
y_k(x,w)=\frac{exp(a_k(x,w))}{\sum_jexp(a_j(x,w))}
\tag{5.25}
$$
which satisfies $0 \leq y_k \leq 1$ and $\sum_k y_k = 1$. Note that the $y_k(x,w)$ are unchanged
if a constant is added to all of the $a_k(x,w)$, causing the error function to be constant for some directions in weight space. This degeneracy is removed if an appropriate regularization term ([[Section 5.5]]) is added to the error function.

Once again, the derivative of the error function with respect to the activation for a particular output unit takes the familiar form (5.18).

In summary, there is a natural choice of both output unit activation function
and matching error function, according to the type of problem being solved. For regression we use linear outputs and a sum-of-squares error, for (multiple independent) binary classifications we use logistic sigmoid outputs and a cross-entropy error function,and for multiclass classification we use softmax outputs with the corresponding multiclass cross-entropy error function. For classification problems involving two classes, we can use a single logistic sigmoid output, or alternatively we can use a network with two outputs having a softmax output activation function.


## Index
- [[1. Parameter optimization]]
- [[2. Local quadratic approximation]]
- [[3. Use of gradient information]]
- [[4. Gradient descent optimization]]
