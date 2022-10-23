We begin our discussion of support vector machines by returning to the two-class classification problem using linear models of the form
$$
y(x) = w^T\phi(x)+b
\tag{7.1}
$$
where $\phi(x)$ denotes a fixed feature-space transformation, and we have made the bias parameter *b* explicit. Note that we shall shortly introduce a dual representation expressed in terms of kernel functions, which avoids having to work explicitly in feature space. The training data set comprises *N* input vectors $x_1, . . . , x_N$, with corresponding target values $t_1, . . . , t_N$ where $t_n \in \{−1, 1\}$, and new data points *x* are classified according to the sign of $y(x)$.

We shall assume for the moment that the training data set is linearly separable in feature space, so that by definition there exists at least one choice of the parameters *w* and *b* such that a function of the form (7.1) satisfies $y(x_n) > 0$ for points having  $t_n = +1$ and $y(x_n) < 0$ for points having $t_n = −1$, so that $t_ny(x_n) > 0$ for all training data points.

There may of course exist many such solutions that separate the classes exactly. In [[Section 4.1.7]], we described the perceptron algorithm that is guaranteed to find a solution in a finite number of steps. The solution that it finds, however, will be dependent on the (arbitrary) initial values chosen for *w* and *b* as well as on the order in which the data points are presented. If there are multiple solutions all of which classify the training data set exactly, then we should try to find the one that
will give the smallest generalization error. The support vector machine approaches this problem through the concept of the [[margin]], which is defined to be the smallest distance between the decision boundary and any of the samples, as illustrated in [[Figure 7.1.png|Figure 7.1]].

![[Figure 7.1.png]]
[[Figure 7.1.png|Figure 7.1]]

In support vector machines the decision boundary is chosen to be the one for which the margin is maximized. The maximum margin solution can be motivated using [[computational learning theory]], also known as [[statistical learning theory]]. However, a simple insight into the origins of maximum margin has been given by Tong and Koller (2000) who consider a framework for classification based on a hybrid of generative and discriminative approaches. They first model the distribution over input vectors *x* for each class using a Parzen density estimator with Gaussian kernels having a common parameter $\sigma^2$. Together with the class priors, this defines an optimal misclassification-rate decision boundary. However, instead of using this optimal boundary, they determine the best hyperplane by minimizing the probability of error relative to the learned density model. In the limit $\sigma^2 \rightarrow 0$, the optimal hyperplane is shown to be the one having maximum margin. The intuition behind this result is that as $\sigma^2$ is reduced, the hyperplane is increasingly dominated by nearby data points relative to more distant ones. In the limit, the hyperplane becomes independent of data points that are not support vectors.

We shall see in [[Figure 10.13]] that marginalization with respect to the prior distribution of the parameters in a Bayesian approach for a simple linearly separable data set leads to a decision boundary that lies in the middle of the region separating the data points. The large margin solution has similar behaviour.

Recall from [[Figure 4.1]] that the perpendicular distance of a point *x* from a hyperplane defined by $y(x) = 0$ where $y(x)$ takes the form (7.1) is given by $|y(x)|/||w||$. Furthermore, we are only interested in solutions for which all data points are correctly classified, so that $t_ny(x_n) > 0$ for all n. Thus the distance of a point $x_n$ to the decision surface is given by
$$
\frac{t_ny(x_n)}{||w||} = \frac{t_n(w^T\phi(x_n)+b)}{||w||}
\tag{7.2}
$$
The margin is given by the perpendicular distance to the closest point $x_n$ from the
data set, and we wish to optimize the parameters *w* and *b* in order to maximize this
distance. Thus the maximum margin solution is found by solving
$$
arg\ max_{w,b}\ \{\frac{1}{||w||}min_n[t_n(w^T\phi(x_n) + b]\}
\tag{7.3}
$$
where we have taken the factor $1/||w||$ outside the optimization over *n* because *w* does not depend on *n*. Direct solution of this optimization problem would be very complex, and so we shall convert it into an equivalent problem that is much easier to solve. To do this we note that if we make the rescaling $w \rightarrow \kappa w$ and $b \rightarrow \kappa b$, then the distance from any point $x_n$ to the decision surface, given by $t_ny(xn)/||w||$, is unchanged. We can use this freedom to set
$$
t_n(w^T\phi(x_n) + b) = 1
\tag{7.4}
$$
for the point that is closest to the surface. In this case, all data points will satisfy the
constraints
$$
t_n(w^T\phi(x_n) + b) \geq 1, n = 1,...,N
\tag{7.5}
$$
This is known as the canonical representation of the decision hyperplane. In the case of data points for which the equality holds, the constraints are said to be [[active]], whereas for the remainder they are said to be [[inactive]]. By definition, there will always be at least one active constraint, because there will always be a closest point, and once the margin has been maximized there will be at least two active constraints. The optimization problem then simply requires that we maximize
$||w||^{−1}$, which is equivalent to minimizing $||w||^2$, and so we have to solve the optimization problem
$$
arg\ min_{w,b}\frac{1}{2}||w||^2
\tag{7.6}
$$
subject to the constraints given by (7.5). The factor of 1/2 in (7.6) is included for later convenience. This is an example of a [[quadratic programming]] problem in which we are trying to minimize a quadratic function subject to a set of linear inequality constraints. It appears that the bias parameter *b* has disappeared from the optimization. However, it is determined implicitly via the constraints, because these require that changes to ||w|| be compensated by changes to *b*. We shall see how this works shortly.

In order to solve this constrained optimization problem, we introduce Lagrange multipliers $a_n \geq 0$, with one multiplier $a_n$ for each of the constraints in (7.5), giving the Lagrangian function
$$
L(w,b,a) = \frac{1}{2}||w||^2 - \sum^N_{n=1}a_n\{t_n(w^T\phi(x_n)+b)-1\}
\tag{7.7}
$$
where $a = (a_1, . . . , a_N)^T$. Note the minus sign in front of the Lagrange multiplier term, because we are minimizing with respect to *w* and *b*, and maximizing with respect to *a*.

## Index
- [[1. Overlapping class distributions]]
- [[2. Relation to logistic regression]]
- [[3. Multiclass SVMs]]
- [[4. SVMs for regression]]
- [[5. Computational learning theory]]