So far in our discussion of linear models for regression, we have assumed that the form and number of basis functions are both fixed. As we have seen in [[Introduction|Chapter 1]], the use of maximum likelihood, or equivalently least squares, can lead to severe over-fitting if complex models are trained using data sets of limited size. However, limiting the number of basis functions in order to avoid over-fitting has the side effect of limiting the flexibility of the model to capture interesting and important trends in the data. Although the introduction of regularization terms can control over-fitting for models with many parameters, this raises the question of how to determine a suitable value for the regularization coefficient $\lambda$. Seeking the solution that minimizes the regularized error function with respect to both the weight vector w and the regularization coefficient $\lambda$ is clearly not the right approach since this leads to the unregularized solution with $\lambda$ = 0.

As we have seen in earlier chapters, the phenomenon of over-fitting is really an unfortunate property of maximum likelihood and does not arise when we marginalize over parameters in a Bayesian setting. In this chapter, we shall consider the Bayesian view of model complexity in some depth. Before doing so, however, it is instructive to consider a frequentist viewpoint of the model complexity issue, known as the [[bias variance]] trade-off. Although we shall introduce this concept in the context of linear basis function models, where it is easy to illustrate the ideas using simple examples, the discussion has more general applicability.

In [[Section 1.5.5]], when we discussed decision theory for regression problems, we considered various loss functions each of which leads to a corresponding optimal prediction once we are given the conditional distribution *p(t|**x**).* A popular choice is the squared loss function, for which the optimal prediction is given by the conditional expectation, which we denote by *h(**x**)* and which is given by
$$
h(x) = \mathbb{E}[t|x] = \int tp(t|x)dt
\tag{3.36}
$$
At this point, it is worth distinguishing between the squared loss function arising from decision theory and the sum-of-squares error function that arose in the maximum likelihood estimation of model parameters. We might use more sophisticated techniques than least squares, for example regularization or a fully Bayesian approach, to determine the conditional distribution *p(t|**x**)*. These can all be combined with the squared loss function for the purpose of making predictions.

We showed in [[Section 1.5.5]] that the expected squared loss can be written in the form
$$
\mathbb{E}[L] = \int \{y(x) - h(x)\}^2p(x)dx + \int\{h(x)-t\}^2p(x,t)dxdt
\tag{3.37}
$$
Recall that the second term, which is independent of $y(x)$, arises from the intrinsic noise on the data and represents the minimum achievable value of the expected loss. The first term depends on our choice for the function $y(x)$, and we will seek a solution for $y(x)$ which makes this term a minimum. Because it is nonnegative, the smallest that we can hope to make this term is zero. If we had an unlimited supply of data (and unlimited computational resources), we could in principle find the regression function $h(x)$ to any desired degree of accuracy, and this would represent the optimal choice for $y(x)$. However, in practice we have a data set *D* containing only a finite number *N* of data points, and consequently we do not know the regression function $h(x)$ exactly.

If we model the $h(x)$ using a parametric function $y(x,w)$ governed by a parameter vector *w*, then from a Bayesian perspective the uncertainty in our model is expressed through a posterior distribution over *w*. A frequentist treatment, however, involves making a point estimate of w based on the data set *D*, and tries instead to interpret the uncertainty of this estimate through the following thought experiment. Suppose we had a large number of data sets each of size *N* and each drawn independently from the distribution $p(t, x)$. For any given data set *D*, we can run our learning algorithm and obtain a prediction function $y(x;D)$. Different data sets from the ensemble will give different functions and consequently different values of the squared loss. The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets.

Consider the integrand of the first term in (3.37), which for a particular data set *D* takes the form
$$
\{y(x;D)-h(x)\}^2
\tag{3.38}
$$
Because this quantity will be dependent on the particular data set *D*, we take its average over the ensemble of data sets. If we add and subtract the quantity $E_D[y(x;D)]$ inside the braces, and then expand, we obtain
$$
\begin{align}
\{y(x;D) - \mathbb{E}_D[y(x;D)] + \mathbb{E}_D[y(x;D)] - h(x)\}^2 \\
= \{y(x;D) - \mathbb{E}_D[y(x;D)]\}^2 + \{\mathbb{E}_D[y(x'D)] - h(x)\}^2 \\
+ 2\{y(x;D) - \mathbb{E}_D[y(x;D)]\}\{\mathbb(E)_D[y(x;D)] - h(x)\}
\end{align}
\tag{3.39}
$$
We now take the expectation of this expression with respect to *D* and note that the final term will vanish, giving

![[Pasted image 20221009201341.png]]

We see that the expected squared difference between $y(x;D)$ and the regression function $h(x)$ can be expressed as the sum of two terms. The first term, called the squared bias, represents the extent to which the average prediction over all data sets differs from the desired regression function. The second term, called the variance, measures the extent to which the solutions for individual data sets vary around their average, and hence this measures the extent to which the function $y(x;D)$ is sensitive to the particular choice of data set. We shall provide some intuition to support these definitions shortly when we consider a simple example.

So far, we have considered a single input value x. If we substitute this expansion back into (3.37), we obtain the following decomposition of the expected squared loss
$$
expected\ loss = (bias)^2 + variance + noise
\tag{3.41}
$$
where
$$
(bias)^2 = \int\{\mathbb{E}_D[y(x;D)] - h(x)\}^2p(x)dx
\tag{3.42}
$$
$$
variance = \int\mathbb{E}_D\{[y(x;D)] - \mathbb{E}_D[y(x;D)]\}^2p(x)dx
\tag{3.42}
$$
$$
noise = \int\{h(x) - t\}^2p(x,t)dxdt
\tag{3.42}
$$
and the bias and variance terms now refer to integrated quantities.

Our goal is to minimize the expected loss, which we have decomposed into the sum of a (squared) bias, a variance, and a constant noise term. As we shall see, there is a trade-off between bias and variance, with very flexible models having low bias and high variance, and relatively rigid models having high bias and low variance. The model with the optimal predictive capability is the one that leads to the best balance between bias and variance. This is illustrated by considering the sinusoidal data set from [[Introduction|Chapter 1]]. Here we generate 100 data sets, each containing *N* = 25 data points, independently from the sinusoidal curve $h(x) = \sin(2\pi x)$. The data sets are indexed by $l = 1, ..., L$, where *L* = 100, and for each data set $D^{(l)}$ we fit a model with 24 Gaussian basis functions by minimizing the regularized error function (3.27) to give a prediction function $y^{(l)}(x)$ as shown in [[Figure 3.5.png|Figure 3.5]]. The top row corresponds to a large value of the regularization coefficient $\lambda$ that gives low variance (because the red curves in the left plot look similar) but high bias (because the two curves in the right plot are very different). Conversely on the bottom row, for which $\lambda$ is small, there is large variance (shown by the high variability between the red curves in the left plot) but low bias (shown by the good fit between the average model fit and the original sinusoidal function). Note that the result of averaging many solutions for the complex model with *M* = 25 is a very good fit to the regression function, which suggests that averaging may be a beneficial procedure. Indeed, a weighted averaging of multiple solutions lies at the heart of a Bayesian approach, although the averaging is with respect to the posterior distribution of parameters, not with respect to multiple data sets.

![[Figure 3.5.png]]
[[Figure 3.5.png|Figure 3.5]]

We can also examine the bias-variance trade-off quantitatively for this example. The average prediction is estimated from
$$
\overline{y}(x) = \frac{1}{L}\sum^L_{l=1}y^{(l)}(x)
\tag{3.45}
$$
and the integrated squared bias and integrated variance are then given by
$$
(bias)^2 = \frac{1}{N}\sum^N_{n=1}\{\overline{y}(x_n)-h(x_n)\}^2
\tag{3.46}
$$
$$
variance = \frac{1}{N}\sum^N_{n=1}\frac{1}{L}\sum^L_{l=1}\{y^{(l)}(x_n)- \overline{y}(x_n)\}^2
\tag{3.46}
$$
where the integral over x weighted by the distribution $p(x)$ is approximated by a finite sum over data points drawn from that distribution. These quantities, along with their sum, are plotted as a function of ln $\lambda$ in [[Figure 3.6.png|Figure 3.6]]. We see that small values of $\lambda$ allow the model to become finely tuned to the noise on each individual data set leading to large variance. Conversely, a large value of $\lambda$ pulls the weight parameters towards zero leading to large bias.

![[Figure 3.6.png]]
[[Figure 3.6.png|Figure 3.6]]

Although the bias-variance decomposition may provide some interesting insights into the model complexity issue from a frequentist perspective, it is of limited practical value, because the bias-variance decomposition is based on averages with respect to ensembles of data sets, whereas in practice we have only the single observed data set. If we had a large number of independent training sets of a given size, we would be better off combining them into a single large training set, which of course would reduce the level of over-fitting for a given model complexity.

Given these limitations, we turn in the next section to a Bayesian treatment of
linear basis function models, which not only provides powerful insights into the issues of over-fitting but which also leads to practical techniques for addressing the question model complexity.