The simplest way to construct a committee is to average the predictions of a set of individual models. Such a procedure can be motivated from a frequentist perspective by considering the trade-off between bias and variance, which decomposes the error due to a model into the bias component that arises from differences between the model and the true function to be predicted, and the variance component that represents the sensitivity of the model to the individual data points. Recall from [[Figure 3.5.png|Figure 3.5]] that when we trained multiple polynomials using the sinusoidal data, and then averaged the resulting functions, the contribution arising from the variance term tended to cancel, leading to improved predictions. When we averaged a set of low-bias models (corresponding to higher order polynomials), we obtained accurate predictions for the underlying sinusoidal function from which the data were generated.

In practice, of course, we have only a single data set, and so we have to find a way to introduce variability between the different models within the committee. One approach is to use [[bootstrap data sets]], discussed in [[3. Bayesian probabilities|Section 1.2.3]]. Consider a regression problem in which we are trying to predict the value of a single continuous variable, and suppose we generate *M* bootstrap data sets and then use each to train a separate copy $y_m(x)$ of a predictive model where $m = 1, . . . , M$. The committee prediction is given by
$$
y_{COM}(x) = \frac{1}{M}\sum^M_{m=1}y_m(x)
\tag{14.7}
$$
This procedure is known as bootstrap aggregation or [[bagging]] (Breiman, 1996).

Suppose the true regression function that we are trying to predict is given by $h(x)$, so that the output of each of the models can be written as the true value plus an error in the form
$$
y_m(x) = h(x) + \epsilon_m(x)
\tag{14.8}
$$
The average sum-of-squares error then takes the form
$$
\mathbb{E}_x[\{y_m(x) - h(x)\}^2] = \mathbb{E}_x[\epsilon_m(x)^2]
\tag{14.9}
$$
where $E_x[·]$ denotes a frequentist expectation with respect to the distribution of the input vector *x*. The average error made by the models acting individually is therefore
$$

E_{AV}=\frac{1}{M}\sum^M_{m=1}\mathbb{E}_x[\epsilon_m(x)^2]
\tag{14.10}
$$
Similarly, the expected error from the committee (14.7) is given by
$$
\begin{align}
E_{COM}=\mathbb{E}_x[\{\frac{1}{M}\sum^M_{m=1}y_m(x)-h(x)\}^2]\\
= \mathbb{E}_x[\{\frac{1}{M}\sum^M_{m=1}\epsilon_m(x)\}^2]
\end{align}
\tag{14.11}
$$
If we assume that the errors have zero mean and are uncorrelated, so that
$$
\mathbb{E}_x[\epsilon_m(x)] = 0
\tag{14.12}
$$
$$
\mathbb{E}_x[\epsilon_m(x)\epsilon_l(x)] = 0, m \neq l
\tag{14.13}
$$
then we obtain
$$
E_{COM} = \frac{1}{M}E_{AV}
\tag{14.14}
$$
This apparently dramatic result suggests that the average error of a model can be reduced by a factor of *M* simply by averaging *M* versions of the model. Unfortunately, it depends on the key assumption that the errors due to the individual models are uncorrelated. In practice, the errors are typically highly correlated, and the reduction in overall error is generally small. It can, however, be shown that the expected committee error will not exceed the expected error of the constituent models, so that $E_{COM} \leq E_{AV}$. In order to achieve more significant improvements, we turn to a more sophisticated technique for building committees, known as boosting.