# Combining Models
In earlier chapters, we have explored a range of different models for solving classification and regression problems. It is often found that improved performance can be obtained by combining multiple models together in some way, instead of just using a single model in isolation. For instance, we might train *L* different models and then make predictions using the average of the predictions made by each model. Such combinations of models are sometimes called [[Committees]]. In [[Section 14.2]], we discuss ways to apply the committee concept in practice, and we also give some insight into why it can sometimes be an effective procedure.

One important variant of the committee method, known as [[boosting]], involves
training multiple models in sequence in which the error function used to train a particular model depends on the performance of the previous models. This can produce substantial improvements in performance compared to the use of a single model and is discussed in [[Section 14.3]].

Instead of averaging the predictions of a set of models, an alternative form of model combination is to select one of the models to make the prediction, in which
the choice of model is a function of the input variables. Thus different models become responsible for making predictions in different regions of input space. One
widely used framework of this kind is known as a [[decision tree]] in which the selection process can be described as a sequence of binary selections corresponding to the traversal of a tree structure and is discussed in [[Section 14.4]]. In this case, the individual models are generally chosen to be very simple, and the overall flexibility of the model arises from the input-dependent selection process. Decision trees can be applied to both classification and regression problems.

One limitation of decision trees is that the division of input space is based on
hard splits in which only one model is responsible for making predictions for any
given value of the input variables. The decision process can be softened by moving
to a probabilistic framework for combining models, as discussed in Section 14.5. For
example, if we have a set of K models for a conditional distribution $p(t|x, k)$ where
*x* is the input variable, t is the target variable, and $k = 1, . . . , K$ indexes the model,
then we can form a probabilistic mixture of the form
$$
p(t|x) = \sum^K_{k=1}\pi_k(x)p(t|x,k)
\tag{14.1}
$$
in which $\pi_k(x) = p(k|x)$ represent the input-dependent mixing coefficients. Such
models can be viewed as mixture distributions in which the component densities, as
well as the mixing coefficients, are conditioned on the input variables and are known as [[mixtures of experts]]. They are closely related to the mixture density network model discussed in [[Section 5.6]].

## Index
- [[Bayesian Model Averaging]]
- [[Committees]]
- [[Boosting]]
- [[Tree-based Models]]
- [[Conditional Mixture Models]]