We turn next to a probabilistic view of classification and show how models with linear decision boundaries arise from simple assumptions about the distribution of the data. In [[4. Inference and decision|Section 1.5.4]], we discussed the distinction between the discriminative and the generative approaches to classification. Here we shall adopt a generative approach in which we model the class-conditional densities p(x|Ck), as well as the class priors p(Ck), and then use these to compute posterior probabilities p(Ck|x) through Bayes’ theorem.

Consider first of all the case of two classes. The posterior probability for class $C_1$ can be written as
$$
\begin{align}
p(C_1|x) = \frac{p(x|C_1)p(C_1)}{p(x|C_1)p(C_1) + p(x|C_2)p(C_2)} \\
= \frac{1}{1 + exp(-a)} = \sigma(a)
\end{align}
\tag{4.57}
$$
where we have defined
$$
a = ln\frac{p(x|C_1)p(C_1)}{p(x|C_2)p(C_2)}
\tag{4.58}
$$
and $\sigma(a)$ is the [[logistic sigmoid]] function defined by
$$
\sigma(a) = \frac{1}{1 + exp(-a)}
\tag{4.59}
$$
which is plotted in [[Figure 4.9.png|Figure 4.9]]. The term ‘sigmoid’ means S-shaped. This type of function is sometimes also called a ‘squashing function’ because it maps the whole real axis into a finite interval. The logistic sigmoid has been encountered already in earlier chapters and plays an important role in many classification algorithms. It satisfies the following symmetry property
$$
\sigma(-a) = 1 - \sigma(a)
\tag{4.60}
$$
as is easily verified. The inverse of the logistic sigmoid is given by
$$
a = \ln{\frac{\sigma}{1 - \sigma}}
\tag{4.61}
$$
and is known as the logit function. It represents the log of the ratio of probabilities ln $[p(C1|x)/p(C2|x)]$ for the two classes, also known as the log odds.

![[Figure 4.9.png]]
[[Figure 4.9.png|Figure 4.9]]

Note that in (4.57) we have simply rewritten the [[posterior probabilities]] in an equivalent form, and so the appearance of the logistic sigmoid may seem rather vacuous. However, it will have significance provided *a(x)* takes a simple functional form. We shall shortly consider situations in which *a(x)* is a linear function of *x*, in which case the [[posterior probability]] is governed by a generalized linear model. For the case of *K >2* classes, we have
$$
\begin{align}
p(C_k|x) = \frac{p(x|C_k)p(C_k)}{\sum_jp(x|C_j)p(C_j)} \\
= \frac{exp(a_k)}{\sum_jexp(a_j)}
\end{align}
\tag{4.62}
$$
which is known as the [[normalized exponential]] and can be regarded as a multiclass generalization of the logistic sigmoid. Here the quantities $a_k$ are defined by
$$
a_k = ln(p(x|C_k)p(C_k))
\tag{4.63}
$$
The normalized exponential is also known as the [[softmax function]], as it represents a smoothed version of the ‘max’ function because, if $a_k \gg a_j$  for all $j \neq k$, then $p(C_k|x) \approx 1$, and $p(C_j|x) \approx 0$.

We now investigate the consequences of choosing specific forms for the classconditional densities, looking first at continuous input variables *x* and then discussing briefly the case of discrete inputs.

## Index
- [[1. Continuous inputs]]
- [[2. Maximum likelihood solution]]