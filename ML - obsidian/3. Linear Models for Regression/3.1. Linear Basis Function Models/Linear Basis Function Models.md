# Linear Basis Function Models

The simplest linear model for regression is one that involves a linear combination of
the input variables
$$
y(x,w) = w_0 + w_1x_1 + ... + w_Dx_D
\tag{3.1}
$$
where $x = (x_1, ..., x_D)^T$. This is often simply known as [[linear regression]]. The key property of this model is that it is a linear function of the parameters $w_0, ..., w_D$. It is also, however, a linear function of the input variables $x_i$, and this imposes significant limitations on the model. We therefore extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form
$$
y(x,w) = w_0 =\sum^{m-1}_{j=1}w_j\phi_j(x)
\tag{3.2}
$$
where $\phi_j(x)$ are known as [[basis functions]]. By denoting the maximum value of the index *j* by *M* − 1, the total number of parameters in this model will be *M*.


The parameter $w_0$ allows for any fixed offset in the data and is sometimes called
a [[bias]] parameter (not to be confused with ‘bias’ in a statistical sense). It is often
convenient to define an additional dummy ‘basis function’ $\phi_0(x) = 1$ so that
$$
y(x,w) =\sum^{M-1}_{j=0}w_j\phi_j(x) = w^T\phi(x)
\tag{3.3}
$$
where $w = (w_0, ..., w_{M−1})^T$ and $\phi = (\phi_0, ..., \phi_{M−1})^T.$ In many practical applications of pattern recognition, we will apply some form of fixed pre-processing, or feature extraction, to the original data variables. If the original variables comprise the vector *x*, then the features can be expressed in terms of the basis functions $\{\phi_j(x)\}$.

By using nonlinear basis functions, we allow the function $y(x,w)$ to be a nonlinear
function of the input vector **x**. Functions of the form (3.2) are called linear models, however, because this function is linear in **w**. It is this linearity in the parameters
that will greatly simplify the analysis of this class of models. However, it also leads to some significant limitations, as we discuss in [[Section 3.6]].

The example of polynomial regression considered in [[Introduction|Chapter 1]] is a particular
example of this model in which there is a single input variable x, and the basis functions take the form of powers of *x* so that $\phi_j(x) = x^j$ . One limitation of polynomial basis functions is that they are global functions of the input variable, so that changes in one region of input space affect all other regions. This can be resolved by dividing the input space up into regions and fit a different polynomial in each region, leading to [[spline functions]] (Hastie et al., 2001).

There are many other possible choices for the basis functions, for example
$$
\phi_j(x) = exp\{\ -\frac{(x-\mu_j)^2}{2s^2}\}
\tag{3.4}
$$
where the $\mu_j$ govern the locations of the basis functions in input space, and the parameter *s* governs their spatial scale. These are usually referred to as ‘Gaussian’
basis functions, although it should be noted that they are not required to have a probabilistic interpretation, and in particular the normalization coefficient is unimportant because these basis functions will be multiplied by adaptive parameters $w_j$.

Another possibility is the sigmoidal basis function of the form
$$
\phi_j(x) = \sigma(\frac{x-\mu_j}{s})
\tag{3.5}
$$
where $\sigma(a)$ is the logistic sigmoid function defined by
$$
\sigma(a) = \frac{1}{1 + exp(-a)}
\tag{3.6}
$$
Equivalently, we can use the ‘tanh’ function because this is related to the logistic
sigmoid by $tanh(a) = 2\sigma(a) − 1$, and so a general linear combination of logistic sigmoid functions is equivalent to a general linear combination of ‘tanh’ functions.
These various choices of basis function are illustrated in Figure 3.1.

![[Figure 3.1.png]]
[[Figure 3.1.png|Figure 3.1]]

Yet another possible choice of basis function is the Fourier basis, which leads to
an expansion in sinusoidal functions. Each basis function represents a specific frequency and has infinite spatial extent. By contrast, basis functions that are localized to finite regions of input space necessarily comprise a spectrum of different spatial frequencies. In many signal processing applications, it is of interest to consider basis functions that are localized in both space and frequency, leading to a class of functions known as [[wavelets]]. These are also defined to be mutually orthogonal, to simplify their application. Wavelets are most applicable when the input values live on a regular lattice, such as the successive time points in a temporal sequence, or the pixels in an image. Useful texts on wavelets include Ogden (1997), Mallat (1999), and Vidakovic (1999).

Most of the discussion in this chapter, however, is independent of the particular
choice of basis function set, and so for most of our discussion we shall not specify
the particular form of the basis functions, except for the purposes of numerical illustration. Indeed, much of our discussion will be equally applicable to the situation in which the vector φ(x) of basis functions is simply the identity $\phi(x) = x$. Furthermore, in order to keep the notation simple, we shall focus on the case of a single target variable *t*. However, in [[Section 3.1.5]], we consider briefly the modifications needed to deal with multiple target variables.

## Index
- [[1. Maximum likelihood and least squares]]
- [[2. Geometry of least squares]]
- [[3. Sequential learning]]
- [[4. Regularized least squares]]
- [[5. Multiple outputs]]