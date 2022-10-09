# Nonparametric Methods
Throughout this chapter, we have focussed on the use of probability distributions having specific functional forms governed by a small number of parameters whose values are to be determined from a data set. This is called the [[parametric]] approach to density modelling. An important limitation of this approach is that the chosen density might be a poor model of the distribution that generates the data, which can result in poor predictive performance. For instance, if the process that generates the data is multimodal, then this aspect of the distribution can never be captured by a
Gaussian, which is necessarily unimodal.

In this final section, we consider some [[nonparametric]] approaches to density estimation that make few assumptions about the form of the distribution. Here we shall focus mainly on simple frequentist methods. The reader should be aware, however, that nonparametric Bayesian methods are attracting increasing interest (Walker et al., 1999; Neal, 2000; M¨uller and Quintana, 2004; Teh et al., 2006).

Let us start with a discussion of histogram methods for density estimation, which we have already encountered in the context of marginal and conditional distributions in [[Figure 1.11.png|Figure 1.11]] and in the context of the central limit theorem in [[Figure 2.6.png|Figure 2.6]]. Here we explore the properties of histogram density models in more detail, focussing on the case of a single continuous variable 
*x*. Standard histograms simply partition *x* into distinct bins of width $\Delta_i$ and then count the number $n_i$ of observations of *x* falling in bin *i*. In order to turn this count into a normalized probability density, we simply divide by the total number *N* of observations and by the width $\Delta_i$ of the bins to obtain probability values for each bin given by
$$
p_i = \frac{n_i}{N\Delta_i}
\tag{2.241}
$$
for which it is easily seen that $\int p(x) dx = 1$. This gives a model for the density
*p(x)* that is constant over the width of each bin, and often the bins are chosen to have the same width $\Delta_i = \Delta$.

In [[Figure 2.24.png|Figure 2.24]], we show an example of histogram density estimation. Here
the data is drawn from the distribution, corresponding to the green curve, which is formed from a mixture of two Gaussians. Also shown are three examples of histogram density estimates corresponding to three different choices for the bin width $\Delta$. We see that when $\Delta$ is very small (top figure), the resulting density model is very spiky, with a lot of structure that is not present in the underlying distribution that generated the data set. Conversely, if $\Delta$ is too large (bottom figure) then the result is a model that is too smooth and that consequently fails to capture the bimodal property of the green curve. The best results are obtained for some intermediate value of $\Delta$ (middle figure). In principle, a histogram density model is also dependent on the choice of edge location for the bins, though this is typically much less significant than the value of $\Delta$.

![[Figure 2.24.png]]
[[Figure 2.24.png|Figure 2.24]]

Note that the histogram method has the property (unlike the methods to be discussed shortly) that, once the histogram has been computed, the data set itself can be discarded, which can be advantageous if the data set is large. Also, the histogram approach is easily applied if the data points are arriving sequentially.

In practice, the histogram technique can be useful for obtaining a quick visualization of data in one or two dimensions but is unsuited to most density estimation applications. One obvious problem is that the estimated density has discontinuities that are due to the bin edges rather than any property of the underlying distribution that generated the data. Another major limitation of the histogram approach is its scaling with dimensionality. If we divide each variable in a *D*-dimensional space into *M* bins, then the total number of bins will be $M^D$. This exponential scaling with *D* is an example of the curse of dimensionality. In a space of high dimensionality, the quantity of data needed to provide meaningful estimates of local probability density would be prohibitive.

The histogram approach to density estimation does, however, teach us two important lessons. First, to estimate the probability density at a particular location, we should consider the data points that lie within some local neighbourhood of that point. Note that the concept of locality requires that we assume some form of distance measure, and here we have been assuming Euclidean distance. For histograms, this neighbourhood property was defined by the bins, and there is a natural ‘smoothing’ parameter describing the spatial extent of the local region, in this case the bin width. Second, the value of the smoothing parameter should be neither too large nor too small in order to obtain good results. This is reminiscent of the choice of model complexity in polynomial curve fitting discussed in [[Introduction|Chapter 1]] where the degree *M* of the polynomial, or alternatively the value $\alpha$ of the regularization parameter, was optimal for some intermediate value, neither too large nor too small. Armed with these insights, we turn now to a discussion of two widely used nonparametric techniques for density estimation, kernel estimators and nearest neighbours, which have better scaling with dimensionality than the simple histogram model.

## Index
- [[1. Kernel density estimators]]
- [[2. Nearest-neighbour methods]]