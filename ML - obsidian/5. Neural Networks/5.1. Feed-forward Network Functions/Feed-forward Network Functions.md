The linear models for regression and classification discussed in [[Linear Models for Regression|Chapters 3]] and [[Linear Models for Classification|4]], respectively, are based on linear combinations of fixed nonlinear basis functions $\phi_j(x)$ and take the form
$$
y(x,w) = f(\sum^M_{j=1}w_j\phi_j(x))
\tag{5.1}
$$
where $f(·)$ is a nonlinear activation function in the case of classification and is the identity in the case of regression. Our goal is to extend this model by making the basis functions $\phi_j(x)$ depend on parameters and then to allow these parameters to be adjusted, along with the coefficients $\{w_j\}$, during training. There are, of course, many ways to construct parametric nonlinear basis functions. Neural networks use basis functions that follow the same form as (5.1), so that each basis function is itself a nonlinear function of a linear combination of the inputs, where the coefficients in the linear combination are adaptive parameters.

This leads to the basic neural network model, which can be described a series of functional transformations. First we construct *M* linear combinations of the input variables $x_1, . . . , x_D$ in the form
$$
a_j = \sum^D_{i=1}w^{(1)}_{ji}x_i+w^{(1)}_{j0}
\tag{5.2}
$$
where $j = 1, . . . , M$, and the superscript (1) indicates that the corresponding parameters are in the first ‘layer’ of the network. We shall refer to the parameters $w^{(1)}_{ji}$ as [[weights]] and the parameters $w^{(1)}_{j0}$ as [[biases]], following the nomenclature of [[Linear Models for Regression|Chapter 3]]. The quantities $a_j$ are known as [[activations]]. Each of them is then transformed using a differentiable, nonlinear [[activation function]] $h(·)$ to give
$$
z_j = h(a_j)
\tag{5.3}
$$
These quantities correspond to the outputs of the basis functions in (5.1) that, in the context of neural networks, are called [[hidden units]]. The nonlinear functions $h(·)$ are generally chosen to be sigmoidal functions such as the logistic sigmoid or the ‘tanh’ function. Following (5.1), these values are again linearly combined to give [[output unit activations]]
$$
a_k = \sum^M_{j=1}w^{(2)}_{kj}z_j+w^{(2)}_{k0}
\tag{5.4}
$$
where $k = 1, . . . , K$, and *K* is the total number of outputs. This transformation corresponds to the second layer of the network, and again the $w^{(2)}_{k0}$ are bias parameters. Finally, the output unit activations are transformed using an appropriate activation function to give a set of network outputs $y_k$. The choice of activation function is determined by the nature of the data and the assumed distribution of target variables and follows the same considerations as for linear models discussed in [[Linear Models for Regression|Chapters 3]] and [[Linear Models for Classification|4]]. Thus for standard regression problems, the activation function is the identity so that $y_k = a_k$. Similarly, for multiple binary classification problems, each output unit activation is transformed using a logistic sigmoid function so that
$$
y_k = \sigma(a_k)
\tag{5.5}
$$
where
$$
\sigma(a) = \frac{1}{1 + exp(-a)}
\tag{5.6}
$$
Finally, for multiclass problems, a softmax activation function of the form (4.62) is used. The choice of output unit activation function is discussed in detail in  [[Section 5.2]].

We can combine these various stages to give the overall network function that, for sigmoidal output unit activation functions, takes the form
$$
y_k(x,w) = \sigma(\sum^M_{j=1}w^{(2)}_{kj}h(\sum^D_{i=1}w^{(1)}_{ji}x_i + w^{(1)}_{j0}) + w^{(2)}_{k0})
\tag{5.7}
$$
where the set of all weight and bias parameters have been grouped together into a vector **w**. Thus the neural network model is simply a nonlinear function from a set of input variables $\{x_i\}$ to a set of output variables $\{y_k\}$ controlled by a vector w of adjustable parameters.

This function can be represented in the form of a network diagram as shown in Figure 5.1. The process of evaluating (5.7) can then be interpreted as a  [[forward propagation]] of information through the network. It should be emphasized that these diagrams do not represent probabilistic graphical models of the kind to be considered in [[Chapter 8]] because the internal nodes represent deterministic variables rather than stochastic ones. For this reason, we have adopted a slightly different graphical notation for the two kinds of model. We shall see later how to give a probabilistic interpretation to a neural network.

![[Figure 5.1.png]]
[[Figure 5.1.png|Figure 5.1]]

As discussed in [[Linear Basis Function Models|Section 3.1]], the bias parameters in (5.2) can be absorbed into the set of weight parameters by defining an additional input variable $x_0$ whose value is clamped at $x_0 = 1$, so that (5.2) takes the form
$$
a_j = \sum^D_{i=0}w^{(0)}_{ji}x_i
\tag{5.8}
$$
We can similarly absorb the second-layer biases into the second-layer weights, so that the overall network function becomes
$$
y_k(x,w) = \sigma(\sum^M_{j=0}w^{(2)}_{kj}h(\sum^D_{i=1}w^{(1)}_{ji}x_i))
\tag{5.9}
$$
As can be seen from [[Figure 5.1.png|Figure 5.1]], the neural network model comprises two stages of processing, each of which resembles the perceptron model of [[Section 4.1.7]], and for this reason the neural network is also known as the [[multilayer perceptron]], or [[MLP]]. A key difference compared to the perceptron, however, is that the neural network uses continuous sigmoidal nonlinearities in the hidden units, whereas the perceptron uses step-function nonlinearities. This means that the neural network function is differentiable with respect to the network parameters, and this property will play a central role in network training.

If the activation functions of all the hidden units in a network are taken to be linear, then for any such network we can always find an equivalent network without hidden units. This follows from the fact that the composition of successive linear transformations is itself a linear transformation. However, if the number of hidden units is smaller than either the number of input or output units, then the transformations that the network can generate are not the most general possible linear transformations from inputs to outputs because information is lost in the dimensionality reduction at the hidden units. In [[Section 12.4.2]], we show that networks of linear units give rise to principal component analysis. In general, however, there is little interest in multilayer networks of linear units.

The network architecture shown in [[Figure 5.1.png|Figure 5.1]] is the most commonly used one in practice. However, it is easily generalized, for instance by considering additional layers of processing each consisting of a weighted linear combination of the form (5.4) followed by an element-wise transformation using a nonlinear activation function. Note that there is some confusion in the literature regarding the terminology for counting the number of layers in such networks. Thus the network in Figure 5.1 may be described as a 3-layer network (which counts the number of layers of units, and treats the inputs as units) or sometimes as a single-hidden-layer network (which counts the number of layers of hidden units). We recommend a terminology in which [[Figure 5.1.png|Figure 5.1]] is called a two-layer network, because it is the number of layers of adaptive weights that is important for determining the network properties.

Another generalization of the network architecture is to include [[skip-layer connections]], each of which is associated with a corresponding adaptive parameter. For instance, in a two-layer network these would go directly from inputs to outputs. In principle, a network with sigmoidal hidden units can always mimic skip layer connections (for bounded input values) by using a sufficiently small first-layer weight that, over its operating range, the hidden unit is effectively linear, and then compensating with a large weight value from the hidden unit to the output. In practice, however, it may be advantageous to include skip-layer connections explicitly.

Furthermore, the network can be sparse, with not all possible connections within a layer being present. We shall see an example of a sparse network architecture when we consider convolutional neural networks in [[Section 5.5.6]].

Because there is a direct correspondence between a network diagram and its mathematical function, we can develop more general network mappings by considering more complex network diagrams. However, these must be restricted to a [[feed-forward]] architecture, in other words to one having no closed directed cycles, to ensure that the outputs are deterministic functions of the inputs. This is illustrated with a simple example in [[Figure 5.2.png|Figure 5.2]]. Each (hidden or output) unit in such a network computes a function given by
$$
z_k = h(\sum_{j}w_{kj}z_j)
\tag{5.10}
$$
where the sum runs over all units that send connections to unit k (and a bias parameter is included in the summation). For a given set of values applied to the inputs of the network, successive application of (5.10) allows the activations of all units in the network to be evaluated including those of the output units.

![[Figure 5.2.png]]
[[Figure 5.2.png|Figure 5.2]]

The approximation properties of feed-forward networks have been widely studied (Funahashi, 1989; Cybenko, 1989; Hornik et al., 1989; Stinchecombe and White, 1989; Cotter, 1990; Ito, 1991; Hornik, 1991; Kreinovich, 1991; Ripley, 1996) and found to be very general. Neural networks are therefore said to be [[universal approximators]]. For example, a two-layer network with linear outputs can uniformly approximate any continuous function on a compact input domain to arbitrary accuracy provided the network has a sufficiently large number of hidden units. This result holds for a wide range of hidden unit activation functions, but excluding polynomials. Although such theorems are reassuring, the key problem is how to find suitable parameter values given a set of training data, and in later sections of this chapter we will show that there exist effective solutions to this problem based on both maximum likelihood and Bayesian approaches.

The capability of a two-layer network to model a broad range of functions is illustrated in [[Figure 5.3.png|Figure 5.3]]. This figure also shows how individual hidden units work collaboratively to approximate the final function. The role of hidden units in a simple classification problem is illustrated in Figure 5.4 using the synthetic classification data set described in Appendix A.

![[Figure 5.3.png]]
[[Figure 5.3.png|Figure 5.3]]

![[Figure 5.4.png]]
[[Figure 5.4.png|Figure 5.4]]

## Index
- [[1. Weight-space symmetries]]