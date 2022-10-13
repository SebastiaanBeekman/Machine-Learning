# Feed-forward Network Functions
The linear models for regression and classification discussed in Chapters 3 and 4, respectively, are based on linear combinations of fixed nonlinear basis functions φj(x)
and take the form
$$
y(x,w) = f(\sum^M_{j=1}w_j\phi_j(x))
\tag{5.1}
$$
where f(·) is a nonlinear activation function in the case of classification and is the identity in the case of regression. Our goal is to extend this model by making the
basis functions φj(x) depend on parameters and then to allow these parameters to
be adjusted, along with the coefficients {wj}, during training. There are, of course,
many ways to construct parametric nonlinear basis functions. Neural networks use
basis functions that follow the same form as (5.1), so that each basis function is itself a nonlinear function of a linear combination of the inputs, where the coefficients in the linear combination are adaptive parameters.

This leads to the basic neural network model, which can be described a series
of functional transformations. First we constructM linear combinations of the input
variables x1, . . . , xD in the form
$$
a_j = \sum^D_{i=1}w^{(1)}_{ji}x_i+w^{(1)}_{j0}
\tag{5.2}
$$
where j = 1, . . . , M, and the superscript (1) indicates that the corresponding parameters are in the first ‘layer’ of the network. We shall refer to the parameters w(1) ji as weights and the parameters w(1) j0 as biases, following the nomenclature of Chapter 3. The quantities aj are known as activations. Each of them is then transformed using a differentiable, nonlinear activation function h(·) to give
$$
z_j = h(a_j)
\tag{5.3}
$$
These quantities correspond to the outputs of the basis functions in (5.1) that, in the
context of neural networks, are called hidden units. The nonlinear functions h(·) are
generally chosen to be sigmoidal functions such as the logistic sigmoid or the ‘tanh’ function. Following (5.1), these values are again linearly combined to give output
unit activations
$$
a_k = \sum^M_{j=1}w^{(2)}_{kj}z_j+w^{(2)}_{k0}
\tag{5.4}
$$
where k = 1, . . . , K, and K is the total number of outputs. This transformation corresponds to the second layer of the network, and again the w(2) k0 are bias parameters. Finally, the output unit activations are transformed using an appropriate activation function to give a set of network outputs yk. The choice of activation function is determined by the nature of the data and the assumed distribution of target variables and follows the same considerations as for linear models discussed in Chapters 3 and 4. Thus for standard regression problems, the activation function is the identity so that yk = ak. Similarly, for multiple binary classification problems, each output unit activation is transformed using a logistic sigmoid function so that
$$

$$

