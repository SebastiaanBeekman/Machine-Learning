# Linear Basis Function Models

The simplest linear model for regression is one that involves a linear combination of
the input variables
$$
y(x,w) = w_0 + w_1x_1 + ... + w_Dx_D
\tag{3.1}
$$
where $x = (x_1, ..., x_D)^T$. This is often simply known as linear regression. The key property of this model is that it is a linear function of the parameters $w_0, ..., w_D$. It is also, however, a linear function of the input variables $x_i$, and this imposes significant limitations on the model. We therefore extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form
$$
y(x,w) = w_0 =\sum^{m-1}_{j=1}w_j\phi_j(x)
\tag{3.2}
$$
where $\phi_j(x)$ are known as *basis functions*. By denoting the maximum value of the index *j* by *M* − 1, the total number of parameters in this model will be *M*.

