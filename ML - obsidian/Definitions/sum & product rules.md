# Sum rule
In order to derive the rules of probability, consider the slightly more general example shown in [[Figure 1.10.png|Figure 1.10]] involving two random variables *X* and *Y*. We shall suppose that *X* can take any of the values $x_i$ where *i = 1, ..., M*, and *Y* can take the values $y_j$ where *j = 1, ...,L.* Consider a total of *N* trials in which we sample both of the variables *X* and *Y*, and let the number of such trials in which $X = x_i$ and $Y = y_j$ be $n_{ij}$. Also, let the number of trials in which *X* takes the value $x_i$ (irrespective of the value that Y takes) be denoted by $c_i$, and similarly let the number of trials in which *Y* takes the value $y_j$ be denoted by $r_j$ .

![[Figure 1.10.png]]
[[Figure 1.10.png|Figure 1.10]]

The probability that *X* will take the value $x_i$ and Y will take the value $y_j$ is written p(X = $x_i$, Y = $y_j$) and is called the **joint** probability of X = $x_i$ and Y = $y_j$ . It is given by the number of points falling in the cell *i,j* as a fraction of the total number of points, and hence
$$
p(X = x_i, Y = y_i) = \frac{n_{ij}}{N}
\tag{1.5}
$$
Here we are implicitly considering the limit N $\rightarrow \infty$. Similarly, the probability that *X* takes the value $x_i$ irrespective of the value of *Y* is written as *p(X = $x_i$)* and is given by the fraction of the total number of points that fall in column *i*, so that
$$
p(X = x_i) = \frac{c_i}{N}
\tag{1.6}
$$
Because the number of instances in column *i* in [[Figure 1.10.png|Figure 1.10]] is just the sum of the number of instances in each cell of that column, we have $c_i = \sum_j{n_{ij}}$ and therefore, from (1.5) and (1.6), we have
$$
p(X = x_i) = \sum^L_{j = 1}p(X = x_i, Y=y_j)
\tag{1.7}
$$
which is the **sum rule** of probability. Note that *p(X = $x_i$)* is sometimes called the [[marginal probability]], because it is obtained by marginalizing, or summing out, the other variables (in this case *Y*).

If we consider only those instances for which *X* = $x_i$, then the fraction of such instances for which *Y* = $y_j$ is written *p(Y = $y_j$ |X = $x_i$)* and is called the [[conditional probability]] of *Y* = $y_j$ given X = $x_i$. It is obtained by finding the fraction of those points in column *i* that fall in cell *i,j* and hence is given by
$$
p(Y = y_j|X = x_i) = \frac{n_{ij}}{c_i}
\tag{1.8}
$$
From (1.5), (1.6), and (1.8), we can then derive the following relationship
$$
\begin{align}  
p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} = \frac{n_{ij}}{c_i} * \frac{c_i}{N} \\
= p(Y = y_j|X = x_i)p(X = x_i)
\end{align}
\tag{1.9}
$$
which is the **product rule** of probability.

With this we can write the two fundamental rules of probability theory in the following form.

### The Rules of Probability
**Sum rule**
$$
p(X) = \sum_Yp(X,Y)
\tag{1.10}
$$
**Product rule**
$$
p(X,Y) = p(Y|X)p(X)
\tag{1.11}
$$
Here *p(X,Y)* is a joint probability and is verbalized as “the probability of *X* and *Y* ”. Similarly, the quantity *p(Y |X)* is a conditional probability and is verbalized as “the probability of *Y* given *X*”, whereas the quantity *p(X)* is a marginal probability and is simply “the probability of *X*”. These two simple rules form the basis for all of the probabilistic machinery that we use throughout this book.

From the product rule, together with the symmetry property p(X,Y) = p(Y,X), we immediately obtain the following relationship between conditional probabilities
$$
p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}
\tag{1.12}
$$
which is called **Bayes’ theorem** and which plays a central role in pattern recognition and machine learning. Using the sum rule, the denominator in Bayes’ theorem can be expressed in terms of the quantities appearing in the numerator
$$
p(X) = \sum_Yp(X|Y)p(Y)
\tag{1.13}
$$
We can view the denominator in Bayes’ theorem as being the normalization constant required to ensure that the sum of the conditional probability on the left-hand side of (1.12) over all values of Y equals one.

In [[Figure 1.11.png|Figure 1.11]], we show a simple example involving a joint distribution over two variables to illustrate the concept of marginal and conditional distributions. Here a finite sample of N = 60 data points has been drawn from the joint distribution and is shown in the top left. In the top right is a histogram of the fractions of data points having each of the two values of Y . From the definition of probability, these fractions would equal the corresponding probabilities p(Y ) in the limit N $\rightarrow \infty$. We can view the histogram as a simple way to model a probability distribution given only a finite number of points drawn from that distribution. Modelling distributions from data lies at the heart of statistical pattern recognition and will be explored in great detail in this book. The remaining two plots in [[Figure 1.11.png|Figure 1.11]] show the corresponding histogram estimates of p(X) and p(X|Y = 1).

![[Figure 1.11.png]]
[[Figure 1.11.png|Figure 1.11]]