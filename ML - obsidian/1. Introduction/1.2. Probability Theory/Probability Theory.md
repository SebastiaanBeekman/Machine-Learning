# Probability Theory
A key concept in the field of pattern recognition is that of uncertainty. It arises both through noise on measurements, as well as through the finite size of data sets. Probability theory provides a consistent framework for the quantification and manipulation of uncertainty and forms one of the central foundations for pattern recognition. When combined with decision theory, discussed in Section 1.5, it allows us to make optimal predictions given all the information available to us, even though that information
may be incomplete or ambiguous.

We will introduce the basic concepts of probability theory by considering a simple example. Imagine we have two boxes, one red and one blue, and in the red box we have 2 apples and 6 oranges, and in the blue box we have 3 apples and 1 orange. This is illustrated in [[Figure 1.9.png|Figure 1.9]]. Now suppose we randomly pick one of the boxes and from that box we randomly select an item of fruit, and having observed which sort of fruit it is we replace it in the box from which it came. We could imagine repeating this process many times. Let us suppose that in so doing we pick the red box 40% of the time and we pick the blue box 60% of the time, and that when we remove an item of fruit from a box we are equally likely to select any of the pieces of fruit in the box.

![[Figure 1.9.png]]
[[Figure 1.9.png|Figure 1.9]]

In this example, the identity of the box that will be chosen is a random variable, which we shall denote by *B*. This random variable can take one of two possible values, namely *r* (corresponding to the red box) or *b* (corresponding to the blue box). Similarly, the identity of the fruit is also a random variable and will be denoted by *F*. It can take either of the values *a* (for apple) or *o* (for orange). To begin with, we shall define the probability of an event to be the fraction of times that event occurs out of the total number of trials, in the limit that the total number of trials goes to infinity. Thus the probability of selecting the red box is 4/10 and the probability of selecting the blue box is 6/10. We write these probabilities as *p(B = r)* = 4/10 and *p(B = b)* = 6/10. Note that, by definition, probabilities
must lie in the interval [0, 1]. Also, if the events are mutually exclusive and if they include all possible outcomes (for instance, in this example the box must be either red or blue), then we see that the probabilities for those events must sum to one.

We can now ask questions such as: “what is the overall probability that the selection procedure will pick an apple?”, or “given that we have chosen an orange, what is the probability that the box we chose was the blue one?”. We can answer questions such as these, and indeed much more complex questions associated with problems in pattern recognition, once we have equipped ourselves with the two elementary rules of probability, known as the [[sum rule]] and the [[product rule]]. Having obtained these rules, we shall then return to our boxes of fruit example.

In order to derive the rules of probability, consider the slightly more general example shown in [[Figure 1.10.png|Figure 1.10]] involving two random variables *X* and *Y*. We shall suppose that *X* can take any of the values $x_i$ where *i = 1, . . . , M*, and *Y* can take the values $y_j$ where *j = 1, . . . ,L.* Consider a total of *N* trials in which we sample both of the variables *X* and *Y*, and let the number of such trials in which $X = x_i$ and $Y = y_j$ be $n_{ij}$ . Also, let the number of trials in which *X* takes the value $x_i$ (irrespective of the value that Y takes) be denoted by $c_i$, and similarly let the number of trials in which *Y* takes the value $y_j$ be denoted by $r_j$ .

![[Figure 1.10.png]]
[[Figure 1.10.png|Figure 1.10]]

The probability that *X* will take the value xi and Y will take the value yj is written p(X = xi, Y = yj) and is called the joint probability of X = xi and Y = yj . It is given by the number of points falling in the cell i,j as a fraction of the total number of points, and hence
$$p(X = x_i, Y = y_i) = \frac{n_{ij}}{N}$$
Here we are implicitly considering the limit N →∞. Similarly, the probability that X takes the value xi irrespective of the value of Y is written as p(X = xi) and is given by the fraction of the total number of points that fall in column i, so that
$$p(X = x_i) = \frac{c_i}{N}$$
Because the number of instances in column i in Figure 1.10 is just the sum of the number of instances in each cell of that column, we have $c_i = \sum_j{n_{ij}}$ and therefore, from (1.5) and (1.6), we have
$$p(X = x_i) = \sum^L_{j = 1}p(X = x_i, Y=y_i)$$
which is the sum rule of probability. Note that p(X = xi) is sometimes called the marginal probability, because it is obtained by marginalizing, or summing out, the other variables (in this case Y ).

If we consider only those instances for which X = xi, then the fraction of such instances for which Y = yj is written p(Y = yj |X = xi) and is called the conditional probability of Y = yj given X = xi. It is obtained by finding the fraction of those points in column i that fall in cell i,j and hence is given by
$$p(Y = y_j|X = x_i) = \frac{n_{ij}}{c_i}$$
From (1.5), (1.6), and (1.8), we can then derive the following relationship
$$
\begin{align}  
p(X = x_i, Y = y_j) = \frac{n_{ij}}{N} = \frac{n_{ij}}{c_i} * \frac{c_i}{N} \\
= p(Y = y_j|X = x_i)p(X = x_i)
\end{align}
$$
which is the product rule of probability.

So far we have been quite careful to make a distinction between a random variable, such as the box B in the fruit example, and the values that the random variable can take, for example r if the box were the red one. Thus the probability that B takes the value r is denoted p(B = r). Although this helps to avoid ambiguity, it leads to a rather cumbersome notation, and in many cases there will be no need for such pedantry. Instead, we may simply write p(B) to denote a distribution over the random variable B, or p(r) to denote the distribution evaluated for the particular value r, provided that the interpretation is clear from the context.

With this more compact notation, we can write the two fundamental rules of probability theory in the following form.

### The Rules of Probability
**Sum rule**
$p(X) = \sum_Yp(X,Y)$

**Product rule**
$p(X,Y) = p(Y|X)p(X)$

Here p(X, Y ) is a joint probability and is verbalized as “the probability of X and Y ”. Similarly, the quantity p(Y |X) is a conditional probability and is verbalized as “the probability of Y given X”, whereas the quantity p(X) is a marginal probability and is simply “the probability of X”. These two simple rules form the basis for all of the probabilistic machinery that we use throughout this book.

From the product rule, together with the symmetry property p(X, Y ) = p(Y,X), we immediately obtain the following relationship between conditional probabilities
$$p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$$
which is called Bayes’ theorem and which plays a central role in pattern recognition and machine learning. Using the sum rule, the denominator in Bayes’ theorem can be expressed in terms of the quantities appearing in the numerator
$$p(X) = \sum_Yp(X|Y)p(Y)$$
We can view the denominator in Bayes’ theorem as being the normalization constant required to ensure that the sum of the conditional probability on the left-hand side of (1.12) over all values of Y equals one.

In [[Figure 1.11.png|Figure 1.11]], we show a simple example involving a joint distribution over two variables to illustrate the concept of marginal and conditional distributions. Here a finite sample of N = 60 data points has been drawn from the joint distribution and is shown in the top left. In the top right is a histogram of the fractions of data points having each of the two values of Y . From the definition of probability, these fractions would equal the corresponding probabilities p(Y ) in the limit N →∞. We can view the histogram as a simple way to model a probability distribution given only a finite number of points drawn from that distribution. Modelling distributions from data lies at the heart of statistical pattern recognition and will be explored in great detail in this book. The remaining two plots in [[Figure 1.11.png|Figure 1.11]] show the corresponding histogram estimates of p(X) and p(X|Y = 1).

![[Figure 1.11.png]]
[[Figure 1.11.png|Figure 1.11]]

Let us now return to our example involving boxes of fruit. For the moment, we shall once again be explicit about distinguishing between the random variables and their instantiations. We have seen that the probabilities of selecting either the red or the blue boxes are given by
$$
\begin{align}
p(B = r) = 4/10 \\
p(B = b) = 6/10
\end{align}
$$
respectively. Note that these satisfy p(B = r) + p(B = b) = 1.

Now suppose that we pick a box at random, and it turns out to be the blue box. Then the probability of selecting an apple is just the fraction of apples in the blue box which is 3/4, and so *p(F = a|B = b)* = 3/4. In fact, we can write out all four conditional probabilities for the type of fruit, given the selected box
$$
\begin{align}
p(F = a|B = r) = 1/4 \\
p(F = o|B = r) = 3/4 \\
p(F = a|B = b) = 3/4 \\
p(F = o|B = b) = 1/4
\end{align}
$$
Again, note that these probabilities are normalized so that 
$$p(F = a|B = r) + p(F = o|B = r) = 1$$
and simiarly
$$p(F = a|B = b) + p(F = o|B = b) = 1$$
We can now use the sum and product rules of probability to evaluate the overall probability of choosing an apple
$$
\begin{align}
p(F = a) = p(F = a|B = r)p(B = r) + p(F = a|B = b)p(B = b) \\
= \frac{1}{4} * \frac{4}{10} + \frac{3}{4} * \frac{6}{10} = \frac{11}{20} 
\end{align}
$$
from which it follows, using the sum rule, that *p(F = o)* = 1 − 11/20 = 9/20.

Suppose instead we are told that a piece of fruit has been selected and it is an orange, and we would like to know which box it came from. This requires that we evaluate the probability distribution over boxes conditioned on the identity of the fruit, whereas the probabilities in (1.16)–(1.19) give the probability distribution over the fruit conditioned on the identity of the box. We can solve the problem of reversing the conditional probability by using Bayes’ theorem to give
$$p(B = r|F = o) = \frac{p(F = o|B = r)p(B = r)}{p(F = o)} = \frac{3}{4} * \frac{4}{10} * \frac{20}{9} = \frac{2}{3}$$
From the sum rule, it then follows that *p(B = b|F = o)* = 1 − 2/3 = 1/3.

We can provide an important interpretation of Bayes’ theorem as follows. If we had been asked which box had been chosen before being told the identity of the selected item of fruit, then the most complete information we have available is provided by the probability *p(B)*. We call this the [[prior probability]] because it is the probability available *before* we observe the identity of the fruit. Once we are told that the fruit is an orange, we can then use Bayes’ theorem to compute the probability *p(B|F)*, which we shall call the [[posterior probability]] because it is the probability obtained after we have observed *F*. Note that in this example, the prior probability of selecting the red box was 4/10, so that we were more likely to select the blue box than the red one. However, once we have observed that the piece of selected fruit is an orange, we find that the *posterior probability* of the red box is now 2/3, so that it is now more likely that the box we selected was in fact the red one. This result accords with our intuition, as the proportion of oranges is much higher in the red box than it is in the blue box, and so the observation that the fruit was an orange provides
significant evidence favouring the red box. In fact, the evidence is sufficiently strong that it outweighs the prior and makes it more likely that the red box was chosen rather than the blue one.

Finally, we note that if the joint distribution of two variables factorizes into the product of the marginals, so that *p(X, Y) = p(X)p(Y)*, then *X* and *Y* are said to be independent. From the product rule, we see that *p(Y |X) = p(Y)*, and so the conditional distribution of *Y* given *X* is indeed independent of the value of *X*. For instance, in our boxes of fruit example, if each box contained the same fraction of apples and oranges, then *p(F|B) = P(F)*, so that the probability of selecting, say, an apple is independent of which box is chosen.

## Index
- [[1. Probability densities]]
- [[2. Expectations and covariances]]