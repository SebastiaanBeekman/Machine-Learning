A key concept in the field of pattern recognition is that of uncertainty. It arises both through noise on measurements, as well as through the finite size of data sets. Probability theory provides a consistent framework for the quantification and manipulation of uncertainty and forms one of the central foundations for pattern recognition. When combined with decision theory, discussed in [[Decision Theory|Section 1.5]], it allows us to make optimal predictions given all the information available to us, even though that information may be incomplete or ambiguous.

We will introduce the basic concepts of probability theory by considering a simple example. Imagine we have two boxes, one red and one blue, and in the red box we have 2 apples and 6 oranges, and in the blue box we have 3 apples and 1 orange. This is illustrated in [[Figure 1.9.png|Figure 1.9]]. Now suppose we randomly pick one of the boxes and from that box we randomly select an item of fruit, and having observed which sort of fruit it is we replace it in the box from which it came. We could imagine repeating this process many times. Let us suppose that in so doing we pick the red box 40% of the time and we pick the blue box 60% of the time, and that when we remove an item of fruit from a box we are equally likely to select any of the pieces of fruit in the box.

![[Figure 1.9.png]]
[[Figure 1.9.png|Figure 1.9]]

In this example, the identity of the box that will be chosen is a random variable, which we shall denote by *B*. This random variable can take one of two possible values, namely *r* (corresponding to the red box) or *b* (corresponding to the blue box). Similarly, the identity of the fruit is also a random variable and will be denoted by *F*. It can take either of the values *a* (for apple) or *o* (for orange). To begin with, we shall define the probability of an event to be the fraction of times that event occurs out of the total number of trials, in the limit that the total number of trials goes to infinity. Thus the probability of selecting the red box is 4/10 and the probability of selecting the blue box is 6/10. We write these probabilities as *p(B = r)* = 4/10 and *p(B = b)* = 6/10. Note that, by definition, probabilities must lie in the interval $[0, 1]$. Also, if the events are mutually exclusive and if they include all possible outcomes (for instance, in this example the box must be either red or blue), then we see that the probabilities for those events must sum to one.

We can now ask questions such as: “what is the overall probability that the selection procedure will pick an apple?”, or “given that we have chosen an orange, what is the probability that the box we chose was the blue one?”. We can answer questions such as these, and indeed much more complex questions associated with problems in pattern recognition, once we have equipped ourselves with the two elementary rules of probability, known as the [[sum & product rules]].

Let us now return to our example involving boxes of fruit. The probabilities of selecting either the red or the blue boxes are given by
$$
p(B = b) = 6/10
\tag{1.14}
$$
$$
p(B = r) = 4/10
\tag{1.15}
$$
respectively. *(Note that these satisfy p(B = r) + p(B = b) = 1)*

Now suppose that we pick a box at random, and it turns out to be the blue box. Then the probability of selecting an apple is just the fraction of apples in the blue box which is 3/4, and so *p(F = a|B = b)* = 3/4. In fact, we can write out all four conditional probabilities for the type of fruit, given the selected box
$$
p(F = a|B = r) = 1/4
\tag{1.16}
$$
$$
p(F = o|B = r) = 3/4
\tag{1.17}
$$
$$
p(F = a|B = b) = 3/4
\tag{1.18}
$$
$$
p(F = o|B = b) = 1/4
\tag{1.19}
$$
Again, note that these probabilities are normalized so that 
$$
p(F = a|B = r) + p(F = o|B = r) = 1
\tag{1.20}
$$
and simiarly
$$
p(F = a|B = b) + p(F = o|B = b) = 1
\tag{1.21}
$$
We can now use the sum and product rules of probability to evaluate the overall probability of choosing an apple
$$
\begin{align}
p(F = a) = p(F = a|B = r)p(B = r) + p(F = a|B = b)p(B = b) \\
= \frac{1}{4} * \frac{4}{10} + \frac{3}{4} * \frac{6}{10} = \frac{11}{20} 
\end{align}
\tag{1.22}
$$
from which it follows, using the sum rule, that *p(F = o)* = 1 − 11/20 = 9/20.

Suppose instead we are told that a piece of fruit has been selected and it is an orange, and we would like to know which box it came from. This requires that we evaluate the probability distribution over boxes conditioned on the identity of the fruit, whereas the probabilities in (1.16) - (1.19) give the probability distribution over the fruit conditioned on the identity of the box. We can solve the problem of reversing the conditional probability by using Bayes’ theorem to give
$$p(B = r|F = o) = \frac{p(F = o|B = r)p(B = r)}{p(F = o)} = \frac{3}{4} * \frac{4}{10} * \frac{20}{9} = \frac{2}{3}$$
From the sum rule, it then follows that *p(B = b|F = o)* = 1 − 2/3 = 1/3.

We can provide an important interpretation of Bayes’ theorem as follows. If we had been asked which box had been chosen before being told the identity of the selected item of fruit, then the most complete information we have available is provided by the probability *p(B)*. We call this the [[prior probability]] because it is the probability available *before* we observe the identity of the fruit. Once we are told that the fruit is an orange, we can then use Bayes’ theorem to compute the probability *p(B|F)*, which we shall call the [[posterior probability]] because it is the probability obtained after we have observed *F*. Note that in this example, the **prior probability** of selecting the red box was 4/10, so that we were more likely to select the blue box than the red one. However, once we have observed that the piece of selected fruit is an orange, we find that the **posterior probability** of the red box is now 2/3, so that it is now more likely that the box we selected was in fact the red one. This result accords with our intuition, as the proportion of oranges is much higher in the red box than it is in the blue box, and so the observation that the fruit was an orange provides significant evidence favouring the red box. In fact, the evidence is sufficiently strong that it outweighs the prior and makes it more likely that the red box was chosen rather than the blue one.

Finally, we note that if the joint distribution of two variables factorizes into the product of the marginals, so that *p(X, Y) = p(X)p(Y)*, then *X* and *Y* are said to be independent. From the product rule, we see that *p(Y |X) = p(Y)*, and so the conditional distribution of *Y* given *X* is indeed independent of the value of *X*. For instance, in our boxes of fruit example, if each box contained the same fraction of apples and oranges, then *p(F|B) = P(F)*, so that the probability of selecting, say, an apple is independent of which box is chosen.

## Index
- [[1. Probability densities]]
- [[2. Expectations and covariances]]
- [[3. Bayesian probabilities]]