# Introduction
The problem of searching for patterns in data is a fundamental one and has a long and successful history. For instance, the extensive astronomical observations of Tycho Brahe in the 16th century allowed Johannes Kepler to discover the empirical laws of planetary motion, which in turn provided a springboard for the development of classical mechanics.

Consider the example of recognizing handwritten digits, illustrated in the figure below. Each digit corresponds to a 28×28 pixel image and so can be represented by a vector *x* comprising 784 real numbers. The goal is to build a machine that will take such a vector *x* as input and that will produce the identity of the digit as the output. This is a nontrivial problem due to the wide variability of handwriting. It could be tackled using handcrafted rules or heuristics for distinguishing the digits based on the shapes of the strokes, but in practice such an approach leads to a proliferation of rules and of exceptions to the rules and so on, and invariably gives poor results.

![[Pasted image 20220920150749.png]]

Far better results can be obtained by adopting a machine learning approach in which a large set of N digits {x1, . . . , xn} called a [[training set]] is used to tune the parameters of an adaptive model. The categories of the digits in the training set are known in advance, typically by inspecting them individually and hand-labelling them. We can express the category of a digit using [[target vector]] t, which represents the identity of the corresponding digit.

The result of running the machine learning algorithm can be expressed as a function y(x) which takes a new digit image *x* as input and that generates an output vector *y*, encoded in the same way as the [[target vector|target vectors]]. The precise form of the function y(x) is determined during the [[training phase]] on the basis of the training data. Once the model is trained it can then determine the identity of new digit images, which are said to comprise a [[test set]]. The ability to categorize correctly new examples that differ from those used for training is known as [[generalization]].

For most practical applications, the original input variables are typically [[preprocessed]] so that the pattern recognition problem will be easier to solve. For example, in the digit recognition problem, the images of the digits are typically translated and scaled so that each digit is contained within a box of a fixed size. This greatly reduces the variability within each digit class, because the location and scale of all the digits are now the same, which makes it much easier for a subsequent pattern recognition algorithm to distinguish between the different classes. This pre-processing stage is sometimes also called [[feature extraction]].

[[Pre-processing to speed up computation|Pre-processing might also be performed in order to speed up computation]]. 

Applications in which the training data comprises examples of the input vectors along with their corresponding target vectors are known as [[supervised learning]] problems.  Cases such as the digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories, are called [[classification]] problems. If the desired output consists of one or more continuous variables, then the task is called [[regression]]. 

In other pattern recognition problems, the training data consists of a set of input vectors x without any corresponding target values. The goal in such [[unsupervised learning]] problems may be to discover groups of similar examples within the data, where it is called [[clustering]], or to determine the distribution of data within the input space, known as [[density estimation]], or to project the data from a high-dimensional space down to two or three dimensions for the purpose of **visualization**.

Finally, the technique of [[reinforcement learning]] (Sutton and Barto, 1998) is concerned with the problem of finding suitable actions to take in a given situation in order to maximize a reward. A general feature of reinforcement learning is the trade-off between **exploration**, in which the system tries out new kinds of actions to see how effective they are, and **exploitation**, in which the system makes use of actions that are known to yield a high reward. Too strong a focus on either exploration or exploitation will yield poor results.

## Index
- [[Example - Polynomial Curve Fitting]]
- [[Probability Theory]]
- [[Model Selection]]
- [[The Curse of Dimensionality]]
- [[Decision Theory]]
- [[Information Theory]]