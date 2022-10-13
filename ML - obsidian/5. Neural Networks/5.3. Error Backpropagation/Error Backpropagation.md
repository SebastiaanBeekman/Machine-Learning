# Error Backpropagation
Our goal in this section is to find an efficient technique for evaluating the gradient
of an error function $E(w)$ for a feed-forward neural network. We shall see that
this can be achieved using a local message passing scheme in which information is
sent alternately forwards and backwards through the network and is known as [[backprop|error backpropagation]], or sometimes simply as [[backprop]].

It should be noted that the term backpropagation is used in the neural computing
literature to mean a variety of different things. For instance, the multilayer
perceptron architecture is sometimes called a backpropagation network. The term
backpropagation is also used to describe the training of a multilayer perceptron using gradient descent applied to a sum-of-squares error function. In order to clarify the terminology, it is useful to consider the nature of the training process more carefully. Most training algorithms involve an iterative procedure for minimization of an error function, with adjustments to the weights being made in a sequence of steps. At each such step, we can distinguish between two distinct stages. In the first stage, the derivatives of the error function with respect to the weights must be evaluated. As we shall see, the important contribution of the backpropagation technique is in providing a computationally efficient method for evaluating such derivatives. Because it is at this stage that errors are propagated backwards through the network, we shall use the term backpropagation specifically to describe the evaluation of derivatives. In the second stage, the derivatives are then used to compute the adjustments to be made to the weights. The simplest such technique, and the one originally considered by Rumelhart et al. (1986), involves gradient descent. It is important to recognize that the two stages are distinct. Thus, the first stage, namely the propagation of errors backwards through the network in order to evaluate derivatives, can be applied to many other kinds of network and not just the multilayer perceptron. It can also be applied to error functions other that just the simple sum-of-squares, and to the evaluation of other derivatives such as the Jacobian and Hessian matrices, as we shall see later in this chapter. Similarly, the second stage of weight adjustment using the calculated derivatives can be tackled using a variety of optimization schemes, many of which are substantially more powerful than simple gradient descent.

## Index
- [[1. Evaluation of error-function derivatives]]
- [[2. A simple example]]
- [[3. Efficiency of backpropagation]]
- [[4. The Jacobian matrix]]