# Sparse Kernel Machines
In the previous chapter, we explored a variety of learning algorithms based on nonlinear kernels. One of the significant limitations of many such algorithms is that
the kernel function $k(x_n, x_m)$ must be evaluated for all possible pairs $x_n$ and $x_m$
of training points, which can be computationally infeasible during training and can
lead to excessive computation times when making predictions for new data points.
In this chapter we shall look at kernel-based algorithms that have [[sparse]] solutions,
so that predictions for new inputs depend only on the kernel function evaluated at a subset of the training data points.

We begin by looking in some detail at the [[support vector machine]] (SVM), which
became popular in some years ago for solving problems in classification, regression,
and novelty detection. An important property of support vector machines is that the
determination of the model parameters corresponds to a convex optimization problem, and so any local solution is also a global optimum. Because the discussion of support vector machines makes extensive use of Lagrange multipliers, the reader is encouraged to review the key concepts covered in Appendix E. Additional information on support vector machines can be found in Vapnik (1995), Burges (1998), Cristianini and Shawe-Taylor (2000), M¨uller et al. (2001), Sch¨olkopf and Smola (2002), and Herbrich (2002).

The SVM is a decision machine and so does not provide posterior probabilities.
We have already discussed some of the benefits of determining probabilities in Section 1.5.4. An alternative sparse kernel technique, known as the [[relevance vector machine]] (RVM), is based on a Bayesian formulation and provides posterior probabilistic outputs, as well as having typically much sparser solutions than the SVM.

## Index
- [[Maximum Margin Classifiers]]
- [[Relevance Vector Machines]]