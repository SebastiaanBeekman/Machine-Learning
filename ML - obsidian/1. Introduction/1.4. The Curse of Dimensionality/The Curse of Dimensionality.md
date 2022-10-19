# The Curse of Dimensionality
In the polynomial curve fitting example we had just one input variable *x*. For practical applications of pattern recognition, however, we will have to deal with spaces of high dimensionality comprising many input variables. As we now discuss, this poses some serious challenges and is an important factor influencing the design of pattern recognition techniques.

In order to illustrate the problem we consider a synthetically generated data set representing measurements taken from a pipeline containing a mixture of oil, water, and gas (Bishop and James, 1993). These three materials can be present in one of three different geometrical configurations known as ‘homogenous’, ‘annular’, and ‘laminar’, and the fractions of the three materials can also vary. Each data point comprises a 12-dimensional input vector consisting of measurements taken with gamma ray densitometers that measure the attenuation of gamma rays passing along narrow beams through the pipe. [[Figure 1.19.png|Figure 1.19]] shows 100 points from this data set on a plot showing two of the measurements $x_6$ and $x_7$ (the remaining ten input values are ignored for the purposes of this illustration). Each data point is labelled according to which of the three geometrical classes it belongs to, and our goal is to use this data as a training set in order to be able to classify a new observation ($x_6$, $x_7$), such as the one denoted by the cross in Figure 1.19. We observe that the cross is surrounded by numerous red points, and so we might suppose that it belongs to the red class. However, there are also plenty of green points nearby, so we might think that it could instead belong to the green class. It seems unlikely that it belongs to the blue class. The intuition here is that the identity of the cross should be determined more strongly by nearby points from the training set and less strongly by more distant points. In fact, this intuition turns out to be reasonable and will be discussed more fully in later chapters.

![[Figure 1.19.png]]
[[Figure 1.19.png|Figure 1.19]]

How can we turn this intuition into a learning algorithm? One very simple approach would be to divide the input space into regular cells, as indicated in [[Figure 1.20.png|Figure 1.20]]. When we are given a test point and we wish to predict its class, we first decide which cell it belongs to, and we then find all of the training data points that fall in the same cell. The identity of the test point is predicted as being the same as the class having the largest number of training points in the same cell as the test point (with ties being broken at random). 

![[Figure 1.20.png]]
[[Figure 1.20.png|Figure 1.20]]

There are numerous problems with this naive approach, but one of the most severe becomes apparent when we consider its extension to problems having larger numbers of input variables, corresponding to input spaces of higher dimensionality. The origin of the problem is illustrated in [[Figure 1.21.png|Figure 1.21]], which shows that, if we divide a region of a space into regular cells, then the number of such cells grows exponentially with the dimensionality of the space. The problem with an exponentially large number of cells is that we would need an exponentially large quantity of training data in order to ensure that the cells are not empty. Clearly, we have no hope of applying such a technique in a space of more than a few variables, and so we need to find a more sophisticated approach.

![[Figure 1.21.png]]
[[Figure 1.21.png|Figure 1.21]]

We can gain further insight into the problems of high-dimensional spaces by returning to the example of polynomial curve fitting and considering how we would extend this approach to deal with input spaces having several variables. If we have *D* input variables, then a general polynomial with coefficients up to order 3 would take the form
$$
y(x,w) = w_0 + \sum^D_{i=1}w_ix_i + \sum^D_{i=1}\sum^D_{j=1}w_{ij}x_ix_j + \sum^D_{i=1}\sum^D_{j=1}\sum^D_{k=1}w_{ijk}x_ix_jx_k
\tag{1.74}
$$
As *D* increases, so the number of independent coefficients (not all of the coefficients are independent due to interchange symmetries amongst the *x* variables) grows proportionally to $D^3$. In practice, to capture complex dependencies in the data, we may need to use a higher-order polynomial. For a polynomial of order *M*, the growth in the number of coefficients is like $D^M$. Although this is now a power law growth, rather than an exponential growth, it still points to the method becoming rapidly unwieldy and of limited practical utility.

Our geometrical intuitions, formed through a life spent in a space of three dimensions, can fail badly when we consider spaces of higher dimensionality. As a simple example, consider a sphere of radius *r* = 1 in a space of *D* dimensions, and ask what is the fraction of the volume of the sphere that lies between radius 
*r* = 1− $\epsilon$ and *r* = 1. We can evaluate this fraction by noting that the volume of a sphere of radius *r* in *D* dimensions must scale as $r^D$, and so we write
$$
V_D(r) = K_Dr^D
\tag{1.75}
$$
where the constant $K_D$ depends only on D. Thus the required fraction is given by
$$
\frac{V_D(1) - V_D(1-\epsilon)}{V_D(1)} = 1 - (1 - \epsilon)^D
\tag{1.76}
$$
which is plotted as a function of $\epsilon$ for various values of *D* in [[Figure 1.22.png|Figure 1.22]]. We see that, for large *D*, this fraction tends to 1 even for small values of $\epsilon$. Thus, in spaces of high dimensionality, most of the volume of a sphere is concentrated in a thin shell near the surface!

![[Figure 1.22.png]]
[[Figure 1.22.png|Figure 1.22]]

As a further example, of direct relevance to pattern recognition, consider the behaviour of a Gaussian distribution in a high-dimensional space. If we transform from Cartesian to polar coordinates, and then integrate out the directional variables, we obtain an expression for the density *p(r)* as a function of radius *r* from the origin. Thus *p(r)$\delta$r* is the probability mass inside a thin shell of thickness *$\delta$r* located at radius *r*. This distribution is plotted, for various values of *D*, in 
[[Figure 1.23.png|Figure 1.23]], and we see that for large *D* the probability mass of the Gaussian is concentrated in a thin shell.

![[Figure 1.23.png]]
[[Figure 1.23.png|Figure 1.23]]

The severe difficulty that can arise in spaces of many dimensions is sometimes called the [[curse of dimensionality]] (Bellman, 1961). In this book, we shall make extensive use of illustrative examples involving input spaces of one or two dimensions, because this makes it particularly easy to illustrate the techniques graphically. The reader should be warned, however, that not all intuitions developed in spaces of low dimensionality will generalize to spaces of many dimensions.

Although the c[[urse of dimensionality]] certainly raises important issues for pattern recognition applications, it does not prevent us from finding effective techniques applicable to high-dimensional spaces. The reasons for this are twofold. First, real
data will often be confined to a region of the space having lower effective dimensionality, and in particular the directions over which important variations in the target variables occur may be so confined. Second, real data will typically exhibit some smoothness properties (at least locally) so that for the most part small changes in the input variables will produce small changes in the target variables, and so we can exploit local interpolation-like techniques to allow us to make predictions of the target variables for new values of the input variables. Successful pattern recognition techniques exploit one or both of these properties. Consider, for example, an application in manufacturing in which images are captured of identical planar objects on a conveyor belt, in which the goal is to determine their orientation. Each image is a point in a high-dimensional space whose dimensionality is determined by the number of pixels. Because the objects can occur at different positions within the image and in different orientations, there are three degrees of freedom of variability between images, and a set of images will live on a three dimensional [[manifold]] embedded within the high-dimensional space. Due to the complex relationships between the object position or orientation and the pixel intensities, this manifold will be highly nonlinear. If the goal is to learn a model that can take an input image and output the orientation of the object irrespective of its position, then there is only one degree of freedom of variability within the manifold that is significant.