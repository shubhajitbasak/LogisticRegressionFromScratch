The goal of this project is to build a machine learning algorithm for
classifier problem. The Classifier should take any number of features
i.e. multivariate dataset and will be predict the classes of multiclass
labels. Here I have chosen to implement the Logistic Regression
Classification Algorithm as it is easy to implement, and we can extend
the concept of Linear Regression to build the same. To minimise the cost
function, I am using classic Gradient Descent Algorithm.

**Details of the Algorithm**:

<em>Binary Classification Implementation:</em>

We will first try to build a binary classifier where there will be two
types of dependent variable. As the Dependent variables are Binary in
nature the linear regression model will not fit properly, so we need a
model that can classify the binary data with the output probability.
Logistic Regression uses the Logistic Function or the Sigmoid Function
to fit the data and predict the probability. A sample has been shown in
the following figure where the sigmoid function refers to the
Probability of passing the Exam and it fits perfectly with the two type
of binary result (0,1). To differentiate the data, we can set a
threshold above which we will predict 1 and below that we predict 1,
here the threshold might be 0.5.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

Figure 1 : Sample Sigmoid Function (En.wikipedia.org, 2018)

Mathematically the sigmoid function looks like :

![](https://latex.codecogs.com/gif.latex?g%28z%29%3D%5Cfrac%7B1%7D%7B1&plus;%5C%20%7Be%7D%5E%7B-z%7D%5C%20%7D)

We know for Linear Regression the hypothesis Function for multivariate
data looks like :

![](https://latex.codecogs.com/gif.latex?h_%5Ctheta%5C%20%28x%29%3D%5C%20%5Ctheta_0%5C%20x_0&plus;%5C%20%5Ctheta_1%5C%20x_1&plus;%5C%20%5Ctheta_1%5C%20x_1&plus;%5Cldots%5Cldots%5Cldots%5Cldots%5Cldots%5Cldots&plus;%5C%20%5Ctheta_n%5C%20x_n)

   ,where ![](https://latex.codecogs.com/gif.latex?x_0) is the bias term set to 1.

Here for mathematical simplicity we will use the Vectorised (Matrix )
form.

So, if there are n variables and m rows of data   
![](https://latex.codecogs.com/gif.latex?X%3D%20%5Cbegin%7Bbmatrix%7D%20x%5E%7B1%7D_0%20%26%20...%20%26%20x%5E%7B1%7D_n%5C%5C%20.%26%20.%20%26%20.%5C%5C%20x%5E%7Bm%7D_0%20%26%20%26%20x%5E%7Bm%7D_n%20%5Cend%7Bbmatrix%7D)    
And    
![](https://latex.codecogs.com/gif.latex?%5Ctheta%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Ctheta_0%5C%5C%20.%5C%5C%20.%5C%5C%20%5Ctheta_n%20%5Cend%7Bbmatrix%7D)

The hypothesis function in Vectorised form will looks like
![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%28X%29%20%3D%20X%20%5Ctheta)

Now we will extend the Hypothesis Function for Logistic Regression using
the Sigmoid Function as --

![](https://latex.codecogs.com/gif.latex?H_%5Ctheta%5C%20%28X%29%3D%20%5Cfrac%7B1%7D%7B1&plus;%5C%20%7Be%7D%5E%7B-X%5Ctheta%7D%5C%20%7D%20%3D%20G%28X%5Ctheta%29)

**Note**. -- Here we have taken the [Theta as a Column Matrix]{.underline}

Now to build the classifier we will try to minimise the error or
difference between the actual Label Value (Dependent Variable) Y and the
result from the hypothesis function, which is known as the Cost
Function.

To get the minima we will take the First Order Partial Derivative and
get the Gradient Function which in Vectorised form will looks like --

Gradient =
![](https://latex.codecogs.com/gif.latex?%28H_%5Ctheta%5C%20%28X%29%20-%20Y%29X%20%3D%20X%5E%7BT%7D%28H_%5Ctheta%5C%20%28X%29%20-%20Y%29)

Then we will try to get the Optimum Theta with Gradient Descent which
looks like --

> Repeat {   
![](https://latex.codecogs.com/gif.latex?%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%20%5Cfrac%7B%5Calpha%20%7D%7Bm%7DX%5E%7BT%7D%28H_%7B%5Ctheta%7D%28X%29-Y%29)   
> }

,where ![](https://latex.codecogs.com/gif.latex?%5Calpha) is the learning rate and m is the total number of
training data.

After several Iteration we will get the ![](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bopt%7D) which will give
the best fit and will minimise the error between the hypothesis output
and the actual Y value.

So, the output of our Model will look like --

![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta_%7Bopt%7D%7D%20%3D%20X%5Ctheta_%7Bopt%7D)

Which gives the probability for the independent variables, as it's a
binary classifier we will classify using some threshold and compare the
probability with the threshold.

**Multi Class (One Vs All) Implementation:**

As I am implementing algorithm in vectorised form it will work for
multiple variable as well.

Till now we have to build our model for Binary Dependent Variables but
for multiclass classification. For this we will use the One Vs All
Algorithm. We will treat each class out as a binary i.e. for that class
only the output is 1 and all others are zero and we will push the data
through the model to get the individual probability of each class. Then
for a new input we will assign the class which have the highest
probability.

Mathematically -

![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%5E%7Bi%7D%28x%29%20%3D%20P%28y%3Di%7Cx%2C%5Ctheta%29)

Here P is the Probability for each class i. Here we will train the
logistic classifier ![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%5E%7Bi%7D%28x%29) for each class i to predict
the probability that y = i . For a new input x to make the prediction we
will pick the class i that maximises ![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%5E%7Bi%7D%28x%29)

**Design Decision and Specifications :**

Here I am implementing the Algorithm using Python Language. As I am
implementing the classifier in vectorised form the classifier expects
the input data in a matrix Form. Following are the Methods and their
respective parameters.

*[Class :]{.underline}* LogisticRegressor**(**alpha **=** 0.5**,**
iterations **=** 15000**,** tol**=**1e-8**)**

*[Parameters:]{.underline}*

alpha : learning rate

iterations : Maximum Number of Iterations for the Gradient Descent
Iterations

tol : maximum tolerance level for the cost function

[Methods:]{.underline}

Build the Model with the Training Data

fit (X,Y) --

Parameters :

X : ***{array-like, sparse matrix}, shape (m, n)***

-   Training Feature Vector where m is the number of sample and n is the
    number of features

> Y : ***array-like, shape (m,)***

-   Target (Dependent Variables) Vector Relative to X

> Returns :
>
> self : object

predict(X) --

Parameters :

X : ***{array-like, sparse matrix}, shape (m, n)***

-   Test Feature Vector where m is the number of sample and n is the
    number of features

> Returns :
>
> ***array, shape(m)***

-   Predicted Class Label per sample

[Pre-Requisites]{.underline}:

-   Code needs to be run in python [3.5 Or above]{.underline} as few
    functionalities used is only available in it.

-   Before running the Class python library [numpy needs to be imported
    as np]{.underline}

[Assumptions]{.underline} :

-   The Number of features should always be less than the number of
    examples

-   Here we are assuming that Training Features(X) are independent and
    the correlation between them is very less

-   The Class or the Dependent Variables are discrete

-   There should not be many outliers in the features

-   For the best result of the classifier the Input Feature Vector
    Should be Normalised before feeding to the system
