---
layout: note
name: Neural Networks and Deep Learning
type: mooc
date: April 9, 2019
---

[Course 1 of the [deeplearning.ai](https://www.coursera.org/specializations/deep-learning) specialization on coursera]


**Basics of Neural Networks** - 

If we use logistic regression for binary classification of images e.g. cat vs non-cat.

So an image is a 64x64x3 matrix of rgb values - unroll them into a single row of 12288 (n) length vector (features).

If we have m training examples, create a matrix of m columns and n rows.

Labels for the training examples = a (1, M) matrix.

***Logistic Regression***:

![image tooltip here](/images/notes/neuralnetworksdeeplearning/logreg.png)

`y_pred = sigmoid(WtX + b)` where `sigmoid(z) = 1/(1 + e^-z)`

Loss function - Can’t use squared error, need alternative:

`Loss(y^, y) = -y*log(y^) + (1-y)*log(1-y^)`

If y = 1, want y^ large

If y = 0, want y^ small

![image tooltip here](/images/notes/neuralnetworksdeeplearning/costfn.png)

Loss function is the error for a single training example, while the cost function computes the average of losses for the entire training set.

We can view logistic regression as a very small neural network.

***Computation Graph***:

If we try to compute a function `J(a, b, c) = 3(a + b*c)`

First compute `u = b*c`, then compute `v = a+u`, then finally `J = 3*v`.

Put these 3 steps in a computation graph. 
This is useful when there is some special output that we try to minimize. In case of lr, we try to minimize the cost function, left to right, from a/b/c to u/v and J (forward computation). To compute derivates, we compute the opposite direction (backprop).

***Computing derivates***:

As `J = 3v`, `dJ/dv = 3`. 

And as `v = a+u`, `dJ/da = 3` (Chain rule: `dJ/da = dJ/dv * dv/da`)

Final output variable: J
=> d(FinalOutputVariable)/dvar

(In python, generally call this dJ/dvar or just dvar)

Most efficient way to compute derivates is to follow right-to-left direction. 

e.g. `dJ/db = dJ/dv * dv/du * du/db` = 3 * 1 * 3 = 9

Gradient descent for logistic regression:

`z = transpose(W) * x + b`

`y^ = a = sigmoid(z)`

`Loss fn = -y*log(a) - (1-y)*log(1-a)`

![image tooltip here](/images/notes/neuralnetworksdeeplearning/compgraph.png)

Compute gradients in the opposite direction of the computation graph. Finally perform gradient descent with alpha using computed derivates of dw1, dw2, db.

***Gradient descent on m examples***:

Overall dw1 for all m examples = average dw1 for each example = (1/m) * sum(dw1 for each example)
```
J = 0, dw1 = 0, dw2 =0 , db = 0
for i = 1 to m:
   zi = transpose(w)*xi + b
   ai = sigmoid(zi)
   J = - [yi * log ai + (1-yi) * log (1-ai)]
   dzi = ai - yi
   dw1 += x1i * dzi
   dw2 += x2i * dzi
   db += dzi

J /= m
# dw1 and dw2 and db are accumulators, finally average them
dw1 /= m
dw2 /= m
db /= m

# now increment gradient
w1 = w1 - alpha * dw1
w2 = w2 - alpha * dww
b = b - alpha * db
```

Use vectorization techniques to get rid of explicit for loops (which are not efficient in deep learning)

***Vectorization***:

In non-vectorized implementations, we use for loops over the feature vector.
e.g.
```
a = np.random.rand(1000000)
b = np.random.rand(1000000)

c = np.dot(a, b) # ~2ms
c = 0
for i in range(1000000):
    c += a[i]*b[i]
# ~ 475ms or 200 times slower
```

Built-in functions use SIMD - single instruction multiple data - GPU parallelism for faster computations.
(Whenever possible, avoid explicit for loops).

e.g. np.exp(v) to apply element-wise exponential, np.log(v), np.maximum(v, 0), 1/v etc


***Vectorizing Logistic Regression (no for loops!)***

To find forward propagation calculations:

Previously, `zi = wT*xi + b` and `ai = sigmoid(zi)`

Now, using np.dot to multiply 1xN w matrix with NxM x matrix, we get 1xM Z matrix result:

`Z = np.dot(transpose(w), X) + b`
Now, `A = sigmoid_for_matrix(Z)`

To find backward propagation derivatives:

Previously, `dzi = ai - yi`

Now, `dZ = A - Y # Y = [y1, y2 … ym]`

Previously, `dw1 += x1i * dzi`

Now, `dW = 1/m * X * tranpose(dZ)`

And `dB = np.sum(dZ) / m`

Finally, a highly vectorized implementation:
```
Z = np.dot(transpose(w), X) + b
A = sigmoid(Z)
dZ = A - Y
dW = 1/m * X * transpose(dZ)
dB = 1/m * np.sum(dZ)
W = W - alpha * dW
B = B - alpha * dB
```

