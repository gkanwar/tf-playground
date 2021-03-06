**Note (Jun 2021):** These were some pretty primitive attempts at what essentially
amounts to flows, before knowing about flows, and trying to learn
interpolators. Keeping this repo around for posterity only.

Need to include derivatives in the loss function itself.
Loss function schematically is a scalar fn of the difference
between the |det J| at the sample and the desired weighting.
Calculating |det J| requires knowing the derivative of each
output factor w.r.t. the inputs (i.e. one backprop per out
DoF).


Structure may look like:
```
[I] --A--> f_1 --B--> f_2 ... [O]
 --A'--> f'_1 --B'--> f'_2 ... [Pi] --> lf().
```


Each primed layer needs feed in from the relevant
values in at the corresponding forward layer. Also
must share parameters with those layers.


Simplest case: Just a linear layer, no non-linear functions.
```
[I] --(Ax+b)--> [O]
```
For each O_i, derivatives given by corresponding column of A.
Jacobian for all output is just |det A|. Given varying desired
weights, this model is (of course) un-trainable.


Next-simplest: Linear layer + isinh.
```
[I] --(Ax+b)--> isinh() = [O]
In math, [O] = isinh(A[I] + b).
d[O]/d[I] = isinh'(A[I]+b) A    (check: LHS or RHS?)
[CHECK]
    O_i = isinh(A_ij I_j + b_i).
    d O_i / d I_k = isinh'(A_ij I_j + b_i) * A_ik
    ... so if we think of isinh'(A[I]+b) as a diagonal matrix
    action, A lives to the RHS of it.
```

The forward network structure looks like
```
[I] --(Ax+b)--> isinh'() --> Pi ----> (*) = det J.
       \-------------------> det --/
```
The loss function compares `p[I]*det J` against `e^(-S[O])`.
Both `p[I]` and `e^(-S[O])` should be computable with smooth differentiable
elements.
e.g. `loss = (p[I]*det(J))^2 - e^(-2 S[O])`.
