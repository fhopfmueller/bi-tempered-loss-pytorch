# Bi-tempered logistic loss: unofficial pytorch port

[Link to repo](https://github.com/fhopfmueller/bi-tempered-loss-pytorch)

Unofficial port from tensorflow to pytorch of parts of google's [bi-tempered loss](https://github.com/google/bi-tempered-loss), paper [here](https://arxiv.org/pdf/1906.03361.pdf).

Typical usage might look something like this:
```
from bi_tempered_loss_pytorch import bi_tempered_logistic_loss
...
t1, t2 = 0.8, 1.2
for epoch in range(epochs):
  for x, y in train_loader:
  ...
  loss = bi_tempered_logistic_loss(model(x), y, t1, t2)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
...
```
The following features are implemented:

- Tempered exponential, tempered logarithm, tempered softmax, tempered sigmoid, and their backward passes
- Bi-tempered logistic loss and its backward pass. Supports label smoothing, and the `reduction` keyword argument that pytorch losses typically have. Allows `t2 < 1.0`. The second argument can be a long tensor of one dimension less than the first argument (categorical labels) or a float tensor of the same dimension as the first argument (one-hot labels).
- Bi-tempered binary logistic loss

There is one difference to the official version under the hood: The forward pass of the tempered softmax requires an iterative procedure to compute a normalization, and one can define a custom gradient to avoid having to backprop through the iterative procedure. https://github.com/google/bi-tempered-loss defines custom gradients directly for the loss functions, not for the function that computes the normalization or for the tempered softmax. I've defined a custom gradient for the function that computes the normalization, which then gets used by the tempered softmax and the loss functions.

Comments are welcome!
