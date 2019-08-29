# bi-tempered-loss-pytorch

Unofficial port from tensorflow to pytorch of parts of google's https://github.com/google/bi-tempered-loss, paper at https://arxiv.org/pdf/1906.03361.pdf.

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

- Tempered exponential, tempered logarithm, and tempered softmax, and their backward passes
- Bi-tempered logistic loss and its backward pass. Supports label smoothing, and the `reduction` keyword argument that pytorch losses typically have

The following features at https://github.com/google/bi-tempered-loss are not implemented:

- Binary bi-tempered logistic loss
- Features from https://github.com/google/bi-tempered-loss/commit/1e0849f2d07fe8b9eb1a8a66fb38dff33221b4c7, i.e. allowing
`t2 < 1.0`, tempered sigmoid, sparse bi-tempered logistic loss

There is one difference under the hood: The forward pass of the tempered softmax requires an iterative procedure to compute a normalization, and one can define a custom gradient to avoid having to backprop through the iterative procedure. https://github.com/google/bi-tempered-loss defines custom gradients directly for the loss functions, not for the function that computes the normalization or for the tempered softmax. I've defined a custom gradient for the function that computes the normalization, which then gets used by the tempered softmax and the loss functions.

Comments are welcome!
