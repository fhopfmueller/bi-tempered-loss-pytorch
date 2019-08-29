import torch
from torch.autograd import gradcheck
from bi_tempered_loss_pytorch import *


def test_normalization():
    """Test the normalization constant."""
    activations = torch.randn((100, 50000), dtype=torch.double, requires_grad=True)
    normalization_constants = compute_normalization(activations, 1.01, num_iters=5)
    assert normalization_constants.shape == (100, 1)
    probabilities = exp_t(activations - normalization_constants, 1.01)
    assert (probabilities.sum(dim=-1) - torch.ones(100, dtype=torch.double)).abs().max() < 1e-5

    activations = torch.randn((10, 50), dtype=torch.double, requires_grad=True)

    grad_test = gradcheck(
            lambda activations: compute_normalization(activations, 1.01, num_iters=5),
            activations,
            eps=1e-6,
            atol=1e-5)
    assert grad_test
    print('normalization constant tests passed')

def test_limit_case_logistic_loss():
    """Test for checking if t1 = t2 = 1.0 yields the logistic loss."""
    labels_onehot = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = torch.LongTensor([0, 1, 2])
    activations = torch.randn((3,3))

    bi_tempered_loss_onehot = bi_tempered_logistic_loss(activations, labels_onehot, 1.0, 1.0, reduction='none')
    bi_tempered_loss = bi_tempered_logistic_loss(activations, labels, 1.0, 1.0, reduction='none')
    logistic_loss = torch.nn.functional.cross_entropy(activations, labels, reduction='none')

    assert (bi_tempered_loss_onehot - logistic_loss).abs().max() < 1e-5
    assert (bi_tempered_loss - logistic_loss).abs().max() < 1e-5

    bi_tempered_loss_onehot = bi_tempered_logistic_loss(activations, labels_onehot, 1.0, 1.0, reduction='mean')
    bi_tempered_loss = bi_tempered_logistic_loss(activations, labels, 1.0, 1.0, reduction='mean')
    logistic_loss = torch.nn.functional.cross_entropy(activations, labels, reduction='mean')

    assert (bi_tempered_loss_onehot - logistic_loss).abs().max() < 1e-5
    assert (bi_tempered_loss - logistic_loss).abs().max() < 1e-5


    print('test if bi tempered reduces to logistic for t1=t2=1.0 passed')

def test_loss_value():
    """Test loss against a precomputed value"""
    
    labels = torch.Tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]])
    activations = torch.Tensor([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]])
    precomputed_loss = torch.Tensor([0.02301914, 0.18972909, 0.93874922])
    loss = bi_tempered_logistic_loss(activations, labels, .5, 1.5, reduction='none')
    assert (loss - precomputed_loss).abs().max() < 1e-5
    print('test of loss against precomputed value passed')

def test_loss_gradient():
    labels = (10*torch.randn( (10, 10), dtype=torch.double)).softmax(dim=1)
    activations = torch.randn((10, 10), requires_grad=True, dtype=torch.double)
    grad_test = gradcheck(
            lambda activations: bi_tempered_logistic_loss(
                activations, labels, 0.5, 1.5, reduction='none', num_iters=10),
            activations,
            eps=1e-6,
            atol=1e-5)
    assert grad_test
    print('test of loss gradient passed')

def test_label_smoothing():
    """Test label smoothing."""
    labels = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = torch.Tensor([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]])
    actual_loss = bi_tempered_logistic_loss(
        activations, labels, 0.5, 1.5, label_smoothing=0.1, reduction='none')
    assert (actual_loss - torch.Tensor([0.76652711, 0.08627685, 1.35443510])).abs().max() < 1e-5
    print('label smoothing test passed')


if __name__=='__main__':
    test_normalization()
    test_limit_case_logistic_loss()
    test_loss_value()
    test_loss_gradient()
    test_label_smoothing()
