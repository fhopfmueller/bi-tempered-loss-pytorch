import torch
from torch.autograd import gradcheck
from bi_tempered_loss_pytorch import *


def test_normalization():
    """Test the normalization constant."""
    for t in [2, 1.1]:
        shape = (10, 1000)

        activations = 10*torch.randn(shape, dtype=torch.double, requires_grad=True)
        normalization_constants = compute_normalization(activations, t, num_iters=5)
        assert normalization_constants.shape == shape[:-1] + (1,)
        probabilities = exp_t(activations - normalization_constants, t)
        assert (probabilities.sum(dim=-1) - torch.ones(shape[:-1], dtype=torch.double)).abs().max() < 1e-5, probabilities.sum(dim=-1)

        activations = .1 * torch.randn((2, 5), dtype=torch.double, requires_grad=True)

        grad_test = gradcheck(
                lambda activations: compute_normalization(activations, t, num_iters=5),
                activations,
                eps=1e-4,
                atol=1e-4)
        assert grad_test
        print(f'normalization constant tests passed for t={t}')
    for t in [.9, .1]:
        shape = (1, 1000)
        activations = 10*torch.randn(shape, dtype=torch.double, requires_grad=True)
        normalization_constants = compute_normalization(activations, t, num_iters=20)
        assert normalization_constants.shape == shape[:-1] + (1,)
        probabilities = exp_t(activations - normalization_constants, t)
        assert (probabilities.sum(dim=-1) - torch.ones(shape[:-1], dtype=torch.double)).abs().max() < 1e-5, probabilities.sum(dim=-1)
        print(f"normalization constant tests passed for t={t}. don't check gradient against numerical since binary search doesn't give a good numerical gradient")

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
    for t1 in [.1, .8, 1.]:
        for t2 in [.8, 1., 1.2, 2.]:
            labels = (10*torch.randn( (1, 10), dtype=torch.double)).softmax(dim=1)
            activations = torch.randn((1, 10), requires_grad=True, dtype=torch.double)
            grad_test = gradcheck(
                    lambda activations: bi_tempered_logistic_loss(
                        activations, labels, t1, t2, reduction='none', num_iters=30),
                    activations,
                    eps=1e-4,
                    atol=1e-4)
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

def test_binary_logistic_loss():
    """Test if binary logistic loss reduces correctly to logistic loss"""
    labels_two_categories = torch.randn(10, 2).softmax(dim=1)
    labels_binary = labels_two_categories[:, 0]
    activations_binary = torch.randn(10)
    activations_two_categories = torch.zeros(10, 2)
    activations_two_categories[:, 0] = activations_binary

    binary_loss = bi_tempered_binary_logistic_loss(activations_binary, labels_binary, .8, 1.2, reduction='none')
    cat_loss = bi_tempered_logistic_loss(activations_two_categories, labels_two_categories, .8, 1.2, reduction='none')
    assert (binary_loss - cat_loss).abs().max() < 1e-5
    print("Test if binary logistic loss reduces correctly to logistic loss passed")

def test_tempered_sigmoid():
    """test tempered sigmoid against precomputed"""
    activations = torch.Tensor([.0, 3., 6.])
    sigmoid_t_1 = tempered_sigmoid(activations, 1.0)
    assert (sigmoid_t_1 - activations.sigmoid()).abs().max() < 1e-5

    sigmoid_t_4 = tempered_sigmoid(activations, 4.0)
    expected_sigmoid_probabilities_t_4 = torch.Tensor([0.5, 0.58516014, 0.6421035])
    assert (sigmoid_t_4 - expected_sigmoid_probabilities_t_4).abs().max() < 1e-5
    print('tempered sigmoid tests passed')

if __name__=='__main__':
    test_normalization()
    test_limit_case_logistic_loss()
    test_loss_value()
    test_loss_gradient()
    test_label_smoothing()
    test_binary_logistic_loss()
    test_tempered_sigmoid()
