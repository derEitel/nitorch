import torch
import torch.nn.functional as F

def bce_kl_loss(outputs, target):
    """ 
    Reconstruction loss for variational auto-encoders.
    Binary-cross entropy reconstruction + KL divergence losses summed
    over all elements and batch. 
    Mostly taken from pytorch examples: 
        https://github.com/pytorch/examples/blob/master/vae/main.py

    Arguments:
        outputs: List of the form [reconstruction, mean, logvariance].
        x: ground-truth.
    """
    recon_x, mu, logvar = outputs
    BCE = F.binary_cross_entropy(recon_x, target, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def mse_kl_loss(outputs, target):
    """ 
    Reconstruction loss for variational auto-encoders.
    Mean squared error reconstruction + KL divergence losses summed
    over all elements and batch. 
    Mostly taken from pytorch examples: 
        https://github.com/pytorch/examples/blob/master/vae/main.py

    Arguments:
        outputs: List of the form [reconstruction, mean, logvariance].
        x: ground-truth.
    """
    recon_x, mu, logvar = outputs
    MSE = F.mse_loss(recon_x, target, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def multihead_loss(outputs, target, loss_function):
    """
    Compute the loss on multiple outputs.

    Arguments:
        outputs: List of network outputs.
        target: List of targets where len(outputs) = len(target).
        loss_function: either list of loss functions with
        len(loss_function) = len(targets) or len(loss_function) = 1.
    """
    assert(len(outputs) == len(target))
    assert(len(loss_function) == len(target) or len(loss_function) == 1)
    # expand loss_function list if univariate
    if len(loss_function) == 1:
        loss_function = [loss_function[0] for i in range(len(target))]
    # compute loss for each head
    total_loss = 0.
    for out, gt, loss_func in zip(outputs, target, loss_function):
        loss = loss_func(out, gt)
        total_loss += loss
    return total_loss
