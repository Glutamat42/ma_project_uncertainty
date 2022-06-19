import torch


def bbb_predict(model, X, num_predictions=10):
    """

    Args:
        model:
        X:
        y:
        num_predictions: authors used 1,2,5 and 10 (https://arxiv.org/pdf/1505.05424.pdf)
        std_multiplier:

    Returns:

    """
    preds = [model(X) for i in range(num_predictions)]
    preds = torch.stack(preds)
    means = preds.mean(dim=0)
    stds = preds.std(dim=0)
    # ci_upper = means + (std_multiplier * stds)  # , std_multiplier=2
    # ci_lower = means - (std_multiplier * stds)
    # ic_acc = (ci_lower <= y) * (ci_upper >= y)
    # ic_acc = ic_acc.float().mean()
    # return means, stds, ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
    return means, stds


def bbb_aleatoric_predict(model, X, num_predictions=10):
    """

    Args:
        model:
        X:
        y:
        num_predictions: authors used 1,2,5 and 10 (https://arxiv.org/pdf/1505.05424.pdf)
        std_multiplier:

    Returns:

    """
    preds = [model(X) for i in range(num_predictions)]
    preds = torch.stack(preds)
    means = preds[:, :, 0]
    # To avoid negative std values and mimic the training behavior .abs() is required here
    stds = preds[:, :, 1].abs()

    # according to "what uncertainties do we need in bayesian deep learning for computer vision"
    mean = means.mean(dim=0)
    var = (means**2).mean(dim=0)-means.mean(dim=0)**2 + stds.mean(dim=0)
    std = var ** 0.5

    return mean.view(-1, 1), std.view(-1, 1)


def bbb_aleatoric_sample_elbo(self,
                              inputs,
                              labels,
                              criterion,
                              criterion_mse,
                              sample_nbr,
                              complexity_cost_weight=1):
    """ sample elbo with support for aleatoric uncertainty and mse loss
    based on on original sample_elbo function
    https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/c2209eaff162f225522aa1bdf01bc0ac8790adc6/blitz/utils/variational_estimator.py

    format for criterion function (predicted_mean, label, predicted_std)
    criterion_mse: mse loss function to provide regular mse values for comparison

    setattr(nn_class, "sample_elbo", bbb_aleatoric_sample_elbo)
    """
    loss = 0
    mse_loss = 0

    # beta = 0.5

    for _ in range(sample_nbr):
        outputs = self(inputs)
        output_mu = outputs[:, 0].view(-1, 1)
        output_std = outputs[:, 1].view(-1, 1)
        # trying out stuff from "ON THE PITFALLS OF HETEROSCEDASTIC UNCERTAINTY ESTIMATION WITH PROBABILISTIC NEURAL NETWORKS"
        # loss += output_std * criterion(output_mu, labels, output_std ** 2)
        loss += criterion(output_mu, labels, output_std ** 2)
        loss += self.nn_kl_divergence() * complexity_cost_weight

        mse_loss += criterion_mse(output_mu, labels)
    return loss / sample_nbr, mse_loss / sample_nbr


def bbb_sample_elbo_with_mse(self,
                             inputs,
                             labels,
                             criterion,
                             criterion_mse,
                             sample_nbr,
                             complexity_cost_weight=1):
    """ sample elbo with support for mse loss
    based on on original sample_elbo function
    https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/c2209eaff162f225522aa1bdf01bc0ac8790adc6/blitz/utils/variational_estimator.py

    format for criterion function (predicted_mean, label, predicted_std)
    criterion_mse: mse loss function to provide regular mse values for comparison

    setattr(nn_class, "sample_elbo", bbb_aleatoric_sample_elbo)
    """
    loss = 0
    mse_loss = 0
    for _ in range(sample_nbr):
        outputs = self(inputs)
        loss += criterion(outputs, labels)
        loss += self.nn_kl_divergence() * complexity_cost_weight

        mse_loss += criterion_mse(outputs, labels)
    return loss / sample_nbr, mse_loss / sample_nbr