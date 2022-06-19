import copy
import logging
import random

import torch

from src.utils.common import dist_tqdm
from src.utils.create_models import get_data_loader

log = logging.getLogger("uncertainty")


def mcbn_predict_aleatoric(model, test_inp, bn_std_var_buffer, iters):
    mcbn_samples = torch.zeros((len(test_inp), iters, 2), device='cuda')

    for mcbn_iter in range(iters):
        # set bn mean/std to the values of a random precalculated values
        bn_params = random.choice(bn_std_var_buffer)
        bn_params_index = 0
        for name, module in model.named_modules():
            if 'bn' in name:
                # module.train()
                module.running_mean = bn_params[bn_params_index][0]
                module.running_var = bn_params[bn_params_index][1]
                module.num_batches_tracked = torch.tensor(1, device='cuda')  # should be irrelevant
                # module.reset_running_stats()
                # module.track_running_stats = False
                bn_params_index += 1

        inputs = test_inp.cuda()

        with torch.no_grad():
            outputs = model(inputs)

        for sample_of_cur_batch in range(len(outputs)):
            mcbn_samples[sample_of_cur_batch][mcbn_iter] = outputs[sample_of_cur_batch]

    means = mcbn_samples[:, :, 0]
    # To avoid negative std values and mimic the training behavior .abs() is required here
    stds = mcbn_samples[:, :, 1].abs()

    # according to "what uncertainties do we need in bayesian deep learning for computer vision"
    mean = means.mean(dim=1)
    var = (means ** 2).mean(dim=1) - mean ** 2 + stds.mean(dim=1)
    std = var ** 0.5
    #
    return mean.view(-1, 1), std.view(-1, 1), mcbn_samples


def mcbn_predict(model, test_inp, bn_std_var_buffer, iters):
    mcbn_samples = torch.zeros((len(test_inp), iters, 1), device='cuda')

    for mcbn_iter in range(iters):
        # set bn mean/std to the values of a random precalculated values
        bn_params = random.choice(bn_std_var_buffer)
        bn_params_index = 0
        for name, module in model.named_modules():
            if 'bn' in name:
                # module.train()
                module.running_mean = bn_params[bn_params_index][0]
                module.running_var = bn_params[bn_params_index][1]
                module.num_batches_tracked = torch.tensor(1, device='cuda')  # should be irrelevant
                # module.reset_running_stats()
                # module.track_running_stats = False
                bn_params_index += 1

        inputs = test_inp.cuda()

        with torch.no_grad():
            outputs = model(inputs)

        for sample_of_cur_batch in range(len(outputs)):
            mcbn_samples[sample_of_cur_batch][mcbn_iter] = outputs[sample_of_cur_batch]

    means = mcbn_samples.mean(dim=1)
    stds = mcbn_samples.std(dim=1)

    return means, stds, mcbn_samples

    # sample_count = len(test_inp)
    # means = torch.zeros(sample_count, device='cuda')
    # stds = torch.zeros(sample_count, device='cuda')
    # sample_predictions = torch.zeros(iters, device='cuda')
    # for j in range(sample_count):
    #     sample = test_inp[j].reshape((1, 3, 224, 224))
    #     for i in range(iters):
    #         batch = torch.concat([sample, train_inp[i]])
    #
    #         sample_predictions[i] = model(batch)[0]  # only first element in batch is relevant, other other samples are train samples
    #     means[j] = sample_predictions.mean()
    #     stds[j] = sample_predictions.std(dim=0)
    # return means, stds


def mcbn_get_bn_params(config, eval_model, test_transforms):
    _bn_std_var_buffer = []

    logging.debug("collect bn parameters for each train batch")

    # set batchnorm layer to per mini batch mode
    _orig_values = []  # TODO remove
    for name, module in eval_model.named_modules():
        if 'bn' in name:
            _orig_values.append(copy.deepcopy((module.running_mean, module.running_var)))
            module.train()  # weights and biases will not be updated because no backprop will be done
            module.reset_running_stats()
            # only one batch is processed, and it should have full impact on "tracked" values.
            # momentum does not have to be resetted because this function should be called with a copy of the model
            # and for mcbn calculations it won't be relevant afterwards.
            module.momentum = 1.  # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            # module.num_batches_tracked = torch.tensor(1, device="cuda")

    tmp_model = copy.deepcopy(eval_model)

    # generate and collect running_mean and running_var for every bn layer and batch
    mcbn_train_dataloader = get_data_loader(data_path=config.data_path,
                                            csv_name=config.csv_name,
                                            batch_size=config.mcbn_batchsize,
                                            num_workers=config.num_workers,
                                            shuffle=True,
                                            transform=test_transforms,
                                            flip=True,
                                            dataset=config.dataset,
                                            manual_seed=config.seed)
    for batch_idx, (train_input, train_target, _) in enumerate(dist_tqdm(mcbn_train_dataloader)):
        train_input = train_input['cameraFront'].cuda()

        eval_model(train_input)

        if (batch_idx + 1) % 1 == 0:
            bn_params = []
            bn_params_index = 0
            for name, module in eval_model.named_modules():
                if 'bn' in name:
                    bn_params.append(copy.deepcopy((module.running_mean, module.running_var)))
                    # bn_params.append(copy.deepcopy((module.running_mean, copy.deepcopy(_orig_values[bn_params_index][1]))))
                    module.reset_running_stats()

                    # module.running_mean = bn_params[bn_params_index][0]
                    # module.running_var = bn_params[bn_params_index][1]
                    # module.num_batches_tracked = torch.tensor(1, device="cuda")
                    bn_params_index += 1
            # bn_params = _orig_values
            _bn_std_var_buffer.append(bn_params)

    eval_model.eval()

    # save values for eval_batchnorm_params.py
    #
    # with open('orig_values_mean.csv', 'w') as f:
    #     values = [x.item() for x in _orig_values[0][0]]
    #     for value in values:
    #         f.write(f"{value};")
    #     f.write('\n')
    #
    # with open('orig_values_var.csv', 'w') as f:
    #     values = [x.item() for x in _orig_values[0][1]]
    #     for value in values:
    #         f.write(f"{value};")
    #     f.write('\n')
    #
    # with open('mcbn_values_mean.csv', 'w') as f:
    #     for i in range(len(_bn_std_var_buffer)):
    #         values = [x.item() for x in _bn_std_var_buffer[i][0][0]]
    #         for value in values:
    #             f.write(f"{value};")
    #         f.write('\n')
    #
    # with open('mcbn_values_var.csv', 'w') as f:
    #     for i in range(len(_bn_std_var_buffer)):
    #         values = [x.item() for x in _bn_std_var_buffer[i][0][1]]
    #         for value in values:
    #             f.write(f"{value};")
    #         f.write('\n')

    return _bn_std_var_buffer


# with open('3.txt', 'w') as f:
#     for name, module in model.named_modules():
#         if 'conf' in name or 'relu' in name or 'pool' in name or 'sample' in name or 'fc' in name or 'backbone.head' in name or 'dropout' in name:
#             f.write(name)
#             f.write("\n")
#             f.write(str(module.state_dict()))
#             f.write("\n --- \n")

