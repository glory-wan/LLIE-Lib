import os.path

from thop import profile
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch
import time
from torchvision import models


def get_model(model_name='resnet50'):
    if model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'resnet101':
        model = models.resnet101()
    elif model_name == 'vgg16':
        model = models.vgg16()
    elif model_name == 'densenet121':
        model = models.densenet121()
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
    elif model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'resnet34':
        model = models.resnet34()
    elif model_name == 'resnet152':
        model = models.resnet152()
    elif model_name == 'vgg11':
        model = models.vgg11()
    elif model_name == 'vgg13':
        model = models.vgg13()
    elif model_name == 'vgg19':
        model = models.vgg19()
    elif model_name == 'densenet169':
        model = models.densenet169()
    elif model_name == 'densenet201':
        model = models.densenet201()
    elif model_name == 'densenet161':
        model = models.densenet161()
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large()
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small()
    elif model_name == 'inception_v3':
        model = models.inception_v3()
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1()
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7()
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5()
    elif model_name == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0()
    elif model_name == 'squeezenet1_0':
        model = models.squeezenet1_0()
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1()
    elif model_name == 'alexnet':
        model = models.alexnet()
    elif model_name == 'googlenet':
        model = models.googlenet()
    elif model_name == 'regnet_y_400mf':
        model = models.regnet_y_400mf()
    elif model_name == 'regnet_y_800mf':
        model = models.regnet_y_800mf()
    elif model_name == 'regnet_x_1_6gf':
        model = models.regnet_x_1_6gf()
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny()
    elif model_name == 'convnext_small':
        model = models.convnext_small()
    elif model_name == 'convnext_base':
        model = models.convnext_base()
    elif model_name == 'convnext_large':
        model = models.convnext_large()
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s()
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m()
    elif model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l()
    elif model_name == 'mnasnet0_5':
        model = models.mnasnet0_5()
    elif model_name == 'mnasnet0_75':
        model = models.mnasnet0_75()
    elif model_name == 'mnasnet1_0':
        model = models.mnasnet1_0()
    elif model_name == 'mnasnet1_3':
        model = models.mnasnet1_3()
    elif model_name == 'regnet_y_1_6gf':
        model = models.regnet_y_1_6gf()
    elif model_name == 'regnet_y_3_2gf':
        model = models.regnet_y_3_2gf()
    elif model_name == 'regnet_y_16gf':
        model = models.regnet_y_16gf()
    elif model_name == 'regnet_x_3_2gf':
        model = models.regnet_x_3_2gf()
    elif model_name == 'regnet_x_8gf':
        model = models.regnet_x_8gf()
    elif model_name == 'vit_b_16':
        model = models.vit_b_16()
    elif model_name == 'vit_b_32':
        model = models.vit_b_32()
    elif model_name == 'vit_l_16':
        model = models.vit_l_16()
    elif model_name == 'vit_l_32':
        model = models.vit_l_32()
    elif model_name == 'swin_t':
        model = models.swin_t()
    elif model_name == 'swin_s':
        model = models.swin_s()
    elif model_name == 'swin_b':
        model = models.swin_b()
    elif model_name == 'swin_l':
        model = models.swin_l()
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    model.eval()
    return model


def get_model_stat(model, input_tensor, output_path=None, device='cuda'):
    flops, params = profile(model, inputs=(input_tensor,))
    try:
        res = summary(model,
                      input_size=tuple(input_tensor.shape[1:]),
                      save_path=output_path,
                      device=device
                      )
        print(res)
    except RuntimeError as e:
        if "out of" in str(e):
            print("CUDA out of memory error during summarizing on GPU")
            torch.cuda.empty_cache()
            save_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            outMemFile = os.path.join(save_dir,f'out_of_Mem_{filename}')
            with open(outMemFile, "w") as f:
                f.write('CUDA out of memory')
        else:
            print(f"Error during summary: {e}")

    return flops, params


def summary(model, input_size, save_path=None, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # device = device.lower()
    # assert device in [
    #     "cuda",
    #     "cpu",
    # ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    print('check device:')
    if "cuda" in device and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    elif device == 'cpu':
        dtype = torch.FloatTensor
    else:
        raise ValueError("Input device is not valid, please specify 'cuda' or 'cpu'")

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # collect summary details
    summary_str = ""
    summary_str += "----------------------------------------------------------------\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================\n"
    summary_str += "Total params: {0:,}\n".format(total_params)
    summary_str += "Trainable params: {0:,}\n".format(trainable_params)
    summary_str += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    summary_str += "----------------------------------------------------------------\n"
    summary_str += "Input size (MB): %0.2f\n" % total_input_size
    summary_str += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
    summary_str += "Params size (MB): %0.2f\n" % total_params_size
    summary_str += "Estimated Total Size (MB): %0.2f\n" % total_size
    summary_str += "----------------------------------------------------------------\n"

    if save_path:
        with open(save_path, "w") as f:
            f.write(summary_str)

    return summary_str
