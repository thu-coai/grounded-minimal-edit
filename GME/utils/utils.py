from config import Config
config = Config()


def gpu_wrapper(item):
    if config.gpu:
        # print(item)
        return item.cuda()
    else:
        return item


def pretty_string(flt):
    ret = '%.6f' % flt
    if flt >= 0:
        ret = "+" + ret
    return ret


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
