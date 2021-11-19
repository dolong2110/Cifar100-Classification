from .basic_nn import Net
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152

def get_model(model_name):
    if model_name == 'basic_nn':
        model = Net()
    elif model_name == 'resnet9':
        model = resnet9()
    elif model_name == 'resnet18':
        model = resnet18()
    elif model_name == 'resnet34':
        model = resnet34()
    elif model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'resnet101':
        model = resnet101()
    elif model_name == 'resnet152':
        model = resnet152()
    # elif model_name == 'mobilenet':
    #     from mobilenet import mobilenet
    #     model = mobilenet()
    else:
        raise 'Model {} not available'.format(model_name)

    return model