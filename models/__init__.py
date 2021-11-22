from .basic_nn import Net
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenet import mobilenet

models_dictionary = {'basic_nn': Net(), 'resnet9': resnet9(), 'resnet18': resnet18(),
                     'resnet34': resnet34(), 'resnet50': resnet50(), 'resnet101': resnet101(),
                     'resnet152': resnet152(), 'mobilenet': mobilenet()}

def get_model(model_name):
    model = models_dictionary.get(model_name)
    if not model:
        raise 'Model {} not available'.format(model_name)
    return model