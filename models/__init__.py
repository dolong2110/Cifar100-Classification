from .basic_nn import Net, LinearRegression
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenet import mobilenet, mobilenetv2

models_dictionary = {'basic_nn': Net(), 'linear_regression': LinearRegression(), 'resnet9': resnet9(),
                     'resnet18': resnet18(), 'resnet34': resnet34(), 'resnet50': resnet50(), 'resnet101': resnet101(),
                     'resnet152': resnet152(), 'mobilenet': mobilenet(), 'mobilenetv2': mobilenetv2()}

def get_model(model_name):
    model = models_dictionary.get(model_name)
    if not model:
        raise 'Model {} not available'.format(model_name)
    return model