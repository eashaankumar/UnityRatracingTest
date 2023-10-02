class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def save_model(model, model_type, model_name):
    from torch import save
    from os import path
    if isinstance(model, model_type):
        return save(model.state_dict(), f'{model_name}.th')
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model_type, model_name):
    from torch import load
    from os import path
    r = model_type()
    r.load_state_dict(load(f'{model_name}.th', map_location='cpu'))
    return r

