import torch
import datetime
from pathlib import Path


def get_file_date():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')


def save_model_state_dict(model: torch.nn.Module, save_path: Path):
    print("DO NOT USE THIS FUNCTION, USE load_model_with_state_dict() INSTEAD!")
    name: str = f"model_{get_file_date()}.dict"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path / name)


def load_model_with_state_dict(model: torch.nn.Module, path_to_model: Path):
    model = model()
    model.load_state_dict(torch.load(path_to_model, weights_only=True))
    return model


def save_model(model: torch.nn.Module, save_path: Path, file_path: str):
    name: str = file_path.split("/")[-1] + f".{get_file_date()}.pt"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.to('cpu'), save_path / name)


def load_model(path_to_model: Path):
    model = torch.load(path_to_model, weights_only=False)
    return model


def print_model_parameters(model: torch.nn.Module):
    def get_num_params(module: torch.nn.Module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"Model Parameters: {get_num_params(model):,} (", end=' ')

    # iterate over all attributes of the model and print the respective number of parameters
    modules = model._modules
    for name, attr in modules.items():
        if isinstance(attr, torch.nn.ModuleDict):
            for true_name, module in attr.items():
                print(f"{true_name} {get_num_params(module):,}", end=' | ')

        elif isinstance(attr, torch.nn.Module):
            print(f"{name} {get_num_params(attr):,}", end=' | ')

    print(')')
