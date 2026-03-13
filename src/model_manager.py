import torch
import datetime
from pathlib import Path


def get_file_date():
    return datetime.datetime.now().strftime('%Y_%m.%d_%H-%M')


def save_model_state_dict(model: torch.nn.Module, save_path: Path):
    name: str = f"model_{get_file_date()}.dict"
    torch.save(model.state_dict(), save_path / name)


def load_model_with_state_dict(model: torch.nn.Module, path_to_model: Path):
    model = model()
    model.load_state_dict(torch.load(path_to_model, weights_only=True))
    return model


def save_model(model: torch.nn.Module, save_path: Path):
    name: str = f"model_{get_file_date()}.pt"
    torch.save(model, save_path / name)


def load_model(path_to_model: Path):
    model = torch.load(path_to_model, weights_only=False)
    return model

