from model import get_model
from config import get_config
import dataset
import trainer
import sys
#import tester
import argparse
import torch
import datetime
from torchinfo import summary
from timm import create_model, list_models
from pprint import pprint
import importlib
import os

torch.backends.cuda.matmul.allow_tf32 = True


def import_path(path, name="cfg"):
    # if name is None:
    #     name = Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module



parser = argparse.ArgumentParser(description="PyTorch Config Loader")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--checkpoint', type=str, required=False, help='Path to the checkpoint file', default='checkpoints/checkpoint')
args = parser.parse_args()
config_file = args.config
checkpoint_file = args.checkpoint if args.checkpoint else "checkpt"
print(checkpoint_file)
cfg = import_path(config_file)
cfg_str = config_file.split(".")[0]
# print(cfg_str)

config = cfg.get_config()

# model = get_model(config)


# config["device"] = (
#         "cuda:1" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
#     )
model = get_model(config).to(config.device)
# model = create_model('vit_tiny_patch16_224', pretrained=False)
# print(list_models("*vit*"))
# exit()
# print(torch.device(config.device))
# summary(model=model,
#         input_size=(config.batch_size, 3, 96, 96), # (batch_size, input_channels, img_width, img_height)
#         col_names=["input_size", "output_size", "num_params", "trainable",   #"params_percent",
#                 "kernel_size",
#                 "mult_adds"],
#         col_width=20,
#         row_settings=["var_names"],
#         depth = 6,
#         mode= 'train',
#         device=torch.device(config.device)
#         )

# pprint(model)
# exit()
# Define transforms and create dataloaders
dataloaders = dataset.get_dataset(config, num_workers=2)

# Create model directory and instantiate config.path
#model_path(config)

if config.pretrained:
    # Load model state dict
    # model.module.load_state_dict(torch.load(config.path), weights_only=True)
    try:
        model.load_state_dict(torch.load(config.path+"_"+config.dataset+"_"+config.model, weights_only=True))
        print("model_loaded-----"*5)
    except:
        print("model not found.... procedding without pretrain ")


if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("checkpoints_loaded -------------------------------------------------------------------------------------")

# Train the model
if config.train:
    # Print arguments (Sanity check)
    print(config)
    print(datetime.datetime.now())
    # Train the model
    trainer.train(model, dataloaders, config)

# Test model
# tester.test(model, dataloaders["test"], config)
