from model import get_model
from config import get_config
import dataset
import trainer
# import tester
import torch
import datetime
from torchinfo import summary
from timm import create_model, list_models
from pprint import pprint

torch.backends.cuda.matmul.allow_tf32 = True


config = get_config()

# model = get_model(config)


# config["device"] = (
#         "cuda:1" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
#     )
model = get_model(config).to(config.device)
# model = create_model('vit_tiny_patch16_224', pretrained=False)
# print(list_models("*vit*"))
# exit()
# print(torch.device(config.device))
summary(model=model,
        input_size=(config.batch_size, 1, 28, 28), # (batch_size, input_channels, img_width, img_height)
        col_names=["input_size", "output_size", "num_params", "trainable",   #"params_percent",
                "kernel_size",
                "mult_adds"],
        col_width=20,
        row_settings=["var_names"],
        depth = 6,
        mode= 'train',
        device=torch.device(config.device)
        )

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

# Train the model
if config.train:
    # Print arguments (Sanity check)
    print(config)
    print(datetime.datetime.now())
    # Train the model
    trainer.train(model, dataloaders, config)

# Test model
# tester.test(model, dataloaders["test"], config)
