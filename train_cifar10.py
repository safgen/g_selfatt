from model import get_model
from config import get_config
import dataset
import trainer
# import tester
import torch
import datetime
torch.backends.cuda.matmul.allow_tf32 = True


config = get_config()

model = get_model(config)


# config["device"] = (
#         "cuda:1" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
#     )
model = get_model(config)

# Define transforms and create dataloaders
dataloaders = dataset.get_dataset(config, num_workers=2)

# Create model directory and instantiate config.path
#model_path(config)

if config.pretrained:
    # Load model state dict
    model.module.load_state_dict(torch.load(config.path), strict=False)
    print("model_loaded-----"*5)

# Train the model
if config.train:
    # Print arguments (Sanity check)
    print(config)
    print(datetime.datetime.now())
    # Train the model
    trainer.train(model, dataloaders, config)

# Test model
# tester.test(model, dataloaders["test"], config)
