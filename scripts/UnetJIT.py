import torch
import argparse
from random import randint
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from diffusion.util import instantiate_from_config
from diffusion import ddpm
from contextlib import nullcontext
from transformers import logging
logging.set_verbosity_error()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="cpu or cuda",
)

parser.add_argument(
    "--precision", type=str, help="evaluate at this precision", choices=["fp32", "fp16"], default="fp16"
)

opt = parser.parse_args()
precision = opt.precision
device = opt.device


if(precision == "autocast"):
    assert device == "gpu"

## JIT compilation of UNet model 
n_iter = 1
batch_size = 1
config = "diffusion/v1-inference.yaml"
config = OmegaConf.load(f"{config}")
Height = 128
Width = 128
scale = 7.5 
seed = 1245
ddim_eta = 0.0
small_batch = "False"
turbo = True
from_file = False
n_samples = 1
unet_bs = 1
fixed_code = False
n_rows = 1
sampler = "plms"


if seed == None:
    seed = randint(0, 1000000)
seed_everything(seed)


if precision == "fp16":
    modelsd  = torch.load("split_model/model1.4_fp16.pth")
else:
    modelsd  = torch.load("split_model/model1.4_fp32.pth")


## Saving Betas, Alphas
model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(modelsd['state_dict'] , strict=False)

if precision == "fp32":
    state = {'state_dict': model.state_dict()}
    torch.save(state, "split_model/model1.4_fp32_betas.pth")

else:
    model.half()
    state = {'state_dict': model.state_dict()}
    torch.save(state, "split_model/model1.4_fp16_betas.pth")

del model

model = instantiate_from_config(config.modelUNet)
model.model1 = ddpm.DiffusionWrapper(model.unetConfigEncode)
model.model2 = ddpm.DiffusionWrapperOut(model.unetConfigDecode)
_, _ = model.load_state_dict(modelsd['state_dict'] , strict=False)

if precision == "fp16":
    precision_scope = autocast
    model.half()
else:
    precision_scope = nullcontext

model.to(device)

# JIT model
inp = torch.rand([1, 4, 64, 64]).to(device), torch.tensor([981]).to(device), torch.rand([1, 77, 768]).to(device)

with precision_scope("cuda"):
    m = torch.jit.trace(model.model1, inp)
    
torch.jit.save(m, "split_model/model1_traced_" + device + "_" + precision + ".pth")


print("Model 1 saved")

# JIT model2
ten =  [torch.rand([1, 320, 32, 32]).to(device),
        torch.rand([1, 320, 32, 32]).to(device),
        torch.rand([1, 320, 32, 32]).to(device),
        torch.rand([1, 320, 16, 16]).to(device),
        torch.rand([1, 640, 16, 16]).to(device),
        torch.rand([1, 640, 16, 16]).to(device),
        torch.rand([1, 640, 8, 8]).to(device),
        torch.rand([1, 1280, 8, 8]).to(device),
        torch.rand([1, 1280, 8, 8]).to(device),
        torch.rand([1, 1280, 4, 4]).to(device),
        torch.rand([1, 1280, 4, 4]).to(device),
        torch.rand([1, 1280, 4, 4]).to(device)]
inp = (torch.rand([1, 1280, 4, 4]).to(device), torch.rand([1, 1280]).to(device),ten, torch.rand([1, 77, 768]).to(device))

with precision_scope("cuda"):
    m = torch.jit.trace(model.model2, inp)

torch.jit.save(m, "split_model/model2_traced_" + device + "_" + precision + ".pth")

print("Model 2 saved")