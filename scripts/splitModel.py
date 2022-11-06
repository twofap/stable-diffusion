import torch
import argparse
from omegaconf import OmegaConf
from diffusion.util import instantiate_from_config
from transformers import logging
from diffusion import ddpm
logging.set_verbosity_error()

def load_model_from_config(ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


parser = argparse.ArgumentParser()

parser.add_argument(
    "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
)

opt = parser.parse_args()
precision = opt.precision


## Splitting model weights

print("Splitting model weights")

config = "diffusion/v1-inference.yaml"
config = OmegaConf.load(f"{config}")

ckpt = "models/diffusion/stable-diffusion-v1/model.ckpt"
sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

model = instantiate_from_config(config.modelUNet)
model.model1 = ddpm.DiffusionWrapper(model.unetConfigEncode)
model.model2 = ddpm.DiffusionWrapperOut(model.unetConfigDecode)
_, _ = model.load_state_dict(sd, strict=False)

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)


modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)



if precision == "autocast":

    model.half()
    modelCS.half()
    modelFS.half()

    state = {'state_dict': model.state_dict()}
    torch.save(state, "split_model/model1.4_fp16.pth")

    state = {'state_dict': modelFS.state_dict()}
    torch.save(state, "split_model/modelFS1.4_fp16.pth")

    state = {'state_dict': modelCS.state_dict()}
    torch.save(state, "split_model/modelCS1.4_fp16.pth")
    print("Split weights into FP16")

else:

    state = {'state_dict': model.state_dict()}
    torch.save(state, "split_model/model1.4_fp32.pth")

    state = {'state_dict': modelFS.state_dict()}
    torch.save(state, "split_model/modelFS1.4_fp32.pth")

    state = {'state_dict': modelCS.state_dict()}
    torch.save(state, "split_model/modelCS1.4_fp32.pth")
    print("Split weights into FP32")

## Split model created