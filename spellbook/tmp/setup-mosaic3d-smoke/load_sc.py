import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/Mosaic3D")
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

with initialize_config_dir(config_dir="/data/Mosaic3D/configs", version_base="1.3"):
    cfg = compose(config_name="eval", overrides=[
        "experiment=train_spunet_multidata_ppt", "data=sc", "trainer=cpu",
        "model/clip=recap_clip", "model/loss=caption", "ckpt_path=/data/mosaic3d/ckpts/sc.ckpt",
    ])
model = instantiate(cfg.model)
model.configure_model() if hasattr(model, "configure_model") else None
ck = torch.load("/data/mosaic3d/ckpts/sc.ckpt", map_location="cpu")
sd = ck["state_dict"]
for k in [k for k in sd if "emb_target" in k]:
    del sd[k]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("TYPE", type(model).__name__)
print("LOADED keys:", len(sd), "missing:", len(missing), "unexpected:", len(unexpected))
print("missing sample:", missing[:5]); print("unexpected sample:", unexpected[:5])
