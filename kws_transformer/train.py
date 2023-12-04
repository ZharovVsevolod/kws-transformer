import torch
from kws.model import ViT_audio

import hydra
from hydra.core.config_store import ConfigStore
from config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:

    net = ViT_audio(
        time_window=cfg.data.time_window,
        frequency=cfg.data.frequency,
        patch_size_t=cfg.data.patch_size_t,
        patch_size_f=cfg.data.patch_size_f,
        embed_dim=cfg.model.embedding_dim,
        num_classes=cfg.model.num_output_classes,
        depth=cfg.model.layers,
        num_heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim,
        norm_type=cfg.model.norm_type
    )
    print(sum(p.numel() for p in net.parameters()))

    x = torch.rand(cfg.training.batch, cfg.data.time_window, cfg.data.frequency)

    output = net(x)

    print(output.shape)
    print(output[:, -1, :])

if __name__ == "__main__":
    main()