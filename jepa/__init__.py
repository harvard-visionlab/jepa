import torch
from omegaconf import OmegaConf
import jepa.models.vision_transformer as vit

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder',
    **kwargs,
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    if frames_per_clip > 1:
        def forward_prehook(module, input):
            if len(input[0].shape)==4:
                input = input[0]  # [B, C, H, W]
                input = input.unsqueeze(2).repeat(1, 1, frames_per_clip, 1, 1)
            return (input)

        encoder.register_forward_pre_hook(forward_prehook)

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder
    
def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    print(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(f'loaded pretrained model with msg: {msg}')
    print(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

def load_model_from_weights_and_config(weights_file, config_file, **kwargs):
    # 1. Load the config (adjust the path to your config file)
    config = OmegaConf.load(config_file)

    # 2. Load the model
    model = init_model('cpu', weights_file, **config.model, **kwargs)
    print(model)

    return model, config