import uuid

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
from comfy.quant_ops import QuantizedTensor

PREFIX = "diffusion_model."


def _merged_params(model, module_name, device, dtype):
    """The module's weight and bias with its LoRA patches applied, as the sampler would see them."""
    m = comfy.utils.get_attr(model.model, module_name)
    if not hasattr(m, "_v"):
        keys = (f"{module_name}.{p}" for p in ("weight", "bias"))
        return {
            key.rsplit(".", 1)[1]: model.patch_weight_to_device(key, device_to=device, return_weight=True)
            for key in keys
            if key in model.patches
        }, None
    # Dynamic VRAM: run the same cast the sampler runs every step, so the merged weights match it bit for bit.
    weight, bias, stream = comfy.ops.cast_bias_weight(
        m,
        dtype=dtype,
        device=device,
        bias_dtype=dtype,
        offloadable=True,
        compute_dtype=dtype,
        want_requant=True,
    )
    return {"weight": weight, "bias": bias}, lambda: comfy.ops.uncast_bias_weight(m, weight, bias, stream)


def bake_patches(model):
    """A copy of `model` with its LoRA patches merged into the weights once, so sampling pays no per-step patch cost."""
    comfy.model_management.load_models_gpu([model])
    sd = model.model.diffusion_model.state_dict()
    device, dtype = model.load_device, model.model.get_dtype_inference()

    for module_name in sorted({key.rsplit(".", 1)[0] for key in model.patches if key.startswith(PREFIX)}):
        merged, release = _merged_params(model, module_name, device, dtype)
        for param, tensor in merged.items():
            key = f"{module_name[len(PREFIX) :]}.{param}"
            if tensor is None or PREFIX + key not in model.patches:
                continue
            parts = tensor.state_dict(key) if isinstance(tensor, QuantizedTensor) else {key: tensor}
            for part_key, part in parts.items():
                if part_key in sd:
                    sd[part_key] = part.to("cpu", copy=True)
        if release is not None:
            release()

    init = getattr(model, "cached_patcher_init", None)
    options = init[1][1] if init is not None and init[0] is comfy.sd.load_diffusion_model else {}
    baked = comfy.sd.load_diffusion_model_state_dict(sd, model_options=options)
    if baked is None:
        raise RuntimeError("MOSS Booster: could not rebuild the baked model")

    out = model.clone(model_override=baked.get_clone_model_override())
    out.patches = {}
    out.patches_uuid = uuid.uuid4()
    out.fast_disk = baked.fast_disk
    # The original factory would rebuild the unbaked model.
    out.cached_patcher_init = None
    return out
