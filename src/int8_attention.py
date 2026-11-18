import comfy.ldm.modules.attention as attention

# Short sequences (the text blocks) gain nothing from INT8 and stay exact.
MIN_INT8_TOKENS = 1024


def int8_attention_override(start_percent):
    """An `optimized_attention_override` that runs INT8 attention once `start_percent` of the steps are done."""
    last = {"sigmas": None, "int8": True}

    def step_allows_int8(opts):
        sigmas, schedule = opts.get("sigmas"), opts.get("sample_sigmas")
        if start_percent <= 0 or sigmas is None or schedule is None:
            return True
        # Decided once per model call: the blocks share the same sigmas tensor.
        if sigmas is not last["sigmas"]:
            step = (schedule > sigmas.max() * 1.0001).sum().item()
            last["sigmas"], last["int8"] = sigmas, step >= start_percent * (len(schedule) - 1)
        return last["int8"]

    def override(func, q, k, v, *args, **kwargs):
        tokens = q.shape[2] if kwargs.get("skip_reshape") else q.shape[1]
        if tokens < MIN_INT8_TOKENS or not step_allows_int8(kwargs.get("transformer_options", {})):
            return func(q, k, v, *args, **kwargs)
        return attention.attention_comfy_kitchen_int8(q, k, v, *args, **kwargs)

    return override
