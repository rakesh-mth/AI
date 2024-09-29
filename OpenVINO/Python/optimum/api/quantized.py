import os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig


def quantized(model_name: str, precision: str, quantization_config: OVWeightQuantizationConfig, device: str):
    """ example:
        model_name = "microsoft/phi-2"
        precision = "f32"
        quantization_config = OVWeightQuantizationConfig(
            bits=4,
            sym=False,
            group_size=128,
            ratio=0.8,
        )
        device = "gpu"
    """

    save_name = model_name.split("/")[-1] + "_openvino"
    # Load kwargs
    load_kwargs = {
        "device": device,
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
            "CACHE_DIR": os.path.join(save_name, "model_cache"),  # OpenVINO will use this directory as cache
        },
        "compile": False,
        "quantization_config": quantization_config
    }

    # Check whether the model was already exported
    saved = os.path.exists(save_name)

    model = OVModelForCausalLM.from_pretrained(
        model_name if not saved else save_name,
        export=not saved,
        **load_kwargs,
    )

    # Load tokenizer to be used with the model
    tokenizer = AutoTokenizer.from_pretrained(model_name if not saved else save_name)

    # Save the exported model locally
    if not saved:
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)

    # TODO Optional: export to huggingface/hub

    model_size = os.stat(os.path.join(save_name, "openvino_model.bin")).st_size / 1024 ** 3
    print(f'Model size in FP32: ~5.4GB, current model size in 4bit: {model_size:.2f}GB')
    return model, tokenizer