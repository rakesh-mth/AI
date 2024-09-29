from optimum.intel import OVWeightQuantizationConfig
from api.quantized import quantized

import openvino as ov
core = ov.Core()
print(core.available_devices)

model_name = "microsoft/phi-2"
precision = "f32"
quantization_config = OVWeightQuantizationConfig(bits=4, sym=False, group_size=128, ratio=0.8)
device = "gpu"

model, tokenizer = quantized(model_name, precision, quantization_config, device)

model.compile()