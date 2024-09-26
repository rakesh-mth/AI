from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_id = "helenai/gpt2-ov"

model = OVModelForCausalLM.from_pretrained(model_id,
                                            # export=True,
                                            ov_config={ "KV_CACHE_PRECISION":"u8",
                                                      "DYNAMIC_QUANTIZATION_GROUP_SIZE":"32",
                                                       "PERFORMANCE_HINT":"LATENCY" })

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe("He's a dreadful magician and")

print(result[0]['generated_text'])