from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import openvino as ov

core = ov.Core()
core.available_devices

def save_model():
    model = OVModelForCausalLM.from_pretrained("gpt2", export=True)
    model.save_pretrained("gpt2-ov")

def opt_pretrained():
    model_id = "helenai/gpt2-ov"
    # model_id = "gpt2-ov"

    model = OVModelForCausalLM.from_pretrained(model_id, # export=True,
                ov_config={ "KV_CACHE_PRECISION":"u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE":"32", "PERFORMANCE_HINT":"LATENCY" })

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    query = input("Ask something: ")

    #result = pipe("He's a dreadful magician and")
    result = pipe(query)

    print(result[0]['generated_text'])

# RUN PRETRAINED METHOD
opt_pretrained()

# SAVE A MODEL
# save_model()