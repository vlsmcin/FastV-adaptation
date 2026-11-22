import requests
from PIL import Image
import time
import os

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TextStreamer,
)

gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
requested_model_size = os.environ.get("FASTV_MODEL_SIZE", "").lower()

if requested_model_size == "7b":
    model_id = "llava-hf/llava-1.5-7b-hf"
    revision = "a272c74"
elif requested_model_size == "13b":
    model_id = "llava-hf/llava-1.5-13b-hf"
    revision = "0fad4a3"
elif gpu_memory_gib >= 20:
    model_id = "llava-hf/llava-1.5-13b-hf"
    revision = "0fad4a3"
else:
    model_id = "llava-hf/llava-1.5-7b-hf"
    revision = "a272c74"

prompt = "USER: <image>\nWhat are these? Describe the image in details\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

fastv_config = {
    "use_fastv": True,
    "fastv_k": 3,
    "fastv_r": 0.75,
    "image_token_start_index": 5,
    "image_token_length": 576,
}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    device_map={"": "cuda:0"},
    attn_implementation="eager",
    fastv_config=fastv_config,  # comment this line to use vanilla decoding
)

processor = AutoProcessor.from_pretrained(
    model_id, 
    revision=revision
)

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors="pt").to(
    "cuda:0",
    torch.float16,
)

streamer = TextStreamer(processor)

print("----------------------------FastV-------------------------------")
print(fastv_config)
print(f"GPU memory: {gpu_memory_gib:.2f} GiB")
print(f"Model: {model_id}")

time_start = time.time()


output = model.generate(
                    **inputs,
                    min_new_tokens=200,
                    max_new_tokens=200,
                    do_sample=False,
                    use_cache=True,
                    streamer=streamer,
                    return_dict_in_generate=True,
                    )

time_end = time.time()

elapsed_time = time_end - time_start

# The number of tokens generated
num_tokens_generated = output.sequences.shape[1] - inputs["input_ids"].shape[1]

# Compute time per token
time_per_token = elapsed_time / num_tokens_generated

# print(processor.decode(output[0][2:], skip_special_tokens=True))
# Decode the output

# print(decoded_output)
print(f"Total time taken: {elapsed_time} seconds")
print(f"Time per token: {time_per_token} seconds/token")
