import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer

model_kwargs = {}

quantization_config = None
# optional quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
model_kwargs["quantization_config"] = quantization_config

tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)

generate_text = pipeline(
    model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_fast=False,
    device_map={"": "cuda:0"},
    model_kwargs=model_kwargs,
)

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print(res[0]["generated_text"])
