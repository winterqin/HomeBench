from transformers import AutoModelForCausalLM, AutoTokenizer
import time  # 导入time模块

model_name = r"E:\Qwen3\models\Qwen3_4B"

# 记录模型加载开始时间
load_start_time = time.time()
print(f"开始加载模型... {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(load_start_time))}")

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 记录模型加载结束时间
load_end_time = time.time()
load_duration = load_end_time - load_start_time
print(f"模型加载完成，耗时: {load_duration:.2f} 秒")

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 记录推理开始时间
infer_start_time = time.time()
print(f"\n开始推理... {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(infer_start_time))}")

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

# 记录推理结束时间
infer_end_time = time.time()
infer_duration = infer_end_time - infer_start_time
print(f"推理完成，耗时: {infer_duration:.2f} 秒")

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n" + "="*50)
print("thinking content:", thinking_content)
print("content:", content)
print("="*50)