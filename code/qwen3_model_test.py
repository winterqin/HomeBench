import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

def calculate_similarity(query_vector, vector_list, threshold):
    query_vector = query_vector / query_vector.norm(p=2)
    vector_list = vector_list / vector_list.norm(p=2, dim=1, keepdim=True)

    # Calculate cosine similarity
    similarities = torch.matmul(vector_list, query_vector)

    # Find indices where similarity exceeds the threshold
    indices = (similarities > threshold).nonzero(as_tuple=True)[0]

    return indices.tolist()

def calculate_topk_similarity(query_vector, vector_list, topk):

    # Ensure the input tensors are normalized for cosine similarity
    query_vector = query_vector / query_vector.norm(p=2)
    vector_list = vector_list / vector_list.norm(p=2, dim=1, keepdim=True)

    # Calculate cosine similarity
    similarities = torch.matmul(vector_list, query_vector)

    # Get the indices of the top-k most similar vectors
    topk_indices = torch.topk(similarities, topk).indices

    return topk_indices.tolist()

class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self,tokenizer,use_rag=False):
        self.tokenizer= tokenizer
        if use_rag:
            f = open("/home/slli/home_assistant/our_dataset/raw_data/rag_test_data.json", "r")
            self.data = json.loads(f.read())
            f.close()
        else:
            f = open("/home/slli/home_assistant/our_dataset/raw_data/test_data.jsonl", "r")
            lines = f.readlines()
            f.close()
            f_home = open("/home/slli/home_assistant/our_dataset/raw_data/new_home_status_method.jsonl", "r")
            lines_home = f_home.readlines()
            home_status = {}
            for line in lines_home:
                data = json.loads(line)
                home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
            f_home.close()
            system = open("/home/slli/home_assistant/our_dataset/raw_data/system.txt", "r").read()
            self.data = []
            for i in range(len(lines)):
                case = lines[i]
                case = json.loads(case) 
                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n" + "<Machine instructions:>"
                home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                input = system + home_status_case + device_method_case  + case_input
                output = case["output"]
                self.data.append({"input": input, "output": output})

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = [
            {"role":"system","content":item["input"]}
        ]
        output_text = item["output"]
        inputs_id = self.tokenizer.apply_chat_template(input_text,add_generation_prompt=True,tokenize=False)
        
        return inputs_id, output_text

class home_assistant_dataset(Dataset):
    def __init__(self,tokenizer,use_rag=False):
        self.tokenizer= tokenizer
        if use_rag:
            f = open(r"E:\Qwen3\HomeBench\dataset\rag_test_data.json", "r")
            self.data = json.loads(f.read())
            f.close()
        else:
            f = open(r"E:\Qwen3\HomeBench\dataset\test_data.jsonl", "r")
            lines = f.readlines()
            f.close()
            f_home = open(r"E:\Qwen3\HomeBench\dataset\home_status_method.jsonl", "r")
            lines_home = f_home.readlines()
            home_status = {}
            for line in lines_home:
                data = json.loads(line)
                home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
            f_home.close()
            examples = open(r"E:\Qwen3\HomeBench\dataset\prompt\example.txt", "r").read()
            system = open(r"E:\Qwen3\HomeBench\dataset\prompt\system.txt", "r").read()
            self.data = []
            for i in range(len(lines)):
                case = lines[i]
                case = json.loads(case) 
                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n" + "<Machine instructions:>"
                home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                input = system + home_status_case + device_method_case + examples + case_input
                output = case["output"]
                self.data.append({"input": input, "output": output})

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = [
            {"role":"user","content":item["input"]}
        ]
        output_text = item["output"]
        inputs_id = self.tokenizer.apply_chat_template(input_text,add_generation_prompt=True,tokenize=False)
        
        return inputs_id, output_text

def chang_json2strchunk(state,methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            state_str += "  state: " + state[room]["state"] + "\n"
            for attribute in state[room]["attributes"].keys():
                state_str += "  " + attribute + ": " + str(state[room]["attributes"][attribute]["value"])
                if "options" in state[room]["attributes"][attribute].keys():
                    state_str += " (options" + str(state[room]["attributes"][attribute]["options"]) + ")\n"
                elif "lowest" in state[room]["attributes"][attribute].keys():
                    state_str += " (range: " + str(state[room]["attributes"][attribute]["lowest"]) + " - " + str(state[room]["attributes"][attribute]["highest"]) + ")\n"
                else:
                    state_str += "\n"

        else:
            for device in state[room].keys():
                if device == "room_name":
                    continue
                else:
                    state_str += "  " + device + "\n"                    
                    state_str += "    state: " + state[room][device]["state"] + "\n"
                    for attribute in state[room][device]["attributes"].keys():
                        state_str += "    " + attribute + ": " + str(state[room][device]["attributes"][attribute]["value"])
                        if "options" in state[room][device]["attributes"][attribute].keys():
                            state_str += " (options" + str(state[room][device]["attributes"][attribute]["options"]) + ")\n"
                        elif "lowest" in state[room][device]["attributes"][attribute].keys():
                            state_str += " (range: " + str(state[room][device]["attributes"][attribute]["lowest"]) + " - " + str(state[room][device]["attributes"][attribute]["highest"]) + ")\n"
                        else:
                            state_str += "\n"
        state_str += "<chunk>"

    method_str = ""
    tmp_room_name = methods[0]["room_name"]
    for method in methods:
        if method["room_name"] != tmp_room_name:
            method_str += "<chunk>"
            tmp_room_name = method["room_name"]
        if method["room_name"] == "None":
            method_str += method["device_name"] + "." + method["operation"] + "("
        else:
            method_str += method["room_name"] + "." + method["device_name"] + "." + method["operation"] + "("
        if len(method["parameters"]) > 0:
            for parameter in method["parameters"]:
                method_str += parameter["name"] + ":" + parameter["type"] + ","
            method_str = method_str[:-1]
        method_str += "),"
    return state_str, method_str

def chang_json2str(state,methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            state_str += "  state: " + state[room]["state"] + "\n"
            for attribute in state[room]["attributes"].keys():
                state_str += "  " + attribute + ": " + str(state[room]["attributes"][attribute]["value"])
                if "options" in state[room]["attributes"][attribute].keys():
                    state_str += " (options" + str(state[room]["attributes"][attribute]["options"]) + ")\n"
                elif "lowest" in state[room]["attributes"][attribute].keys():
                    state_str += " (range: " + str(state[room]["attributes"][attribute]["lowest"]) + " - " + str(state[room]["attributes"][attribute]["highest"]) + ")\n"
                else:
                    state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name":
                    continue
                else:
                    state_str += "  " + device + "\n"
                    
                    state_str += "    state: " + state[room][device]["state"] + "\n"
                    for attribute in state[room][device]["attributes"].keys():
                        state_str += "    " + attribute + ": " + str(state[room][device]["attributes"][attribute]["value"])
                        if "options" in state[room][device]["attributes"][attribute].keys():
                            state_str += " (options" + str(state[room][device]["attributes"][attribute]["options"]) + ")\n"
                        elif "lowest" in state[room][device]["attributes"][attribute].keys():
                            state_str += " (range: " + str(state[room][device]["attributes"][attribute]["lowest"]) + " - " + str(state[room][device]["attributes"][attribute]["highest"]) + ")\n"
                        else:
                            state_str += "\n"

    method_str = ""
    for method in methods:
        if method["room_name"] == "None":
            method_str += method["device_name"] + "." + method["operation"] + "("
        else:
            method_str += method["room_name"] + "." + method["device_name"] + "." + method["operation"] + "("
        if len(method["parameters"]) > 0:
            for parameter in method["parameters"]:
                method_str += parameter["name"] + ":" + parameter["type"] + ","
            method_str = method_str[:-1]
        method_str += ");"
    return state_str, method_str

def compute_accuracy(generated_texts, expected_texts):
    macro_num_correct = 0
    micro_num_correct = 0
    micro_num_total = 0
    for generated_text, expected_text in zip(generated_texts, expected_texts):
        generated_text = generated_text.replace("<Machine instructions:>", "")
        generated_text = generated_text.replace(" ", "")
        generated_text = generated_text.replace("\n", "")
        
        ## 匹配{}中的内容
        # generated_text= generated_text.replace("{","")
        # generated_text = generated_text.replace("}","")
        # generated_text = generated_text.replace("<","")
        # generated_text = generated_text.replace(">","")
        generated_text = re.findall(r'\{(.*?)\}', generated_text)
        if len(generated_text) > 1:
            print("generated_text:",generated_text)
        else:
            generated_text = generated_text[0]



        ## 匹配‘''' '''中的内容
        expected_text = expected_text.replace("'''", "")
        expected_text = expected_text.replace(" ","")
        expected_text = expected_text.replace("\n","")
        generated_text = generated_text.split(",")
        expected_text = expected_text.split(",")

        generated_text = [x for x in generated_text if x != ""]
        expected_text = [x for x in expected_text if x != ""]
        generated_text = set(generated_text)
        expected_text = set(expected_text)
        print("generated_text:",generated_text)
        print("expected_text:",expected_text)

        if generated_text == expected_text:
            macro_num_correct += 1
        if len(expected_text) == 0 and len(generated_text) == 0:
            micro_num_correct += 1
            micro_num_total += 1
        elif len(expected_text) == 0:
            micro_num_total += 1
        else:
            micro_num_correct += len(generated_text & expected_text)
            micro_num_total += len(expected_text)
    return macro_num_correct / len(generated_texts), micro_num_correct / micro_num_total

def model_test(model_name,use_rag=False,use_few_shot=False,test_type=None):
    if model_name == "llama":
        model_id = '/home/slli/home_assistant/model/llama3-8b-Instruct'
    elif model_name == "qwen":
        model_id = 'E:\Qwen3\models\Qwen3_4B'
    elif model_name == "mistral":
        model_id = '/home/slli/home_assistant/model/Mistral-7B-Instruct-v0.3'
    elif model_name == "gemma":
        model_id = '/home/slli/home_assistant/model/Gemma-7B-Instruct-v0.3'

    print(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map="auto")
    if use_rag:
        test_dataset = rag_home_assistant_dataset(tokenizer)
    elif use_few_shot:
        test_dataset = home_assistant_dataset(tokenizer)
    else:
        test_dataset = no_few_shot_home_assistant_dataset(tokenizer)
        print("test_dataset:",len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=1)
    res = []
    start_time = time.time()
    for inputs_id, output_text in tqdm(test_loader):
        if model_name == "llama" or model_name == "mistral":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(list(inputs_id), return_tensors="pt",padding=True).to(model.device)
        if model_name == "llama":
            terminator = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            logits = model.generate(**inputs,max_new_tokens=1024,eos_token_id=terminator,do_sample=True,temperature=1.0,top_p=0.9,pad_token_id=tokenizer.eos_token_id)
        else:
            logits = model.generate(**inputs,max_new_tokens=1024,do_sample=True,temperature=1.0,top_p=0.9)
        response = logits[:, len(inputs['input_ids'][0]):]
        generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)


        for i in range(len(generated_texts)):
            res.append({"generated": generated_texts[i], "expected": output_text[i]})
    end_time = time.time()
    print("time:",end_time-start_time)
    f = open("/home/slli/home_assistant/our_dataset/raw_data/" + model_name + "_" + test_type + "_test_result.json", "w")
    f.write(json.dumps(res))

# def qwen_test():
#     model_id = 'E:\Qwen3\models\Qwen3_4B'
#     print(torch.cuda.is_available())
#     tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side='left')
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     test_dataset = home_assistant_dataset(tokenizer)
#     test_loader = DataLoader(test_dataset, batch_size=2)
#     res = []
#     for inputs_id, output_text in test_loader:
#         inputs = tokenizer(list(inputs_id), return_tensors="pt",padding=True).to(model.device)
#         logits = model.generate(**inputs,max_new_tokens=1024,do_sample=True,temperature=1.0,top_p=0.9)
#         response = logits[:, len(inputs['input_ids'][0]):]
#         generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
#         for i in range(len(generated_texts)):
#             res.append({"generated": generated_texts[i], "expected": output_text[i]})

#     with open("E:\Qwen3\HomeBench\outputs\qwen_test_result.json", "w") as f:
#         f.write(json.dumps(res))
def qwen_test():
    import torch
    import json
    import time  # 导入time模块
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    
    model_id = r'E:\Qwen3\models\Qwen3_4B'
    
    # 记录整体开始时间
    overall_start_time = time.time()
    print(f"{'='*60}")
    print(f"开始执行测试: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # 记录模型加载开始时间
    load_start_time = time.time()
    print(f"\n[阶段1] 开始加载模型和Tokenizer...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    load_end_time = time.time()
    print(f"[阶段1] 模型加载完成，耗时: {load_end_time - load_start_time:.2f} 秒")
    
    # 记录数据加载开始时间
    data_start_time = time.time()
    print(f"\n[阶段2] 开始加载数据集...")
    
    test_dataset = home_assistant_dataset(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    data_end_time = time.time()
    print(f"[阶段2] 数据集加载完成，耗时: {data_end_time - data_start_time:.2f} 秒")
    print(f"数据集共 {len(test_dataset)} 条数据，将分 {len(test_loader)} 个batch处理")
    
    res = []
    total_infer_time = 0  # 累计推理时间
    
    # 记录整体推理开始时间
    infer_loop_start = time.time()
    print(f"\n[阶段3] 开始推理循环...")
    
    # 处理每个batch
    for batch_idx, (inputs_id, output_text) in enumerate(test_loader):
        batch_start_time = time.time()
        print(f"\n  [Batch {batch_idx+1}/{len(test_loader)}] 开始处理...")
        
        # 处理输入
        processed_inputs = []
        for input_text in inputs_id:
            messages = [{"role": "user", "content": input_text}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            processed_inputs.append(text)
        
        inputs = tokenizer(
            processed_inputs,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # 生成响应
        with torch.no_grad():
            batch_infer_start = time.time()
            logits = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=1.0,
                top_p=0.9
            )
            batch_infer_end = time.time()
            batch_infer_duration = batch_infer_end - batch_infer_start
        
        # 提取生成的部分
        response = logits[:, inputs['input_ids'].shape[1]:]
        generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
        
        # 保存结果
        for i in range(len(generated_texts)):
            res.append({
                "generated": generated_texts[i],
                "expected": output_text[i]
            })
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        print(f"  [Batch {batch_idx+1}/{len(test_loader)}] 完成")
        print(f"    - 纯推理耗时: {batch_infer_duration:.2f} 秒")
        print(f"    - Batch总耗时: {batch_duration:.2f} 秒")
        print(f"    - 本batch生成 {len(generated_texts)} 条回复")
        
        total_infer_time += batch_infer_duration
    
    infer_loop_end = time.time()
    
    print(f"\n[阶段3] 推理循环完成")
    print(f"    - 总推理耗时: {total_infer_time:.2f} 秒")
    print(f"    - 含数据处理总耗时: {infer_loop_end - infer_loop_start:.2f} 秒")
    
    # 记录保存结果时间
    save_start_time = time.time()
    print(f"\n[阶段4] 开始保存结果...")
    
    output_path = r'E:\Qwen3\HomeBench\outputs\qwen_test_result.json'
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    
    save_end_time = time.time()
    print(f"[阶段4] 结果保存完成，耗时: {save_end_time - save_start_time:.2f} 秒")
    
    # 整体统计
    overall_end_time = time.time()
    print(f"\n{'='*60}")
    print(f"测试全部完成！")
    print(f"结果保存到: {output_path}")
    print(f"共处理 {len(res)} 条数据")
    print(f"总耗时: {overall_end_time - overall_start_time:.2f} 秒")
    print(f"{'='*60}")


if __name__ == "__main__":
    # llama_test()
    qwen_test()
    # qwen_test("qwen",use_rag=False,use_few_shot=True,test_type="error_input")
    # rag_dataset('/home/slli/home_assistant/model/Qwen2.5-7B-Instruct')
    # f = open("/home/slli/home_assistant/our_dataset/raw_data/llama_test_result.json", "r")
    # res = json.loads(f.read())
    # f.close()
    # generated_texts = []
    # expected_texts = []
    # for item in res:
    #     generated_texts.append(item["generated"])
    #     expected_texts.append(item["expected"][0])
    # print(compute_accuracy(generated_texts, expected_texts))
