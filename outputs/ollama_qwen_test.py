import ollama

response = ollama.chat(
    model='qwen3-nothink',
    messages=[{
        'role': 'user',
        'content': "请解决以下问题，并分步展示你的推理过程。问题：如果小明有5个苹果，又买了3袋苹果，每袋有4个，他现在总共有多少个苹果？"
    }]
)
print(response['message']['content'])