# coding=utf-8
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import argparse
from fastapi import FastAPI, Request
import uvicorn, json, datetime
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from fastapi import FastAPI, Request, HTTPException


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
app = FastAPI()

# 在FastAPI应用中添加CORS中间件，处理跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 这里可以指定允许的源，或者使用["*"]允许任何源
    allow_credentials=True, # 这里可以设置是否允许跨域请求携带凭证（如cookies，授权头等）
    allow_methods=["*"], # 这里可以指定允许的HTTP方法，或者使用["*"]允许所有标准方法
    allow_headers=["*"], # 这里可以指定允许的HTTP请求头，或者使用["*"]允许所有请求头
)

# 验证token的函数
async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if (
            token_type.lower() == "bearer"
            and token == "sk-aaabbbcccdddeeefffggghhhiiijjsyzy"
        ):  # 这里配置你的token
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer, prompt_template
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    question = json_post_list.get('prompt')
    prompt = prompt_template.format(
        user_question=question.replace("#","")
    )
    sql_type = "自然语言转换成SQL查询"
    if sql_type in prompt:
        prompt += "```sql"
    else:
        prompt += ">>>"
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    print("加载模型并生成SQL查询以回答您的问题...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    print("==========输入========")
    print(prompt)
    generated_query = (
            pipe(
                prompt,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )[0]["generated_text"]
    )

    response = generated_query

    if sql_type in prompt:
      response = response.split("`sql")[-1].split("`")[0].split(";")[0].strip() + ";"

    else:
      response = response.split(">>>")[-1].split("`")[0].strip()

    print("========输出========")
    print(response)
    torch_gc()
    return response

if __name__ == '__main__':
    prompt_template = """
### Instructions:
{user_question}
### Response:
根据您的指示，以下是我为回答问题而生成的结果 `{user_question}`:
"""
    tokenizer = AutoTokenizer.from_pretrained("sqlcoder-7b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("sqlcoder-7b",
                                      trust_remote_code=True,
                                      torch_dtype=torch.float16,
                                      device_map="auto",
                                      use_cache=True)
    nest_asyncio.apply()
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1,log_level="info")#本地使用