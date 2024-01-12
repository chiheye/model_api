# coding=utf-8

# 导入所需的库和模块
import argparse  # 用于处理命令行参数的库
import time  # 用于处理时间的库
from contextlib import asynccontextmanager  # 用于异步上下文管理器的库
from typing import List, Literal, Optional, Union  # 用于类型提示的库

import numpy as np  # 用于数值计算的库
import tiktoken  # 用于计算文本中的令牌数的库
import torch  # PyTorch深度学习库
import uvicorn  # ASGI服务器的库
from fastapi import Depends, FastAPI, HTTPException, Request  # FastAPI库的相关模块
from fastapi.middleware.cors import CORSMiddleware  # FastAPI中间件，处理CORS
from pydantic import BaseModel, Field  # 用于数据验证和设置的库
from sentence_transformers import SentenceTransformer  # 用于文本嵌入的库
from sklearn.preprocessing import PolynomialFeatures  # 用于多项式特征扩展的库
from sse_starlette.sse import EventSourceResponse  # 用于处理Server-Sent Events的库
from starlette.status import HTTP_401_UNAUTHORIZED  # HTTP状态码
from transformers import AutoModel, AutoTokenizer  # Hugging Face Transformers库

# 创建一个异步上下文管理器，用于收集GPU内存
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)

# 添加CORS中间件，处理跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义聊天消息的数据模型
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    prompt: str  # Update to include only "prompt"
    question: str  # Update to include only "question"

# 定义增量消息的数据模型
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    prompt: Optional[str] = None  # Update to include only "prompt"
    question: Optional[str] = None  # Update to include only "question"

# 定义聊天完成请求的数据模型
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False

# 定义聊天完成响应中的选择数据模型
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

# 定义聊天完成响应中的流式选择数据模型
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

# 定义聊天完成响应的数据模型
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

# 验证token的函数
async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if (
            token_type.lower() == "bearer"
            and token == "sk-aaabbbcccdddeeefffggghhhiiijjjkkk"
        ):  # 这里配置你的token
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )

# 定义嵌入请求的数据模型
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

# 定义嵌入响应的数据模型
class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

# 计算文本字符串中的令牌数量的函数
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 扩展特征的函数
def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        expanded_embedding = np.pad(
            expanded_embedding, (0, target_length - len(expanded_embedding))
        )
    return expanded_embedding

# 处理创建聊天完成请求的端点
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    prompt = request.messages[-1].prompt  # Use only "prompt"
    question = request.messages[-1].question  # Use only "question"

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        prompt = prev_messages.pop(0).prompt + prompt
        question = prev_messages.pop(0).question + question

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append(
                    {"role": prev_messages[i].role, "prompt": prev_messages[i].prompt, "question": prev_messages[i].question})
                history.append(
                    {"role": prev_messages[i+1].role, "prompt": prev_messages[i+1].prompt, "question": prev_messages[i+1].question})

    if request.stream:
        generate = predict(prompt, question, history, request.model)  # Use only "prompt" and "question"
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, prompt, question, history=history)  # Use only "prompt" and "question"

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", prompt=response, question=""),  # Use only "prompt"
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

# 预测函数，用于处理流式请求
async def predict(prompt: str, question: str, history: None, model_id: str):
    global model, tokenizer

    if history is None:
        history = []

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, prompt, question, history):  # Use only "prompt" and "question"
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(prompt="", question=new_text),  # Use only "prompt"
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[
                                       choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'

# 处理获取嵌入向量请求的端点
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest, token: bool = Depends(verify_token)
):
    # 计算嵌入向量和tokens数量
    embeddings = [embeddings_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    embeddings = [
        expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
        for embedding in embeddings
    ]

    # Min-Max normalization 归一化
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response

# 解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="None", type=str, help="Model name")
    args = parser.parse_args()

    model_dict = {
        "None":"chatglm3-6b",
    }

    model_name = model_dict.get(args.model_name, "chatglm3-6b")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.float16).cuda()
    embeddings_model = SentenceTransformer('m3e-large', device='cuda')

    # 运行FastAPI应用
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
