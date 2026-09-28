import os
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI()

model = None
tokenizer = None
device = None
num_layers = 0
num_heads = 0

cached_full_ids = None
cached_attentions = None

# MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_NAME = "Qwen/Qwen3-0.6B"

class GenerateRequest(BaseModel):
    prompt: str
    max_thinking_tokens: int = 4096
    max_tokens: int = 4096


def load_model():
    global model, tokenizer, device, num_layers, num_heads

    print(f"[1/2] 加载分词器: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"[2/2] 加载模型: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.backends.mps.is_available() or torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"模型加载完成 | 设备: {device} | 层数: {num_layers} | 注意力头: {num_heads}")


@app.get("/api/model-info")
async def model_info():
    return {
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    global cached_full_ids, cached_attentions

    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        gen1 = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=req.max_thinking_tokens,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )

    phase1_new = gen1[0, input_ids.shape[1]:]
    think_end_id = 151668

    if think_end_id not in phase1_new.tolist():
        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        end_token = torch.tensor([think_end_id, space_id], dtype=torch.long, device=device)
        current_ids = torch.cat([gen1[0], end_token], dim=0).unsqueeze(0)
    else:
        current_ids = gen1

    current_mask = torch.ones(current_ids.shape, dtype=torch.long, device=device)

    with torch.no_grad():
        gen2 = model.generate(
            current_ids, attention_mask=current_mask,
            max_new_tokens=req.max_tokens,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )

    full_ids = gen2[0]
    tokens = [tokenizer.decode([t.item()]) for t in full_ids]

    with torch.no_grad():
        outputs = model(full_ids.unsqueeze(0), output_attentions=True)

    attentions_raw = outputs.attentions

    cached_full_ids = full_ids
    cached_attentions = attentions_raw

    attention_data = []
    if attentions_raw is not None:
        for layer_attn in attentions_raw:
            if layer_attn is None:
                attention_data.append(None)
                continue
            attn_mean = layer_attn[0].mean(dim=0).cpu().tolist()
            attention_data.append(attn_mean)

    return JSONResponse({
        "tokens": tokens,
        "prompt_length": prompt_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "attentions": attention_data,
    })


@app.get("/api/attention-head")
async def attention_head(layer: int, head: int):
    if cached_attentions is None:
        return JSONResponse({"error": "请先生成文本"}, status_code=400)
    if layer < 0 or layer >= len(cached_attentions):
        return JSONResponse({"error": f"层索引越界: {layer}"}, status_code=400)

    layer_attn = cached_attentions[layer]
    if layer_attn is None:
        return JSONResponse({"attention": None})

    if head == -1:
        mat = layer_attn[0].mean(dim=0).cpu().tolist()
    else:
        if head < 0 or head >= layer_attn.shape[1]:
            return JSONResponse({"error": f"头索引越界: {head}"}, status_code=400)
        mat = layer_attn[0, head].cpu().tolist()

    return JSONResponse({"attention": mat})


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
