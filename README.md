# Token 注意力可视化

使用 Qwen3-0.6B 生成文本，展示 prompt、回复及特殊 token，并按 Transformer 层和注意力头查看选中 token 的注意力分布。

## 本地运行

在项目根目录创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./start.sh
```

打开 http://127.0.0.1:8000 。首次启动需要从 Hugging Face 下载 `Qwen/Qwen3-0.6B`。模型通过 Accelerate 自动分配设备；代码在检测到 CUDA 或 Apple MPS 时选择 float16，否则选择 float32。

依赖版本记录自整理仓库时已有的本地环境，尚未在全新环境中验证安装。

输入 prompt 后点击「生成」，再点击 token、切换层级和注意力头查看分布。绿色背景越深，表示该行内相对注意力越高。侧栏 Top-5 当前使用该层各头的平均注意力。

`Max Thinking` 和 `Max Tokens` 分别限制两阶段生成的新 token 数量。当前实现第一阶段需要正数额度；完整注意力矩阵会随序列长度平方增长，建议先用较小额度运行。

## 文件

- `server.py`：FastAPI 服务、模型加载、文本生成及注意力提取。
- `index.html`：无需构建的前端页面。
- `start.sh`：使用项目内 `.venv` 启动服务。
- `requirements.txt`：Python 依赖及版本。

## 接口

- `GET /api/model-info`：模型名称、层数、注意力头数。
- `POST /api/generate`：接受 `prompt`、`max_thinking_tokens`、`max_tokens`，返回 token 和各层平均注意力矩阵。
- `GET /api/attention-head?layer=0&head=0`：返回最近一次生成的指定层、指定头的注意力矩阵；`head=-1` 表示各头平均。

服务监听本机地址，注意力缓存由进程内请求共享，适合本地单用户探索。
