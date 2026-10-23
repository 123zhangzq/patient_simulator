# Qwen Therapist–Patient 对话实验

在 **Linux + NVIDIA GPU** 服务器上，本地加载一份 Qwen 权重，让 Therapist 和 Patient 使用各自的 prompt，交替完成 10 轮对话。每轮由 Therapist 先说、Patient 回复，默认共生成 **20 条消息**。

本仓库只包含代码、配置和 prompt。模型在服务器首次部署时下载，以后复用服务器上的模型目录。本项目按 Linux GPU 环境编写；依照任务要求，开发时没有安装推理环境、下载模型或运行任何本地测试，实际运行需在你的 GPU 服务器上完成。

## 1. 模型选择与服务器要求

默认配置 `config.json` 使用 **`Qwen/Qwen3.5-9B`**。截至 2026-10-23，选择这一较新的、接近 8B 规模的 Qwen 模型；它实际是 **9B**，不是 7B/8B。若严格要求 8B，仓库同时提供 `config.qwen3-8b.json`，使用 **`Qwen/Qwen3-8B`**。两者的官方模型页面分别为 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 和 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)。本实验只使用文本对话能力。

建议服务器配置：

| 项目 | 建议 |
| --- | --- |
| 操作系统 | Linux，以下命令以 Bash 为例 |
| Python | 3.11，使用独立虚拟环境 |
| GPU | 单张 NVIDIA CUDA GPU，建议至少 24 GB 可用显存 |
| 系统内存 | 建议 32 GB 或更多 |
| 磁盘 | 建议预留 40–60 GB，用于模型、Python 环境和下载临时文件 |
| 网络 | 首次安装依赖和下载模型时，能够访问 GitHub、PyTorch 下载站、Hugging Face |

上述容量是部署预算估计，不是经过本项目实测的最低配置。实际显存取决于模型、精度、上下文长度以及同卡其他进程。代码将模型完整放在一张 GPU 上，不启用量化或 CPU offload；Therapist 和 Patient 共享同一份模型，顺序生成，不加载两套权重。

服务器需要已有可用的 NVIDIA 驱动。可先在**服务器**终端查看：

```bash
nvidia-smi
```

如果命令不存在或看不到 GPU，请先让服务器管理员配置驱动或 GPU 容器。`nvidia-smi` 中的 `CUDA Version` 表示驱动所支持的 CUDA 上限，不代表已安装对应版本的 CUDA Toolkit。PyTorch 需要选择与驱动、GPU 架构兼容的 CUDA wheel；本项目不需要自行编译 CUDA 扩展。

## 2. 将这个独立仓库上传到 GitHub

当前工作区中的 **`code/` 本身就是独立 Git 仓库根目录**。上传的是它里面的文件，不要把模型权重和虚拟环境上传到 GitHub。

先在 GitHub 网页创建一个空仓库，例如 `patient-simulator`。创建时先不要添加 README、License 或 `.gitignore`，再在自己的电脑中执行；把路径、用户名和仓库名换成自己的：

```bash
cd /你的路径/Patient_Simulator/code
git add .
git commit -m "Initial Qwen dialogue simulator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/patient-simulator.git
git push -u origin main
```

如果 Git 要求身份信息，先为这个仓库设置 `git config user.name "你的名字"` 和 `git config user.email "你的邮箱"`，再提交。若已经有 `origin`，使用 `git remote set-url origin 新仓库地址`。GitHub 推送身份验证与 Hugging Face 模型下载身份验证彼此独立；可使用你已有的 GitHub SSH 或 HTTPS 凭据。

`.gitignore` 已排除 `models/`、`outputs/`、`.venv/` 等生成目录。后续修改代码或 prompt 后，仍在此仓库里提交并推送；本项目不会代你创建 GitHub 仓库或上传。

## 3. 第一次在 Linux GPU 服务器部署

以下命令都在 **Linux GPU 服务器** 执行。没有特别说明时，工作目录是克隆后的仓库根目录。

### 3.1 获取代码和基础工具

Ubuntu/Debian 服务器若缺少基础工具，并且你有管理员权限，可安装：

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

其他发行版使用相应的包管理器；无管理员权限时使用服务器已有工具或请管理员安装。然后克隆自己上传的仓库：

```bash
git clone https://github.com/YOUR_USERNAME/patient-simulator.git
cd patient-simulator
```

**这里已在仓库根目录，不要再执行 `cd code`。** 应当可以在当前目录看到 `config.json`、`download_model.py` 和 `run_dialogue.py`。

### 3.2 建立 Python 3.11 虚拟环境

如果服务器已经有 `python3.11`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

若提示缺少 `venv` 模块，请安装该 Python 对应的 `venv` 包。Ubuntu/Debian **只有在当前系统软件源确实提供 Python 3.11 时**，才可安装 `python3.11-venv`；不同 Linux 版本的默认 Python 不同，不能假设所有 Ubuntu 版本都能直接安装 3.11。

如果没有 Python 3.11，可用服务器已有的 Conda 提供 Python。若连 Conda 也没有，以下是 **Linux x86_64** 用户目录安装 Miniforge 的完整示例；ARM 服务器应改用 [Miniforge 官方页面](https://github.com/conda-forge/miniforge) 对应架构的安装包。已有 Conda 时跳过下载和安装三行，直接创建环境：

```bash
curl -L -o /tmp/Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

conda create -n patient-sim-python python=3.11 -y
conda activate patient-sim-python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

这里 Conda 仅提供 Python 3.11，项目依赖仍统一装在仓库的 `.venv/` 中。以后使用 `source .venv/bin/activate` 即可；保留用于提供 Python 的 Conda 环境，不要删除它。

### 3.3 安装 CUDA 版 PyTorch 和项目依赖

先打开 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)，选择 **Linux → Pip → Python → 适合当前 GPU 与驱动的 CUDA 版本**，在已激活的 `.venv` 中执行其命令。默认 Qwen3.5 使用完整多模态模型类加载权重，即使只做文本生成，也安装版本匹配的 `torchvision` 和 Pillow；不需要 `torchaudio`。

本项目提供的固定版本示例是 [PyTorch 官方列出的 2.10.0 + torchvision 0.25.0 组合](https://pytorch.org/get-started/previous-versions/#v2100)。服务器 GPU 与驱动支持 CUDA 12.8 wheel 时，可执行：

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`cu128` 是安装示例，不表示每台服务器都适用。同一组合也提供 `cu126` wheel；如与服务器 GPU、驱动兼容，可将上述网址末尾改为 `cu126`。更新的 GPU 可能需要更新的 PyTorch/CUDA wheel，较旧驱动可能需要管理员升级，按官网与服务器实际情况选择。不要安装官网中的 CPU-only 版本，也不要混搭不匹配的 `torch` 与 `torchvision`。

`requirements.txt` 固定 [Transformers 5.20.2](https://github.com/huggingface/transformers/releases/tag/v5.20.2)，并包含 Accelerate、Hugging Face Hub、Safetensors 和 Pillow。PyTorch 与 torchvision 单独安装，避免依赖文件替你选择错误的 CUDA 构建。不要复用旧项目中的 Transformers 4.x 环境。Qwen3.5 使用原生的 `Qwen3_5ForConditionalGeneration`，保留视觉权重但只输入文本；Qwen3 使用 `AutoModelForCausalLM`。未启动 vLLM、HTTP 服务或云端推理 API。

### 3.4 Hugging Face：是否需要点击同意或设置 API key

**本仓库默认的两个 Qwen 模型是公开模型，正常情况下不需要申请访问、不需要点击 Agree，也不需要 Hugging Face token。** 可以直接跳到下一节下载。模型页上没有“同意”按钮是正常现象；页面中的 license 说明应自行阅读。无需设置 OpenAI、DashScope 或其他云端模型 API key。

如果之后换成 gated（需要授权）或私有模型，或希望登录后下载，流程如下：

1. 在自己电脑的浏览器打开相应模型的 Hugging Face 页面，并登录自己的 Hugging Face 账号。
2. **只有页面要求时**，阅读条款、填表并点击 `Agree` / `Agree and access repository` / `Request access` 等按钮。某些模型需等待作者批准；仅创建 token 不会自动获得访问权。
3. 打开 [Hugging Face Access Tokens 设置](https://huggingface.co/settings/tokens)，创建具有 **Read** 权限的 User Access Token，或使用范围仅覆盖所需模型的细粒度读取 token。
4. 回到服务器、保持虚拟环境激活，执行：

   ```bash
   hf auth login
   ```

5. 根据 CLI 提示完成登录：可以选择粘贴 access token；较新的 CLI 也可能默认提供浏览器授权流程。若提示是否将 token 加入 Git 凭据，单纯下载模型不需要，可选 `n`。

这个 token 用于**下载认证**，不是云端推理 API key。不要写进 Python 文件、`config.json`、README、Git 提交或终端命令历史。`hf auth login` 会将凭据保存在当前服务器用户的 Hugging Face 配置目录。详细行为见 [HF CLI 登录文档](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login) 与 [gated 模型文档](https://huggingface.co/docs/hub/models-gated)。

### 3.5 首次下载模型

使用默认 Qwen3.5-9B：

```bash
python download_model.py
```

脚本读取 `config.json`，将模型与 tokenizer 所需文件下载到 `models/Qwen3.5-9B/`，并在模型目录写入 `download_manifest.json`，记录实际下载的 Hugging Face commit。下载阶段需要网络，不占用 GPU。

如果选择严格 8B 的 Qwen3-8B，改为：

```bash
python download_model.py --config config.qwen3-8b.json
```

它使用单独的 `models/Qwen3-8B/` 目录。**只下载你准备使用的那一个模型即可。** 两份配置使用不同路径，避免互相覆盖。

若下载因网络中断失败，可保持相同配置重新运行下载命令，复用已下载文件。只有命令成功结束后再运行实验。下载脚本负责联网获取模型，运行脚本始终读取本地文件，不会在缺文件时自动联网补齐。

## 4. 首次运行 10 轮对话

默认 Qwen3.5-9B：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py
```

Qwen3-8B：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --config config.qwen3-8b.json
```

`CUDA_VISIBLE_DEVICES=0` 选择服务器上编号为 0 的 GPU。要使用编号为 2 的 GPU，就改成 `CUDA_VISIBLE_DEVICES=2`；程序会把选中的那张卡视为当前进程的第一张 GPU。不要同时运行多个实验占满同一张卡。

运行时按顺序生成：

```text
第 1 轮：Therapist → Patient
第 2 轮：Therapist → Patient
...
第 10 轮：Therapist → Patient
```

`prompts/opening.txt` 是触发 Therapist 第一次发言的启动指令，不计入这 20 条消息。每次调用使用当前角色自己的 system prompt；历史中该角色的发言映射为 `assistant`，另一角色的发言映射为 `user`。Patient 的私有背景只放入 Patient 的 system prompt，Therapist 只看到其自己的 prompt 和已公开的对话内容。

代码关闭模型的 thinking 输出，便于直接获得角色台词。每个角色共享模型权重，但 prompt 和历史角色映射彼此独立；修改 prompt 不会训练或修改权重。

## 5. 第二次及之后运行：直接复用已下载模型

重新登录同一服务器后，进入**原来那个仓库目录**，激活已有虚拟环境并直接运行：

```bash
cd /你在服务器上的路径/patient-simulator
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py
```

如果用的是 Qwen3-8B，最后一行仍需添加配置：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --config config.qwen3-8b.json
```

**不需要重新安装环境，不需要重新下载模型，也不需要登录 Hugging Face 或提供 token。** 保留 `.venv/` 和 `models/` 即可。每次运行会重新从第 1 轮开始一个新实验，创建新的输出目录。

推理加载使用 `local_files_only=True`。如需显式让 Hugging Face 库进入离线模式，可在这一条运行命令前增加 `HF_HUB_OFFLINE=1`：

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python run_dialogue.py
```

离线选项只用于推理；首次下载模型时不要设置它。如果把服务器上已经完整下载的模型目录复制到另一台机器，应连同 `download_manifest.json` 一起复制，设置相应 `model_dir` 并安装依赖后也可离线运行。Hugging Face 的离线变量说明见[官方文档](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)。

## 6. 在哪里修改 Therapist 和 Patient 的 prompt

直接编辑以下 UTF-8 文本文件，保存后重新运行即可：

| 文件 | 用途 |
| --- | --- |
| `prompts/therapist.txt` | Therapist 的角色、谈话风格、提问策略与限制 |
| `prompts/patient.txt` | Patient 的角色、虚构背景、症状描述、情绪与行为方式 |
| `prompts/opening.txt` | 触发 Therapist 第一条发言的启动指令 |

例如，想换 Patient 案例，就修改 `prompts/patient.txt` 中的背景；想换 Therapist 策略，就修改 `prompts/therapist.txt`。不要把 Patient 私有背景抄到 Therapist prompt 或 opening 中，否则 Therapist 会在对话开始前知道它。

可以另建 prompt 文件，在 JSON 配置中修改 `therapist_prompt`、`patient_prompt`、`opening_prompt` 的路径。每次实验会保存当次 prompt 的快照；以后再编辑原文件，也不会改变已有输出中的快照。

## 7. 配置、轮数与复现

默认运行读取 `config.json`。如果指定 `--config`，下载和运行应使用**同一份配置**。JSON 不支持注释；修改时保留双引号并避免多余逗号。

| 字段 | 作用 |
| --- | --- |
| `model_id` | Hugging Face 模型 ID |
| `revision` | 下载版本，默认 `main`；可改为具体 commit SHA |
| `model_dir` | 服务器本地模型目录 |
| `dtype` | `auto`、`bfloat16` 或 `float16`；`auto` 在当前 GPU 支持 BF16 时使用 BF16，否则使用 FP16 |
| `rounds` | 对话轮数，默认 10；一轮始终是 Therapist + Patient 两条 |
| `max_new_tokens` | 每条回复最多生成的 token 数，默认 256 |
| `max_context_tokens` | 本项目采用的上下文总预算，默认 8192，包含输入与预留的新回复 |
| `temperature`、`top_p`、`top_k` | 采样参数，控制回复的随机性与候选范围 |
| `repetition_penalty` | 重复惩罚系数 |
| `seed` | 随机种子 |
| `therapist_prompt`、`patient_prompt`、`opening_prompt` | 三个 prompt 文件路径 |
| `output_dir` | 实验输出父目录，默认 `outputs/` |

配置里的相对路径都以**该配置文件所在目录**为基准。默认配置放在仓库根目录，所以 `models/`、`prompts/`、`outputs/` 都位于仓库内。若模型要放到数据盘，可将 `model_dir` 改为绝对路径，例如 `/data/models/Qwen3.5-9B`；仍需先用这份配置执行下载。

轮数和种子也可只对本次运行覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --rounds 10 --seed 42
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --config config.qwen3-8b.json --rounds 10 --seed 123
```

改变 `seed` 可生成另一场对话；同一种子不保证在不同 GPU、PyTorch 或 Transformers 版本间逐字一致。

首次下载时，脚本将 `revision` 解析成明确的模型 commit 并记录在 `download_manifest.json` 的 `resolved_revision` 中。`main` 可能随模型作者更新而变化，但推理不会联网检查更新。要固定长期复现实验，可以将配置的 `revision` 改为该清单中的完整 commit SHA，再使用同一配置执行下载，以更新清单中的 `requested_revision`。

运行要求 `download_manifest.json` 存在，且其中的模型 ID、请求版本与当前配置一致。单独修改 `model_id` 或 `revision` 不会替换磁盘上的模型，而是会提示重新下载；换模型或版本时建议同时使用一个新的 `model_dir`。

## 8. 输出在哪里

每次运行都会创建以 UTC 时间命名的独立目录：

```text
outputs/本次运行时间戳/
├── metadata.json
├── dialogue.jsonl
└── dialogue.md
```

- `dialogue.md`：方便直接阅读的完整对话。
- `dialogue.jsonl`：一条生成消息写一行，包含 `round`、`speaker`、`created_at_utc`、`content`、`input_tokens`、`output_tokens`、`finish_reason`，便于后续分析。默认成功完成时有 20 条生成消息。
- `metadata.json`：本次配置与实际运行参数、prompt 快照、运行环境版本、GPU、精度、模型下载清单以及状态，另记录 `completed_rounds` 和 `utterances`。`status` 为 `running`、`completed`、`failed` 或 `interrupted`。

生成消息会逐条写入。因此后续某轮失败时，之前已生成的消息仍保留；中断或失败的实验不应当作完整的 10 轮结果。再次运行会新建实验，不会自动续写之前的对话。

结束原因中，`length` 表示已经达到 `max_new_tokens`，这条回复可能被截断；`eos` 表示模型生成结束标记。若上下文输入加上预留回复超过配置预算，程序报错停止，不会悄悄裁掉历史。要增加回复长度或轮数，应同步评估上下文预算与显存。

## 9. 常见问题

| 情况 | 处理方式 |
| --- | --- |
| 下载提示 401/403 | 先确认模型 ID。默认两个 Qwen 模型为公开模型；若换成 gated/private 模型，检查网页授权是否已批准、服务器是否登录同一账号，以及 token 是否有读取权限。 |
| 下载超时或连接失败 | 检查服务器是否可访问 Hugging Face；恢复网络后重新运行同一下载命令。认证 token 不能解决网络不通。 |
| 找不到本地模型、tokenizer 或下载清单 | 确认下载成功，下载和运行使用同一 `--config`，并且 `model_dir` 指向包含 `download_manifest.json` 的完整模型目录；运行脚本不会自动下载。 |
| Transformers 不认识 `qwen3_5` 或无法导入对应模型类 | 激活正确的 `.venv`，按 `requirements.txt` 安装较新版本，不要混用旧环境。 |
| CUDA 不可用 | 检查服务器驱动、容器 GPU 映射和 CUDA 版 PyTorch 安装；程序要求 CUDA，不会自动改用 CPU。 |
| CUDA out of memory | 确认选中空闲 GPU；降低回复长度、上下文预算或轮数，并使用更大显存的卡或 8B 配置。仅降低上下文预算不会裁剪历史，预算不足仍会报错。 |
| BF16 不支持 | 使用默认 `dtype: "auto"`，或明确改为 `"float16"`。 |
| 回复在句子中间结束 | 查看该条记录的 `finish_reason`；若为 `length`，提高 `max_new_tokens` 或让 prompt 要求更简短的回答。 |
| 想第二次运行却又在下载 | 应直接运行 `run_dialogue.py`，无需再次执行 `download_model.py`。 |

更多依据可查阅 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[Hugging Face 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download) 和 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)。
