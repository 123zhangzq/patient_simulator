# Therapist–Patient 对话实验

## 1. 简介

在 **Linux + NVIDIA GPU** 服务器上，本地加载一份 Qwen 权重，按 `cases/` 中的场景逐个模拟 Therapist–Patient 对话。case 同时定义患者资料、活动起点、已有经历和治疗师的推进方式。每轮由 Therapist 先说、Patient 回复，两者共享同一份模型。场景参考真实认知训练交流，但允许合理适配，不要求复刻 TYB 的课程流程或原患者回答。

首批包含姓名学习与回忆、工作任务补偿策略、姓名练习回顾三个 case，分别设置最多 8、10、8 轮新对话。姓名任务从当前教学开始；另外两例先提供明确选取的真实对话前缀，再继续交流。活动可以提前结束，不要求凑满轮数。模型在服务器首次部署时下载，以后复用服务器上的模型目录。本项目按 Linux GPU 环境编写；列出 case、预览 prompt 和运行功能测试不需要 GPU。

## 项目结构与运行方式

主要源码、配置与 prompt 文件如下：

```text
.
├── README.md
├── requirements.txt
├── config.json
├── config.qwen3-8b.json
├── download_model.py
├── settings.py
├── run_dialogue.py
├── cases/
│   ├── README.md
│   ├── name_recall/
│   │   ├── case.json
│   │   └── reference_dialogue.json
│   ├── work_task_compensation/
│   │   ├── case.json
│   │   └── reference_dialogue.json
│   └── name_practice_review/
│       ├── case.json
│       └── reference_dialogue.json
├── core/
│   ├── cases.py
│   ├── profiles.py
│   ├── patient_setup.py
│   ├── llm.py
│   ├── therapist.py
│   ├── interfaces.py
│   ├── registry.py
│   └── runner.py
├── baselines/
│   └── single_prompt/
│       └── patient.py
├── patient_agents/
│   ├── patient.py
│   ├── static.py
│   ├── context.py
│   ├── response.py
│   └── policy.json
└── prompts/
    ├── therapist.txt
    ├── therapist_control.json
    ├── opening.txt
    ├── baselines/
    │   └── single_prompt/
    │       └── patient.txt
    └── patient_agents/
        └── response.txt
```

根目录的 `run_dialogue.py` 是统一命令行入口；`download_model.py` 继续负责下载，`settings.py` 继续负责配置与路径处理。两份 JSON 配置也保留在根目录。

`core/cases.py` 负责发现、校验和筛选 case，以及合成治疗师 prompt。`core/llm.py` 统一加载模型并生成回复，整个批次只加载一次。`core/therapist.py` 定义 Therapist；`core/interfaces.py` 定义角色之间共用的接口；`core/registry.py` 根据方法名选择 Patient 实现；`core/runner.py` 统一控制各 case 的轮次、公开对话历史、随机种子和结果保存。

两种 Patient 方法使用同一个 case 中嵌入的真实患者 profile，以及相同的 `session.shared_context` 和 `session.initial_history`，可以在相同场景、模型和种子下比较：

| 方法 | 患者如何获得信息 |
| --- | --- |
| `single_prompt` | 私有 prompt 加入患者资料与共同情境，每次回复接收给定前缀及完整新增对话，调用模型一次 |
| `patient_agents` | 静态编译器提供持续可见的人物卡，动态控制器处理前缀和新输入的编码与提取，输出模块据此调用模型一次 |

`patient_agents/static.py`、`context.py`、`response.py` 分别承担这三个职责，由 `patient.py` 组合。首版静态和动态模块使用可替换的规则，只有输出模块调用 LLM；三个职责不意味着加载三个模型或每轮调用三个 LLM。治疗师和患者仍共享一份模型权重。扩展 Patient 方法时，通过注册和选择方法复用同一个 runner，不需要复制整套对话流程。

Patient 接口位于 `core/interfaces.py`。两个实现都接收共享 `llm`、私有 prompt，以及 `patient_profile`、`seed`、`options` 和 `prompts`。runner 先调用一次 `initialize_history(public_history)`：基线在之后的模型调用中使用完整前缀；`patient_agents` 按顺序处理前缀的治疗师输入和患者自述，不生成新台词。相同的学习输入不意味着强制相同的编码结果。

随后 `respond(public_history)` 接收仅含 `speaker`、`content` 的公开消息序列，返回 `GenerationResult`，基本字段为 `content`、`input_tokens`、`output_tokens`、`finish_reason`。runner 传入不可修改的历史快照，只将 `content` 作为公开台词加入后续对话。`patient_agents` 不重复处理已初始化的前缀，也不把完整历史转交输出 LLM。治疗师另外返回可选控制字段 `dialogue_action`、`stop_reason`，它们不会进入患者听到的台词。注册表仅在选中方法后导入对应类。

case 加载和批量运行不新增依赖。已有服务器的环境、`models/` 和 `outputs/` 可以继续使用。默认 prompt、角色台词、终端日志和生成的对话文件标题均使用英语；README 使用中文。每个 case 的 `patient_profile` 保存测量资料、真实已知事实、初始记忆、未知信息及来源。运行不读取 Excel 或 `reference_dialogue.json`；只有显式整理进 `case.session.initial_history` 的前缀台词会作为已有对话输入两种方法。前缀中的原患者自述是条件，不能再作为待预测答案评分。

`session.mode` 支持 `new_activity`（无前缀的新活动）和 `continuation`（有前缀的继续交流）。`shared_context` 只放双方已知的情境与前缀已确立的熟悉习惯概要，不能把待回忆的姓名答案放在这里长期可见。`evaluation_note` 和每条前缀的 `source` 留在 case 快照供审计，不作为模型输入。未写 `session` 的旧 case 按无前缀的新活动运行。

当前机制是实验原型。动态模块按标点拆分输入，用词语重合和提示关键词近似检索；注意容量、编码概率、保留率、提取概率及情绪变化规则尚未临床校准。MoCA 总分不直接设定记忆，SDMT 与 DASS 保留为证据，不把它们换算成反应速度或情绪诊断。语言表达默认 `unknown`，不再根据 MoCA language=2 自动设成 `reduced`；需要限制表达时用显式实验覆盖。输出模型仍可能违反信息边界，需结合每轮 trace 检查实际表现。

人物卡中的稳定背景和已知事实持续可见，不再靠当前问题的词面重合筛选。任务材料和对话经历进入注意、编码和提取机制。最近的已编码信息形成工作记忆；近期患者自述完整保留，避免当前选择或计划因句子截取而消失。较早信息按提取概率和每轮保留率决定是否可及，`working_memory_turns` 控制近期窗口。预置的 `patient_profile.initial_memories` 属于过去经历，第一轮也要经过提取；`session.initial_history` 则逐轮处理实际给定的前缀，二者用途不同。类别与选择提示分项保留在证据中，提示收益仍使用统一的可调 `cue_bonus`，尚未根据这些分项做个体化换算。

动态输出的 `information_access` 区分当前输入是否完整、是否返回了早先发言，不宣称这些发言足以回答当前问题。没有提取出历史不代表活动从未发生，也不自动触发“不理解”或挫败状态。人物卡、给定经历、当前输入与输出 prompt 共同约束表达；语义判断仍由输出模型完成，并非已经实现了可靠的事实验证器。

## 2. 模型选择与服务器要求

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

## 4. 按 case 运行对话

先查看 case 或预览完整治疗师 prompt，这两种操作不加载模型，也不导入 PyTorch/Transformers：

```bash
python run_dialogue.py --list-cases
python run_dialogue.py --case name_recall --dry-run
python run_dialogue.py --case name_recall --patient patient_agents --dry-run --show-patient
```

默认递归寻找 `cases/` 中所有名为 `case.json` 的文件，按相对路径排序。`reference_dialogue.json` 只供追溯，运行不会读取它；从中明确选取并嵌入 case 的前缀会按 `session` 配置使用。普通 `--dry-run` 展示治疗师 prompt、共同情境与前缀、患者 ID、`patient_profile_sha256`、`session_sha256` 及治疗师控制文本；加 `--show-patient` 才同时展示患者资料及方法配置，仍不会生成对话。三个 case 的内容说明、来源定位和新增方法见 [cases/README.md](cases/README.md)。

运行三个职责模块，或在同一 case 上运行基线：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --patient patient_agents --seed 42
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --patient single_prompt --seed 42
```

两个方法读取同一个嵌入 profile 与 session，分别记录 `patient_profile_sha256` 和 `session_sha256`。基线直接看到 profile 和完整公开历史；`patient_agents` 的输出 LLM 看到稳定人物卡、共同情境和动态模块给出的本轮信息。比较时同时保持 case、模型、轮数上限和配置一致，并使用多个种子观察变化。自由对话中的治疗师会随患者回答改变下一步，因此相同配置不保证相同后续问题；要隔离患者机制差异，还需使用相同刺激的对照。

注意与记忆消融只适用于 `patient_agents`：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --patient patient_agents --no-attention
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --patient patient_agents --no-memory
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --patient patient_agents --no-attention --no-memory
```

`--no-attention` 关闭当前输入和近期工作记忆的容量截取，并使用 preserved 档的回复单元预算；记忆阶段的编码失败仍然有效。`--no-memory` 关闭编码失败、记忆衰减、提取失败及提取数量限制；仍只接收经过注意模块的信息，且不会补出从未输入的过去经历。两者均不关闭静态表达规则，也不把方法替换成基线。

用默认 Qwen3.5-9B 顺序运行全部 case：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py
```

只运行一个 case，或只运行一个类别：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_practice_review
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --category practice_review
```

不传 `--patient` 时，默认选择 `single_prompt` 基线。可重复 `--case` 选择多个；多个条件同时提供时取交集：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --case work_task_compensation
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --cases-dir /path/to/my_cases
```

`--cases-dir` 也接受单个 case JSON 文件。该参数的相对路径以当前工作目录为基准；不传时始终定位到本仓库的 `cases/`，从其他目录启动也有效。

Qwen3-8B：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --patient single_prompt --config config.qwen3-8b.json
```

`CUDA_VISIBLE_DEVICES=0` 选择服务器上编号为 0 的 GPU。要使用编号为 2 的 GPU，就改成 `CUDA_VISIBLE_DEVICES=2`；程序会把选中的那张卡视为当前进程的第一张 GPU。不要同时运行多个实验占满同一张卡。

运行时按顺序生成：

```text
第 1 轮：Therapist → Patient
第 2 轮：Therapist → Patient
...
第 N 轮：Therapist → Patient
```

`prompts/opening.txt` 是启动指令模板，其中 `{opening_question}` 由 case 提供；模板本身不计入公开台词。第一条新治疗师发言直接使用 `case.therapist.opening_question`，不调用模型，记录的输入和输出 token 数均为 0。后续治疗师调用返回 `{"utterance": "...", "action": "continue|clarify|close"}`，只有 `utterance` 进入公开历史。控制指令和兜底结束语保存在 `prompts/therapist_control.json`，不增加额外的裁判 agent 或模型调用；不符合格式的输出会报错并保存 trace。

公开历史由明确给定的前缀和本次新增台词组成。治疗师和 `single_prompt` 使用完整历史，其中自身发言映射为 `assistant`，对方发言映射为 `user`；`patient_agents` 的输出调用使用人物卡与本轮结构化 context。患者私有 profile、内部状态和 trace 不进入治疗师输入；双方都能看到的 `shared_context` 及前缀是有意共享的条件。

轮数上限的优先级为 `--rounds` > case 的 `rounds` > 配置文件的 `rounds`。一轮是 Therapist 与 Patient 的两条新台词；前缀不计入新轮数。治疗师标记 `close` 后，保留本轮患者的一次回应，然后结束。达到轮数上限也会停止。`completion_criteria` 用来指导模型的结束判断，不是独立的成功评分器；`status=completed` 只表示程序正常完成。

默认最多连续两次澄清。模型标记 `clarify` 时计数；相邻治疗师问句忽略大小写、标点和空白后相同，也会自动计数，即使模型错误标成 `continue`。之后控制指令要求提供具体线索、示范、转入有依据的活动步骤或结束；第三次连续澄清触发可编辑的兜底结束语。换一种说法的语义重复仍依赖模型正确标记，不能保证全部识别。普通结束记录 `stop_reason=therapist_close`，澄清超限为 `clarification_limit`，轮数耗尽为 `round_limit`。这些原因都不等于记忆成功或策略有效。

同批所有 case 共享模型权重，但每个 case 都新建 Therapist、Patient，初始化自己的 session 前缀，并重置随机种子。上一段模拟和患者私有状态不会带入下一段。两个方法每条新 Patient 回复都调用模型一次。

代码关闭模型的 thinking 输出，便于直接获得角色台词。默认双方使用英语对话。每个角色共享模型权重，但 prompt 和历史角色映射彼此独立；修改 prompt 不会训练或修改权重。

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

**不需要重新安装环境，不需要重新下载模型，也不需要登录 Hugging Face 或提供 token。** 保留 `.venv/` 和 `models/` 即可。每次运行会创建一个新批次，并按筛选结果逐个从第 1 轮开始模拟；不传 `--patient` 仍使用 `single_prompt`。

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
| `prompts/therapist_control.json` | 治疗师 JSON 输出约定、澄清上限提示、最后一轮指令与兜底结束语 |
| `prompts/baselines/single_prompt/patient.txt` | `single_prompt` 基线 Patient 的通用扮演与证据约束；个体资料来自 case |
| `prompts/patient_agents/response.txt` | 输出模块如何根据可见 context 说话及承认不确定性 |
| `patient_agents/static.py` | profile 到模拟假设及输出约束的编译规则 |
| `patient_agents/context.py` | 输入接收、记忆编码、提取和逐轮交互状态 |
| `patient_agents/policy.json` | 默认容量、概率、阈值、提示关键词等实验参数 |
| `prompts/opening.txt` | 通用启动模板，必须恰好包含一个 `{opening_question}` |
| `cases/<case目录>/case.json` | 活动主题、患者 profile、session 共同背景与前缀、治疗师目标与开场、轮数上限 |
| `cases/<case目录>/reference_dialogue.json` | 原始逐行对话与 Excel 定位，供核对；运行不读取，选用前缀须显式整理到 case |

默认 prompt 模板使用英语，当前三个 case 的患者语言也均为英语。Patient 根据 case 中的语言字段输出；增加其他语言的 case 时也应相应调整治疗师语言。修改所有场景共用的治疗师行为时编辑 `prompts/therapist.txt`，控制格式与结束语在 `therapist_control.json`；修改某个场景时分别编辑 case 的 `session` 和 `therapist`。Patient 侧按选定方法使用对应 prompt，不要把私有背景或未见目标回答放进治疗师字段。静态编译与动态控制没有隐藏的 LLM prompt；修改机制应编辑对应模块或 policy。

合成分工如下：

```text
prompts/therapist.txt + case.therapist + session.shared_context
    → Therapist system prompt
prompts/opening.txt 中的 {opening_question} + case.therapist.opening_question
    → 启动指令；首条台词直接使用 opening_question
prompts/therapist_control.json + 当前轮数与澄清计数
    → 后续 Therapist 调用的输出格式与推进约束
配置指定的 patient_prompt + case.patient_profile + session.shared_context
    → single_prompt 的 Patient system prompt
case.patient_profile + patient_agents/policy.json + config.patient_agents
    → 持续可见的人物卡 + 实验参数
session.initial_history + 新输入
    → baseline 完整历史；patient_agents 顺序编码与动态 context
prompts/patient_agents/response.txt + shared_context + 人物卡 + 动态 context
    → patient_agents 的 Patient 输出调用
```

`summary`、来源字段、`session.evaluation_note` 和完整的 `reference_dialogue.json` 不参与上述合成。可选的 `therapist.reference_utterances` 只是表达示例，不代表已经发生的对话；`session.initial_history` 才是明确给定的已有交流。姓名回忆 case 从当前文字教学开始，不再假定患者上周已经学过，也不声称有照片。两个 continuation case 只评价给定前缀之后的生成，不能把前缀事实的复述当作独立预测。切点和材料见 [cases/README.md](cases/README.md)。

可以另建 prompt 文件，在 JSON 配置中修改 `therapist_prompt`、`patient_prompt`、`opening_prompt` 的路径；`patient_agents` 的输出 prompt 由 `patient_agents.response_prompt` 指定。自定义 opening 模板同样需要 `{opening_question}`。每个 case 的输出会保存 prompt、profile、有效 policy 和配置快照；以后编辑原文件不会改变已有快照。

兼容旧配置：若 `patient_prompt` 仍为旧默认路径 `prompts/patient.txt`，只有在它解析为该配置目录下的这一旧默认位置、且旧文件不存在时，`settings.py` 才会回退到新路径 `prompts/baselines/single_prompt/patient.txt`。如果旧文件仍存在，例如你保留了自己的 Patient prompt，就继续读取旧文件。其他自定义路径缺失时仍会报错，不会被自动替换。

## 7. 配置、轮数与复现

默认运行读取 `config.json`。如果指定 `--config`，下载和运行应使用**同一份配置**。JSON 不支持注释；修改时保留双引号并避免多余逗号。

| 字段 | 作用 |
| --- | --- |
| `model_id` | Hugging Face 模型 ID |
| `revision` | 下载版本，默认 `main`；可改为具体 commit SHA |
| `model_dir` | 服务器本地模型目录 |
| `dtype` | `auto`、`bfloat16` 或 `float16`；`auto` 在当前 GPU 支持 BF16 时使用 BF16，否则使用 FP16 |
| `rounds` | case 未指定时的新对话轮数上限，默认 10；一轮是 Therapist + Patient 两条新台词，不包括前缀 |
| `max_new_tokens` | 每条回复最多生成的 token 数，默认 256 |
| `max_context_tokens` | 本项目采用的上下文总预算，默认 8192，包含输入与预留的新回复 |
| `temperature`、`top_p`、`top_k` | 采样参数，控制回复的随机性与候选范围 |
| `repetition_penalty` | 重复惩罚系数 |
| `seed` | 随机种子 |
| `therapist_prompt`、`patient_prompt`、`opening_prompt` | 三个 prompt 文件路径 |
| `patient_agents.response_prompt` | `patient_agents` 输出模块的 prompt 路径 |
| `patient_agents.policy_path` | 默认读取 `patient_agents/policy.json`；可指向另一个完整 policy 文件 |
| `patient_agents.attention_enabled`、`patient_agents.memory_enabled` | 是否启用注意过滤与记忆损伤机制；命令行消融开关可关闭它们 |
| `patient_agents.parameters` | 按名称覆盖 policy 中已有数值参数，例如 `retrieval_limit`；空对象使用默认值 |
| `patient_agents.traits` | 显式覆盖 `attention`、`memory`、`language` 的实验分档，记录为假设，不修改测量原值 |
| `output_dir` | 实验输出父目录，默认 `outputs/` |

配置里的相对路径都以**该配置文件所在目录**为基准。默认配置放在仓库根目录，所以 `models/`、`prompts/`、`outputs/` 都位于仓库内。若模型要放到数据盘，可将 `model_dir` 改为绝对路径，例如 `/data/models/Qwen3.5-9B`；仍需先用这份配置执行下载。

`patient_agents.parameters` 只接受 policy 已有的参数名；概率和每轮保留率取 0–1，数量和阈值使用正整数。注意和记忆分档可选 `preserved`、`reduced`、`markedly_reduced`、`unknown`；语言可选 `preserved`、`reduced`、`unknown`。这些名称是程序中的实验分组，不是临床诊断。修改分档阈值、检索上限或词表后，应保存对应配置并检查 trace，不能把示例默认值报告为已验证的患者认知参数。

默认 policy 版本为 `heuristic-v2`。稳定人物事实不再按词面检索，语言分数不再自动决定表达分档，因此已删除 `known_fact_limit` 和 `language_preserved_min`。使用旧自定义 policy 或参数覆盖时，请同步删除这两个参数并与当前 policy 字段保持一致；无效字段会明确报错，不会被静默忽略。语言默认 `unknown`，若研究设计需要其他分档，可在 `patient_agents.traits.language` 显式设置并保留依据。

方法、轮数上限和种子也可只对本次运行指定。`--rounds` 会覆盖所有选中 case 的上限；原有 `--config`、`--patient` 和 `--seed` 保持可用：

```bash
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_recall --rounds 10 --seed 42
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --config config.qwen3-8b.json --rounds 12 --seed 123
```

随机种子在加载模型前设置，并在每个 case 创建角色前重置；同一 case 内 Therapist 和 Patient 的模型采样按生成顺序使用同一全局随机流。动态控制器另用该 seed 初始化自己的随机数生成器，记录编码与提取决策。改变 `seed` 可生成另一场对话。各 case 的随机流不随其他 case 是否被选中而改变；同一种子不保证不同方法产生相同轨迹，也不保证在不同 GPU、PyTorch 或 Transformers 版本间逐字一致。

首次下载时，脚本将 `revision` 解析成明确的模型 commit 并记录在 `download_manifest.json` 的 `resolved_revision` 中。`main` 可能随模型作者更新而变化，但推理不会联网检查更新。要固定长期复现实验，可以将配置的 `revision` 改为该清单中的完整 commit SHA，再使用同一配置执行下载，以更新清单中的 `requested_revision`。

运行要求 `download_manifest.json` 存在，且其中的模型 ID、请求版本与当前配置一致。单独修改 `model_id` 或 `revision` 不会替换磁盘上的模型，而是会提示重新下载；换模型或版本时建议同时使用一个新的 `model_dir`。

## 8. 输出在哪里

每次运行都会创建以 UTC 时间命名的独立目录：

```text
outputs/本次运行时间戳/
├── batch_metadata.json
├── name_recall/
│   ├── metadata.json
│   ├── initial_history.json
│   ├── dialogue.jsonl
│   ├── patient_trace.jsonl
│   ├── therapist_trace.jsonl
│   └── dialogue.md
├── work_task_compensation/
│   └── ...
└── name_practice_review/
    └── ...
```

- `dialogue.md`：先展示给定前缀并标明“不由本次生成、不评分”，再展示新对话，方便连续阅读。
- `initial_history.json`：单独保存两种方法收到的前缀 `speaker` / `content`；新活动保存空列表。前缀的来源仍可在 metadata 中的完整 case 快照查到。
- `dialogue.jsonl`：一条新台词写一行，包含 `round`、`speaker`、`created_at_utc`、`content`、`input_tokens`、`output_tokens`、`finish_reason`。不包含给定前缀；正常结束后有 `completed_rounds` 的两倍条消息。首条固定开场的 token 数为 0。
- `metadata.json`：配置、轮数上限、prompt、case/profile/session 快照、源 case 文件路径、环境及模型信息。`patient_profile_sha256` 标识完整规范化 profile；`session_sha256` 标识模型可见的 mode、共同情境和前缀，不包含来源注释与评价说明。`patient_setup` 保存方法资源，`patient_initialization` 保存初始化结果（`patient_agents` 包括编译结果、有效 policy 和前缀处理 trace）。另有 `initial_history_utterances`、`completed_rounds`、`utterances`、`stop_reason` 和 `status`（`running`、`completed`、`failed`、`interrupted`）；完成轮数和台词数只统计新增部分。profile 内的来源工作簿哈希用于追溯，不代表运行时重新读取或校验了 Excel。
- `patient_trace.jsonl`：每次新 Patient 调用的内部记录。`patient_agents` 记录输入单元、接收与遗漏、编码、提取概率及结果、信息访问状态和交给输出模块的 context；两种方法均记录实际 `model_input`，失败时也尽可能保留调用输入。外层 `round` 是新增轮号，内部 context 的 `turn` 包括已处理的前缀轮次。源前缀中的患者自述会标记 `from_initial_history`；患者刚生成的话仅记录为“说过的话”，不会自动升级为真实个人事实。trace 不进入公开对话或治疗师 prompt。
- `therapist_trace.jsonl`：每次新 Therapist 调用的记录。首轮标记 `source=case_opening`，没有模型输入；后续记录 `model_input`、模型原始输出、请求动作、最终动作、澄清计数及可能的结束覆盖。用于区分模型自主收束、澄清超限与轮数上限。
- `batch_metadata.json`：整个批次的状态，以及每个 case 的输出位置、轮数上限、完成进度、结束原因和错误。未开始的 case 标为 `pending`。

生成消息会逐条写入。某个 case 失败或被中断时保留已有对话和失败状态，并停止批次；之前已完成的 case 保留，后续 case 不运行。再次运行会新建批次，不自动续写。

单条模型输出的 `finish_reason` 与整段活动的 `stop_reason` 不同：`length` 表示达到 `max_new_tokens`，`eos` 表示生成结束（固定开场也使用此值）。治疗师 JSON 若因截断或格式问题无法解析，程序报错并保留 trace。若上下文输入加上预留回复超过配置预算，程序报错停止，不会悄悄裁掉历史；较长的给定前缀也占用上下文预算。要增加回复长度或轮数，应同步评估上下文预算与显存。

无需 GPU 的功能验证（Python 3.11+）：

```bash
python -m unittest discover -s tests -v
```

测试覆盖 case/profile/session 校验、前缀输入与审计信息隔离、无模型预览，以及用模拟模型验证批量运行、提前结束、澄清上限和失败记录。机制测试检查注意、记忆、前缀只处理一次、稳定人物信息持续可见与随机种子控制。它们验证程序行为，不验证真实模型的对话质量、认知映射的临床效度或 GPU 推理。本轮改动尚未运行真实 LLM，需在服务器检查自然表达、动作标记是否准确以及是否仍会编造经历。

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
