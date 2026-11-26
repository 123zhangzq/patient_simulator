# 标准化患者对话 Benchmark

这个 benchmark 用多轮对话比较语言模型扮演标准化患者的表现。每个实例包含两份英文角色说明：`therapist.txt` 给治疗师，`patient.txt` 给患者。生成对话后，可以阅读实验记录，也可以单独运行 LLM judge 评价患者表现。

## 项目架构

```text
.                                   # 项目根目录
├── README.md
├── config.json                     # 模型别名、角色分配、重复次数和生成参数
├── .env.example                    # API 密钥文件模板
├── .env                            # API 用户在本机填写，Git 忽略
├── requirements.txt                # OpenRouter 运行依赖
├── requirements-local.txt          # 本地 Qwen 推理依赖（含基础依赖）
├── run_dialogue.py                 # 命令行入口：列出、预览或运行实例
├── download_model.py               # 首次下载本地模型权重
├── evaluation/
│   ├── evaluate.py                 # 对指定的一场会话或一批已完成的会话评分
│   └── judge_prompt.txt            # 可编辑的评分指令
├── instances/                      # 随 benchmark 发布的五个实例
│   ├── manifest.json               # 实例 ID、起点模式、轮数上限和文本哈希
│   ├── name_recall/                # 每个实例都有下列两份完整输入
│   │   ├── therapist.txt
│   │   └── patient.txt
│   ├── name_practice_review/
│   ├── work_task_compensation/
│   ├── smart_goal_xc017/
│   └── smart_goal_yc053/
├── core/                           # 对话运行代码
│   ├── config.py                   # 加载并校验运行配置
│   ├── instances.py                # 读取清单和角色文本，校验内容版本
│   ├── runner.py                   # 播出固定开场、轮流调用模型、保存对话
│   ├── usage.py                    # 汇总 token 用量与 OpenRouter 实际费用
│   └── providers/
│       ├── openrouter.py           # 远程 API 调用
│       └── local_qwen.py           # 本地 Qwen 推理
├── construct_ins/                  # 实例制作工具，仅开发版包含
│   ├── README.md                   # 新增和修改病例的说明
│   ├── templates/                  # 通用治疗师和患者角色指令
│   ├── scenarios/                  # 场景、活动目标和固定开场
│   ├── cases/                      # 患者资料和会话前史
│   ├── references/                 # 制作时核对的来源资料
│   └── build.py                    # 生成或校验 instances/ 中的成品
├── models/                         # 默认的本地权重目录，下载后生成
└── outputs/                        # 每次运行的对话、输入快照和元数据
```

制作病例时，`construct_ins/build.py` 将模板、场景和患者资料整理成两份角色文本。运行已发布的实例时，`run_dialogue.py` 直接读取 `instances/` 的成品，按 `config.json` 为双方选择模型，并将新对话写入 `outputs/`。

以下命令都在项目根目录执行。建议使用 Python 3.11 或更新版本，并在独立的 Python 环境中安装依赖。选择下面一条路线即可。

## 路线一：通过 OpenRouter API 运行（默认）

适合从普通电脑调用远程模型。准备一个有可用额度的 OpenRouter 账号即可，无需 GPU。

### 首次运行

1. 进入包含 `config.json` 和 `requirements.txt` 的项目目录，创建并激活虚拟环境，然后安装 API 依赖。将第一行的示例路径换成你的实际路径。

   **macOS / Linux**

   ```bash
   cd "/path/to/project"
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

   创建环境时可以指定 Python 解释器。例如，macOS 上的 Anaconda 安装在 `/opt/anaconda3` 时，第二行可写成 `/opt/anaconda3/bin/python -m venv .venv`。

   **Windows PowerShell**

   ```powershell
   cd "C:\path\to\project"
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. 在 [OpenRouter 控制台](https://openrouter.ai/settings/keys) 创建 API key。在项目目录中创建 `.env`（可以复制 `.env.example`，已有 `.env` 时直接编辑），填入自己的 key：

   ```dotenv
   OPENROUTER_API_KEY=你的密钥
   ```

   `.env` 只保存在本机，已由 Git 忽略。如果通过系统环境变量设置了同名密钥，程序优先使用环境变量。

3. 运行一个实例：

   ```bash
   python run_dialogue.py --instance name_recall
   ```

   默认配置让治疗师和患者都通过 OpenRouter 调用 `openai/gpt-6-sol`（配置名 `openrouter_default`）。`models` 里另留有 `patient_astra`（`openai/gpt-6-astra`）条目，可用 `--patient patient_astra` 换用。终端会打印双方发言、token 用量、费用和结果目录。

   双方都使用默认采样设置，配置中不传 `temperature` 或 `top_p`。`run.generation.max_new_tokens` 设为 4096，为推理和最终发言共同预留预算；它是每次调用的上限，不是要求每句话达到这个长度。

### 再次运行或更换模型

下次打开终端时，进入同一项目并激活已创建的 `.venv`，程序会读取保存的密钥。macOS / Linux 示例：

```bash
cd "/path/to/project"
source .venv/bin/activate
python run_dialogue.py --instance work_task_compensation
```

Windows PowerShell 使用上面的 `.\.venv\Scripts\Activate.ps1` 激活环境，再执行同一条运行命令。

要固定治疗师、比较不同的 API 患者模型，在 `config.json` 的 `models` 下复制 `patient_astra` 条目，给副本取名（例如 `patient_b`），将其 `model_id` 改为想测试的 OpenRouter 模型 ID。然后运行：

```bash
python run_dialogue.py --instance name_recall --therapist openrouter_default --patient patient_astra --patient patient_b
```

## 路线二：在本地 GPU 上运行 Qwen

这条路线使用支持 CUDA 的 NVIDIA GPU。首次下载权重时需要联网；下载完成后，对话可离线运行，无需 API key。

### 首次运行

1. 激活用于本地推理的 Python 环境，按 [PyTorch 安装指引](https://pytorch.org/get-started/locally/) 安装与 GPU 驱动兼容的 CUDA 版 PyTorch 和 torchvision。然后安装项目的本地推理依赖：

   ```bash
   python -m pip install -r requirements-local.txt
   ```

2. 下载模型。配置中的 `qwen_local` 默认指向 `Qwen/Qwen3.5-9B`：

   ```bash
   python download_model.py --model qwen_local
   ```

   默认保存目录为 `models/Qwen3.5-9B/`。如需使用其他磁盘，请在下载前设置 `config.json` 中的 `models.qwen_local.model_dir`。下载器会将权重、tokenizer 和记录版本的 `download_manifest.json` 保存在同一目录。

3. 让治疗师和患者都使用本地模型：

   ```bash
   python run_dialogue.py --instance name_recall --therapist qwen_local --patient qwen_local
   ```

   两个角色各自使用独立的输入，一次运行只加载一份模型权重。

### 再次运行

保留下载好的模型目录，进入同一 Python 环境，直接运行对话命令。程序从配置的目录加载权重，无需重复安装或下载：

```bash
python run_dialogue.py --instance work_task_compensation --therapist qwen_local --patient qwen_local
```

如果以后都使用本地模型，可将 `config.json` 中的 `roles.therapist` 设为 `"qwen_local"`、`roles.patients` 设为 `["qwen_local"]`。保存后，每次运行只需指定 `--instance`。

## 选择实例和运行设置

下面两条命令只读取数据集，可以在安装模型依赖或配置密钥之前使用：

```bash
python run_dialogue.py --list-instances
python run_dialogue.py --instance name_recall --dry-run
```

`--list-instances` 列出可用实例，`--dry-run` 显示选中的模型、运行设置和双方的完整角色文本。`--instance` 可重复传入，不指定时运行全部实例。重复次数和生成参数在 `config.json` 的 `run` 中设置；`--seed` 和 `--rounds` 可覆盖本次运行的种子和最大轮数。

## 实例与会话记录

| Instance | 会话起点 | 最多新增轮数 |
|---|---|---:|
| name_recall | 本次练习中人工编写的 6 条学习前史 | 8 |
| name_practice_review | 已提供的 16 条交流 | 8 |
| work_task_compensation | 已提供的 6 条交流 | 10 |
| smart_goal_xc017 | 新活动，无前史 | 12 |
| smart_goal_yc053 | 同一个 SMART Goal 场景，另一位患者 | 12 |

固定开场取自 `therapist.txt` 末尾的 `[Opening utterance]`，由程序原样播出。一轮包含 therapist 和 patient 各一条台词。对已给定前史的实例，这些发言已经分别写在双方角色文件中，轮数从固定开场起算。Therapist 可在收尾台词后另起一行写 `[END_SESSION]`。程序移除该标记，让 patient 回应收尾后结束；达到上限的会话另记为 `round_limit`。

每次执行会打印一个独立输出目录：

```text
outputs/<run>/
  batch_metadata.json
  inputs/<instance>/{therapist,patient}.txt
  sessions/<patient-model>/<instance>/repeat_001/
    dialogue.md
    dialogue.jsonl
    metadata.json
```

`dialogue.md` 适合阅读，分开展示给定历史和新对话。`dialogue.jsonl` 只保存新发言，区分固定开场与模型生成，并记录服务端可用的用量信息。`metadata.json` 保存实际轮数、结束原因、采样设置、数据集版本和双方 prompt 的 SHA-256；`inputs/` 保存本次所用的原文。达到轮数上限是运行结果，教育适用性需要另行评定。

### Token 用量与费用

无需额外配置。只在整次 run 结束时打印输入、输出、总 token 和 OpenRouter 费用（USD），包括 therapist、patient 各自用量和总计。固定开场不调用模型，不计入调用次数。

逐次用量保存在 `dialogue.jsonl`，每场和整批汇总分别保存在 `metadata.json`、`batch_metadata.json` 的 `usage` 中，包括 `total`、`by_role` 和 `by_model`。

费用使用 [OpenRouter 返回的 `usage.cost`](https://openrouter.ai/docs/cookbook/administration/usage-accounting)，是账户实际扣除的美元额度，不用 token 单价估算，不包含充值手续费。输入 token 包括每次发送的角色说明和对话历史；输出 token 按 API 返回值记录，推理 token、缓存命中 token 另存为明细，不重复加到总量中。

没有返回的字段显示 `N/A`；仅部分调用返回费用时显示已知费用小计和覆盖调用数。本地 Qwen 只统计 token，费用显示 `N/A`，不估算硬件开销。汇总覆盖已记录的模型回复；运行中断时仍保存已有用量，但没有收到有效回复的请求是否收费需以服务端账单为准。

## 评价已生成的对话

在 `config.json` 中用 `roles.judge` 选择评分模型。默认别名 `openrouter_default` 对应 `openai/gpt-6-sol`，使用同一份 `.env` 中的 OpenRouter 密钥。评审的生成参数单独放在 `evaluation.generation`。

每次通过 `--input` 指定要评的内容。将下面的 `RUN_ID` 替换为生成对话时打印的运行目录名。加 `--dry-run` 可在调用模型前核对会话清单和输入快照：

```bash
python evaluation/evaluate.py --input outputs/RUN_ID --dry-run
python evaluation/evaluate.py --input outputs/RUN_ID
```

只评一场时，传该场的 `repeat_001/` 目录或其中的 `dialogue.jsonl`，例如：

```bash
python evaluation/evaluate.py --input outputs/RUN_ID/sessions/openrouter_default/name_recall/repeat_001/dialogue.jsonl
```

整批评价只处理已完成的会话。Judge 读取该次运行保存的两份角色文本和新对话，只评生成的患者发言。给定前史、固定开场和治疗师发言帮助判断上下文。模型别名和 token 用量不会传给 judge。

`evaluation/judge_prompt.txt` 按 Expert Review Questionnaire v1.0 逐项评分：A1–A5 评真实性，B1–B5 评训练价值，每场对十条陈述各输出 1–5 赞同度、简短理由和对应轮次；A2、A3、B4、B5 为负向陈述，按原文打分、保留原始方向。可直接编辑该文件后对原对话重评。每次评价在原运行目录的 `evaluations/<评价时间>/` 保存独立的 `results.jsonl`、judge prompt 快照和配置记录。未按格式返回的评价标为 `invalid_rating`，没有分数。

评价只在本次运行结束时打印 token、费用汇总，与生成对话的费用分开统计。逐次用量和汇总分别保存在 `results.jsonl` 的逐次 `usage` 和评价 `metadata.json` 的汇总 `usage` 中。返回了内容但评分格式无效的调用仍计入用量。

## 调整实例

如需试验自己的角色说明，复制整个 `instances/` 目录，编辑相应实例的 TXT，然后在运行命令中加上 `--instances-dir 副本路径 --allow-edited-prompts`。结果会标记输入已修改，并保存实际使用的文本及其哈希。

如使用带有制作工具的开发版，新增场景和重新生成实例的步骤见 `construct_ins/README.md`。
