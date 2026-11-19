# 标准化患者对话 Benchmark

这个 benchmark 用多轮对话比较语言模型扮演标准化患者的表现。每个实例包含两份英文角色说明：`therapist.txt` 给治疗师，`patient.txt` 给患者。运行结束后可以查看完整对话和实验记录。

在包含 `config.json` 的项目目录中执行以下命令。建议使用 Python 3.11 或更新版本，并在独立的 Python 环境中安装依赖。选择下面一条路线即可。

## 路线一：通过 OpenRouter API 运行（默认）

适合从普通电脑调用远程模型。准备一个有可用额度的 OpenRouter 账号即可，无需 GPU。

### 首次运行

1. 在 Python 环境中安装 API 依赖：

   ```bash
   python -m pip install -r requirements.txt
   ```

2. 在 [OpenRouter 控制台](https://openrouter.ai/settings/keys) 创建 API key。把项目里的 `.env.example` 复制为同目录下的 `.env`，在其中填入自己的 key：

   ```dotenv
   OPENROUTER_API_KEY=你的密钥
   ```

   `.env` 只保存在本机，已由 Git 忽略。如果通过系统环境变量设置了同名密钥，程序优先使用环境变量。

3. 运行一个实例：

   ```bash
   python run_dialogue.py --instance name_recall
   ```

   默认配置让治疗师通过 OpenRouter 调用 `openai/gpt-6-sol`（配置名 `openrouter_default`），患者调用 `openai/gpt-6-astra`（配置名 `patient_astra`）。终端会打印双方发言和结果目录。

   这两个模型使用默认采样设置，配置中不传 `temperature` 或 `top_p`。`run.generation.max_new_tokens` 设为 4096，为推理和最终发言共同预留预算；它是每次调用的上限，不是要求每句话达到这个长度。

### 再次运行或更换模型

进入同一项目并激活同一 Python 环境，直接运行下一个实例。程序会自动读取保存的 `.env`：

```bash
python run_dialogue.py --instance work_task_compensation
```

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

## 调整实例

如需试验自己的角色说明，复制整个 `instances/` 目录，编辑相应实例的 TXT，然后在运行命令中加上 `--instances-dir 副本路径 --allow-edited-prompts`。结果会标记输入已修改，并保存实际使用的文本及其哈希。

如使用带有制作工具的开发版，新增场景和重新生成实例的步骤见 `construct_ins/README.md`。
