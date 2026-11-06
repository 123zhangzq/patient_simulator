# 对话场景 cases

每个目录是一个可以独立运行的场景。`case.json` 同时定义患者资料、对话起点和活动目标；`single_prompt` 与 `patient_agents` 接收同一份人物资料、共同背景和前置对话。运行不需要打开原始 Excel 或参考转录。

## 场景

| id | 起点 | 新生成的交流 | 轮数上限 |
| --- | --- | --- | --- |
| `name_recall` | 提供明确标注为研究者编写的本次 session 学习前缀，没有上周课程或图片 | 回顾姓名，也讨论怎样想到、哪些联想有帮助；根据实际回答提示、解释或转向另一项 | 8 |
| `work_task_compensation` | 提供原记录第 219–230 行作为已发生的对话，患者已描述电子表格笔记 | 听取使用笔记的经历，跟进相关细节、困难或看法；需要时讨论支持或调整 | 10 |
| `name_practice_review` | 提供原记录第 76–136 行作为已发生的对话，患者已报告写姓名、关系和特征联想 | 回顾练习方法与体验，允许自然展开个人感受；不另外安排虚构姓名小测验 | 8 |
| `smart_goal_xc017` | 新活动，无对话前缀；profile 包含清单、日历及提前准备的自述 | 从共同开场讨论来周一个有意义、可行的小目标 | 12 |
| `smart_goal_yc053` | 新活动，无对话前缀；profile 包含兼职工作及曾提出的每周作息、锻炼安排 | 同一主题、开场和治疗师设置，由该患者的想法推进 | 12 |

场景借鉴原始资料，但不要求照搬 TYB 话语或完整课程。原有三个场景都是给定前缀后的继续交流：姓名回顾使用**明确编写的实验条件**，后两个使用真实记录中的前缀。姓名前缀中的简短回应只表示材料已呈现，不预设成功编码或回忆；它不是该患者的真实学习记录。前缀中的患者台词属于输入，不能再作为独立预测目标。

两个 SMART Goal 场景是新的规划谈话，设定发生在来源自述之后，模拟日期沿用来源 session 日期。两者的 `therapist`、`session` 和轮数上限完全相同，首句均为 “Let's think about a small goal for the coming week. What would you like to work on?”。不同之处集中在患者的背景、测量和有限历史自述，不预写来周的具体目标、星期安排或患者台词。SMART 是治疗师协助规划的参考，不是逐项答题脚本；可以保持已有方法，也可以暂不确定计划。

case 描述当前活动、可用材料、交流目标和结束条件。推进方式是可选方向，不是固定的问答顺序或成功路径。共通说话方式由 `prompts/therapist.txt` 管理：根据患者的话回应、解释和追问，不规定每轮只能一句话，也不在每一步重新征询“尝试还是休息”或为凑满轮数扩展新的假设。这里没有按 case ID 执行的专用代码流程；新增 case 可沿用同一接口。

原始转录的一行可能只是同一发言中的片段。姓名回忆可以自然地只说一个名字或半句话，谈熟悉方法和体验时也可以展开；治疗师可以先回应患者的具体内容，再作解释或提出相关问题。这里借鉴这种交流关系，不复制未提供的患者回答，也不预设同样的错误、情绪或结局。

`rounds` 是新生成轮次的上限，不包含前缀。`--rounds` 优先于 case 内的值；case 未填写时使用配置值。活动完成、患者不想继续或进一步澄清已无帮助时，治疗师可以提前结束；连续澄清计数只作提醒，不自动终止。前缀与新生成内容分开记录，避免把给定文本计为生成结果。

## 文件与字段

```text
cases/
└── 一个case目录/
    ├── case.json
    └── reference_dialogue.json
```

| 字段 | 用途 |
| --- | --- |
| `schema_version` | 当前为整数 `1` |
| `id` | 唯一的小写标识，仅允许小写字母、数字和单个下划线，最长 80 字符 |
| `category` | 内容类别，可以复用或新增；同一类别可有多个 case |
| `title`、`summary` | 人工阅读的名称、设计说明和适用边界 |
| `rounds` | 新生成对话的轮数上限；一轮通常包含治疗师和患者各一条发言 |
| `source` | 可选的原工作簿、工作表、Excel 行号及原标识，供追溯，不注入模型 |
| `patient_profile` | 必需的患者资料，包含测量证据、稳定事实、可选初始经历及来源 |
| `session` | 可选的共同背景和前置对话，见下表 |
| `therapist` | 该活动的目标、材料、开场、推进方式和结束条件 |

`session` 的四个字段必须一起提供：

| 字段 | 内容 |
| --- | --- |
| `mode` | `new_activity` 或 `continuation` |
| `shared_context` | 双方共同知道的简短背景；只写实际提供的前提，不包含私有答案或未来结果 |
| `initial_history` | 按时间排序的前置对话。每条为 `speaker`、`content`、`source`；说话人只能为 `Therapist` 或 `Patient` |
| `evaluation_note` | 说明哪些内容已经作为输入、评价从哪里开始，以及哪些结论不能由本场景得出；用于审计 |

`new_activity` 的前置对话为空。`continuation` 的前置对话非空，从治疗师开始，双方交替，以患者结束；新的 `opening_question` 承接这个起点。相邻同一说话人的原始碎片可以按顺序合并，不改写其内容。真实前缀的每条 `source` 指向已核对的原始位置；编写的前缀必须明确标注为 authored simulation setup，不能冒充原始记录。来源字段供审计，不作为发言送进模型。

共同背景可以保留前缀已明确建立的熟悉习惯，例如“患者曾报告用电子表格记新任务”。这并不证明方法总能成功，也不意味着后面所有记忆细节都应随时可用。具体前置发言仍由动态模块编码、保存和提取；baseline 获得同一前缀并使用其完整历史。不能只给一种方法额外经历。

`therapist` 字段如下：

| 字段 | 用途 |
| --- | --- |
| `context` | 当前活动如何开始，已有何种经历或材料 |
| `objectives` | 本次希望弄清或完成什么 |
| `materials` | 可用材料、参考答案、可选例子及其适用范围 |
| `opening_question` | 第一条新生成的治疗师发言，承接前缀或开始新活动；最后聚焦一个主要问题 |
| `conversation_plan` | 根据患者反应推进，不是必须逐轮照读的脚本 |
| `response_guidance` | 本场景特别需要注意的澄清、提示、事实边界和结束方式 |
| `completion_criteria` | 什么情况下已有有用结果或应停止，也允许未解决、没有改善或拒绝 |
| `reference_utterances` | 可选的历史治疗师示例；不是已经发生的台词，也不是必须照读的脚本 |

`reference_dialogue.json` 完整保留原始选段，仅作追溯。运行时不读取该文件。制作 case 时明确选入 `session.initial_history` 的前缀会进入双方输入；明确整理进 `patient_profile.known_facts` 的有限历史自述只进入患者输入。其余参考内容不可见。引用整个参考文件、只把文件放在同目录，均不会自动建立患者经历。

## 怎样避免空白经历与答案泄漏

在制作 case 时先决定采用哪一种起点：

1. **现场建立经历**：先呈现材料或实际进行练习，再询问回忆与体验。学习暴露相同不代表强制编码、理解或回忆成功。
2. **真实前缀续写**：提供切点之前已经发生的对话，只评价切点之后的新生成。不能把已提供事实的复述算作独立预测。
3. **明确的模拟前史**：如果研究设计需要另写一个经历条件，应清楚标注为研究者设定，不冒充该真实患者已知经历，也不能据其与原回答相似就宣称复现成功。

资料中没记录、明确未做过、发生过但没编码、编码后暂时取不出，是不同状态。缺少场景前史应在制作 case 时补齐或改用现场活动，不能让患者通过无限要求重复来承担这个缺口。

给治疗师提供一个来源明确的前缀，不等于把所有历史治疗师判断都当成事实。早先的夸奖、假设和引导问题仍可能不准确。患者此前说过的话是既有报告，也不自动证明动作完成、策略有效或临床改善。

姓名练习中的资料编码是去标识标签，不应拿来充当真实人名刺激。新增材料可用明确标注的虚构名字；没有图片时不要声称展示过人脸。没有写字工具时，“愿意写下”只是意图，不能记成已经产生了笔记。

## 人物资料及证据

患者身份通过原始说话人编码与 profile 中的 SS 角色核对。人口学来自 `Home Visit Stroke Project.xlsx` 的 `Home Visit 1`；年龄保留基线记录，不推算为模拟当天年龄。评分取有明确日期且早于模拟日期的 `Home Visit 2`。

| case | 患者 | 模拟日期 | 评估日期，按源记录 | MoCA 总分 | 背景行 / 评分行 |
| --- | --- | --- | --- | --- | --- |
| `name_recall` | XC030 | 2022-12-23 | 2022-11-23 | 28 | Home Visit 1 第 107 行 / Home Visit 2 第 42 行 |
| `work_task_compensation` | EC011 | 2022-12-01 | 2022-10-26 | 29 | Home Visit 1 第 27 行 / Home Visit 2 第 6 行 |
| `name_practice_review` | YC026 | 2022-12-05 | 2022-07-11 | 25 | Home Visit 1 第 154 行 / Home Visit 2 第 21 行 |
| `smart_goal_xc017` | XC017 | 2022-11-24 | 2022-10-28 | 27 | Home Visit 1 第 94 行 / Home Visit 2 第 11 行 |
| `smart_goal_yc053` | YC053 | 2022-12-16 | 2022-11-22 | 26 | Home Visit 1 第 181 行 / Home Visit 2 第 38 行 |

YC026 的日期按原表保存为 **2022-07-11**，没有猜改。`consent_date` 只是同意日期，不等于未填写的访视评估日期。EC011 的定向力合计与六个分项冲突，因此运行字段为 `null`，原值和冲突保留在 provenance。

`patient_profile` 保留以下字段：

| 字段 | 含义 |
| --- | --- |
| `schema_version`、`participant_id`、`role` | 版本、参与者编号和角色；当前角色为 `stroke_survivor` |
| `assessment_date`、`simulation_date` | 评估与模拟日期；评估日期不能晚于模拟日期 |
| `background` | 年龄、语言、教育及其他有来源的标量背景；未知用 `null` |
| `assessments` | `moca`、`sdmt`、`dass21` 原尺度结果；未知不填 0，不能从筛查分数推定具体职业、个性或生活经历 |
| `known_facts` | 有依据的稳定事实，每条含 `id`、`text`、`source`，可选 `cue_tags` |
| `initial_memories` | 单独提供的过去经历，格式同上；当前为空，原有三个场景由 `session.initial_history` 提供前缀，SMART Goal 的有限历史自述在 `known_facts` 中 |
| `provenance` | 来源路径、哈希、工作表和单元格、原始值及处理说明；不交给输出模型 |
| `unknowns` | 当前证据的边界；区分“评估表未测量”和“前缀已另行提供”，避免互相矛盾 |

本次重构没有更改三个患者的测量值，也没有将参考片段后半段的回答填入 profile。手抖、视力差或较低筛查分项不表示所有日常活动都失败；人物仍可以解释熟悉方法、做当前选择、纠正治疗师或表示不确定。分数到模拟机制的对应仍是可替换的实验假设，见 `patient_agents/policy.json`。

## 原始资料的切分

以下工作簿相对 `LLM_TYB_Data_SMU_Deidentified_20261012/02_whole_session_data/`，均为 `utterances` 工作表，Excel 行号包括标题行：

| case | 工作簿 | 原参考片段 | 本次给定前缀 | 未作为前缀的后续 |
| --- | --- | --- | --- | --- |
| `name_recall` | `Group 2/TYB2_S8_20221223_master_annotated_reviewer2.xlsx` | 42–58；材料核对 877–891 | 研究者另行编写的学习暴露，不取自原患者回答 | 全部原患者回答 |
| `work_task_compensation` | `Group 1/TYB1_S6_20221201_master_annotated.xlsx` | 219–258 | 219–230 | 231–258 |
| `name_practice_review` | `Group 1/TYB1_S7_20221205_master_annotated.xlsx` | 76–169 | 76–136 | 137–169 |
| `smart_goal_xc017` | `Group 1/TYB1_S5_20221124_master_annotated.xlsx` | 425–435 | 无；仅将有限自述整理进患者 profile | 该选段用于历史条件，不作待预测答案 |
| `smart_goal_yc053` | `Group 2/TYB2_S5_20221216_master_annotated_reviewer2.xlsx` | 247–262 | 无；仅将有限自述整理进患者 profile | 该选段用于历史条件，不作待预测答案 |

两个真实前缀场景的新开场经过改编，后续并不是对原转录逐句重建。保留的原后半段可供分析，但不能把新问题下的回答与原句作无条件一一对照。姓名回顾的前缀提供相同 session 内的学习暴露，不代表测得的时间间隔，也不是原先一周延迟或视觉识别的复刻。

SMART Goal 这对 case 用于观察模型是否根据完整 profile 与真实自述作出有依据的不同选择，并非隔离单个认知分数的因果实验。已有清单习惯或曾提出每周安排，不保证新计划已经执行或有效。YC053 的语言测验分项也不等于必须短句、失语或无法规划。先看新输出是否承接人物已有生活方式、区分意图与经历，再比较多个 seed；不要求两个人每轮都明显不同。

## 新增和运行

复制相近目录，修改唯一 `id`、人物资料和活动，明确 `session.mode` 与输入边界。如果更换患者，必须核对完整身份、背景、分数与来源，不能只改编号。运行只读嵌入的数据，原始 Excel 是制作时的证据。

```bash
python run_dialogue.py --list-cases
python run_dialogue.py --category smart_goal --patient single_prompt --dry-run --show-patient
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case smart_goal_xc017 --case smart_goal_yc053 --patient single_prompt --seed 42
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --category smart_goal --patient patient_agents --seed 42
python run_dialogue.py --case name_practice_review --patient patient_agents --dry-run --show-patient
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_practice_review --patient patient_agents
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case name_practice_review --patient single_prompt
```

预览不加载模型或生成对话。实际运行保存配置与 case 快照、生成对话和 `patient_trace.jsonl`；检查模型当轮的 `model_input`，才能知道它实际获得了哪些材料。对比两种方法时同时检查 profile 与前缀，不能只确认它们用了同一个患者编号。
