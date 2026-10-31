# 对话场景 cases

每个目录代表一个可独立运行的对话场景，并嵌入本例的患者 profile。`category` 是内容类别，同一类别可以有任意多个 case，也可以新增类别，不需要修改 Python 代码。Patient 的生成方法由 `--patient` 选择；`single_prompt` 和 `patient_agents` 使用同一份 case 患者资料。

## 首批场景

| id | category | 内容与开场 | 轮数预算 |
| --- | --- | --- | --- |
| `name_recall` | `learning_recall` | 沿用原始复习起点：上周练过姓名策略，现在先问女士姓名，再问男士姓名、记忆方法并适量提示 | 8 |
| `work_task_compensation` | `compensation_daily_life` | “Do you have any questions about keeping track of new tasks and checking your work?”；随后了解具体做法、检查与调整 | 10 |
| `name_practice_review` | `practice_review` | 邀请分享怎样练习组员姓名、记住了多少，再追问实际方法、频率和体验 | 12 |

这些轮数是模拟预算，不是原始对话的行数或轮数。运行时 `--rounds` 优先于 case 内的 `rounds`；case 没填 `rounds` 时使用配置文件的值。当前程序运行到有效轮数结束；`completion_criteria` 只作为语言模型的收束指导，不是自动停止条件。

## 每个目录的内容

```text
cases/
└── 一个case目录/
    ├── case.json
    └── reference_dialogue.json
```

`case.json` 保存人工整理后的场景设计和患者资料。`therapist` 合成治疗师提示，`patient_profile` 供选定的患者方法处理；两侧私有输入分开。字段含义：

| 字段 | 用途 |
| --- | --- |
| `schema_version` | 当前为整数 `1` |
| `id` | 全部 cases 中唯一的小写标识，仅允许小写字母、数字和单个下划线，最长 80 字符；同时用于选择与输出目录 |
| `category` | 同样格式的类别标识；可以复用现有类别或新增 |
| `title`、`summary` | 便于人工识别和说明整理方式；不放入模型提示 |
| `rounds` | 可选的正整数，一轮是治疗师、患者各一条消息 |
| `source` | 可选的原始工作簿、工作表、Excel 行号和 utterance/dialogue 标识；不放入模型提示 |
| `patient_profile` | 必需的患者资料对象；与参考对话分开，包含证据、已知事实、初始记忆和来源 |
| `therapist.context` | 本次活动的起点和情境，包括是否已有材料或先前学习 |
| `therapist.objectives` | 本次交流希望弄清或完成什么 |
| `therapist.materials` | 可用材料、目标答案及判断边界；没有材料时填空列表 |
| `therapist.opening_question` | 第一条治疗师发言，允许包含必要的简短材料介绍，最后聚焦一个主要问题 |
| `therapist.conversation_plan` | 如何根据回答逐步展开；不是逐轮必须照读的脚本 |
| `therapist.reference_utterances` | 可选的原治疗师有效话语列表，合成到治疗师 prompt 中作为提问、追问和反馈示例；不代表已经发生的对话 |
| `therapist.response_guidance` | 该任务中特别需要注意的追问、澄清、提示或反馈方式 |
| `therapist.completion_criteria` | 一个有用的结束应当明确哪些内容，也允许无改善、未解决或患者暂不继续 |

`reference_dialogue.json` 是可选的追溯材料。首批文件保留所选片段的全部原始行，包括短答与 `include=no` 行；每行有 `excel_row`、`utt_id`、说话人、原文、时间和原始标注。完整文件不被加载进模型；其中选出的治疗师有效原句或连续摘句另存为 `therapist.reference_utterances`，供运行时参考。原始患者回答不进入模型，也不会被预填为当前患者回答。

`source.workbook` 以 `LLM_TYB_Data_SMU_Deidentified_20261012/` 为相对根，Excel 行号包含标题行。运行不依赖原始工作簿在场；其作用是便于人工追溯。

## 嵌入的患者 profile

三个身份通过原始说话人编码与 profile 中的 SS 角色核对。人口学来自 `Home Visit Stroke Project.xlsx` 的 `Home Visit 1`；评分来自有明确日期、早于模拟录制日期的 `Home Visit 2`。不把后续随访评分或目标患者回答加入初始状态。

| case | 患者 ID | 模拟日期 | 评估日期（按源记录） | MoCA 总分 | 原始资料行 / 评估行 |
| --- | --- | --- | --- | --- | --- |
| `name_recall` | XC030 | 2022-12-23 | 2022-11-23 | 28 | `Home Visit 1` 第107行 / `Home Visit 2` 第42行 |
| `work_task_compensation` | EC011 | 2022-12-01 | 2022-10-26 | 29 | `Home Visit 1` 第27行 / `Home Visit 2` 第6行 |
| `name_practice_review` | YC026 | 2022-12-05 | 2022-07-11 | 25 | `Home Visit 1` 第154行 / `Home Visit 2` 第21行 |

YC026 的评估日期是原表实际保存的 **2022-07-11**，没有根据课程日期猜改为 11 月 7 日。`Home Visit 1` 未给独立评估日期，背景中的 `consent_date` 只表示同意日期。EC011 的定向力合计 `Home Visit 2!AL6=1` 与 `AF6:AK6` 六项均为1冲突，因此 `assessments.moca.orientation` 为 `null`；原值、分项及问题保存在 provenance，未自动修正。

`patient_profile` 的顶层字段必须完整，缺失数值使用 `null`，不使用0替代：

| 字段 | 内容与约束 |
| --- | --- |
| `schema_version` | 整数 `1` |
| `participant_id`、`role` | 参与者编号；当前 `role` 必须为 `stroke_survivor`，不可混入照护者资料 |
| `assessment_date` | `YYYY-MM-DD` 或确实未知时为 `null`；不能晚于模拟日期 |
| `simulation_date` | 必需的 `YYYY-MM-DD` 日期 |
| `background` | 必需包含 `age`、`language`、`education_years`，未知为 `null`；可增加有来源的标量，例如生活协助情况和卒中年份 |
| `assessments.moca` | 必需包含 `total`、`attention`、`language`、`delayed_recall`、`category_cue`、`multiple_choice_cue`、`visuospatial_executive`、`abstraction`、`orientation`、`naming`；可增加有来源的具体分项 |
| `assessments.sdmt` | `total`，保留原尺度；异常或未确定值为 `null` |
| `assessments.dass21` | `depression`、`anxiety`、`stress`，本批按所选访视录入总分保存，不乘2、不赋临床严重程度标签 |
| `known_facts` | 已核实的个人事实列表；每条包含唯一 `id`、`text`、`source`，可选 `cue_tags` 字符串列表 |
| `initial_memories` | 模拟开始前确实建立的经历记忆，条目格式同上；本批三个 case 均为 `[]` |
| `provenance` | 来源路径、SHA256、工作表/单元格、原始值、处理说明和数据问题；用于审计，不交给输出模型 |
| `unknowns` | 尚不能由资料建立的事实或能力，使用字符串列表明确记录 |

例如，生活独立性和同住情况可以成为已知事实；工作中使用电子表格、完成姓名练习或记住目标姓名不能从场景主题推出。语言编码也不等于语言熟练度。`patient_agents` 只把部分认知分项转成显式的实验假设，具体策略见 `patient_agents/policy.json`，这些映射尚未临床校准。

各字段来源已经嵌入 JSON，运行只校验并读取 case，不打开 Excel、数据字典或 `reference_dialogue.json`。profile 的来源哈希是制作时记录的版本依据；运行另计算 `patient_profile_sha256`，便于确认不同方法确实使用同一份输入。改变 profile 后应更新其 provenance，保留未知和冲突，不能静默修补原始数据。

## 与三个原始实例的对应

| case | 原文件（`02_whole_session_data/` 下） | `utterances` 的 Excel 行 | `utt_id` |
| --- | --- | --- | --- |
| `name_recall` | `Group 2/TYB2_S8_20221223_master_annotated_reviewer2.xlsx` | 42–58 | 41–57 |
| `work_task_compensation` | `Group 1/TYB1_S6_20221201_master_annotated.xlsx` | 219–258 | 218–257 |
| `name_practice_review` | `Group 1/TYB1_S7_20221205_master_annotated.xlsx` | 76–169 | 75–168 |

首例保留原实例的“上周学过，现在复习”任务起点。开场清理了原单元格中的饮食话题转场和静音提醒，使用：

> Last week we practised some strategies for remembering these people's names. Can you remember this lady's name?

Tan Mei Li、Lim Chong Wei 及联想保存在治疗师参考材料中，开场不先给出答案。完整男士姓名由同文件第 877–891 行核对，原文保存在 `material_utterances`。当前文本 runner 尚未重建患者的前次教学或提供原人物图片，三个 case 的 `initial_memories` 均为空。所以任务情境不等于患者确实已学过，姓名场景目前不能称为受控的延迟记忆 benchmark。若患者不认识任务或缺少材料，治疗师应先澄清。需要评价保留或提示恢复时，应先明确呈现学习材料，再控制间隔、干扰和提示；不能把参考患者的回答当作先前记忆注入。

治疗师指导保留任务目标与有效追问方式，允许当前患者有不同做法和反应。不会预设患者使用电子表格、认识原组员、完成了作业、存在某种情绪或已经进步。原始转录中描述的个人经验与临床说法不自动成为任务事实或标准答案。

## 增加一个 case

复制最接近的现有 case 目录作为完整模板，保留必需的 `patient_profile` 结构，改成新的唯一 `id`，再修改 `category`、主题与 `therapist` 字段。目录可以继续分层；加载器递归寻找名字恰好是 `case.json` 的文件，按相对路径排序。

例如新建 `reminder_practice_review` 时，先复制 `name_practice_review`，再把 `id` 改为 `reminder_practice_review`，更新标题、目标、开场和推进方式。如果仍使用同一患者，应明确保留该身份并核对新的模拟日期；如果换患者，必须同时替换完整 profile 的身份、评分、事实和来源，不能只改 `participant_id`。新增过去经历只能来自明确的先前材料，并说明来源与时间；没有这些材料时保持 `initial_memories: []`。没有依据的评分填 `null`，未知的背景写入 `unknowns`。

如填写 `source`，需要包含 `workbook`、`worksheet`、`excel_rows: [起始行, 结束行]`、`utt_ids: [起始ID, 结束ID]`、`dialog_ids: [...]`。没有原始转录时可以省略整个 `source` 和 `reference_dialogue.json`。

在代码仓库根目录执行：

```bash
python run_dialogue.py --list-cases
python run_dialogue.py --case reminder_practice_review --patient patient_agents --dry-run --show-patient
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case reminder_practice_review --patient patient_agents
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case reminder_practice_review --patient single_prompt
```

先完成上述新 case 的创建再运行这些命令。普通 `--dry-run` 输出治疗师提示、启动指令、患者 ID 和 profile 哈希；`--show-patient` 同时显示患者信息与方法配置。两者都不加载模型或生成对话。实际运行另保存 `patient_trace.jsonl`，用于核对患者每轮获得了什么信息及发生了哪些内部决策。
