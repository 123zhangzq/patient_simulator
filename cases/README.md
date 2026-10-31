# 对话场景 cases

每个目录代表一个可独立运行的对话内容。`category` 是内容类别，同一类别可以有任意多个 case，也可以新增类别，不需要修改 Python 代码。Patient 的人设和生成方法独立于 cases。

## 首批场景

| id | category | 内容与开场 | 轮数预算 |
| --- | --- | --- | --- |
| `name_recall` | `learning_recall` | 教给两个人名和文字联想后，问女士的完整姓名；继续尝试另一姓名、适量提示和策略讨论 | 8 |
| `work_task_compensation` | `compensation_daily_life` | “When someone shows you a new task, how do you keep track of the steps?”；了解已有方法、具体例子、检查与调整 | 10 |
| `name_practice_review` | `practice_review` | “What, if anything, have you tried recently to help you remember people's names?”；回顾实际练习、方法、困难和体验 | 12 |

这些轮数是模拟预算，不是原始对话的行数或轮数。运行时 `--rounds` 优先于 case 内的 `rounds`；case 没填 `rounds` 时使用配置文件的值。当前程序运行到有效轮数结束；`completion_criteria` 只作为语言模型的收束指导，不是自动停止条件。

## 每个目录的内容

```text
cases/
└── 一个case目录/
    ├── case.json
    └── reference_dialogue.json
```

`case.json` 保存人工整理后的场景设计，只有其中的 `therapist` 用来合成模型提示。字段含义：

| 字段 | 用途 |
| --- | --- |
| `schema_version` | 当前为整数 `1` |
| `id` | 全部 cases 中唯一的小写标识，仅允许小写字母、数字和单个下划线，最长 80 字符；同时用于选择与输出目录 |
| `category` | 同样格式的类别标识；可以复用现有类别或新增 |
| `title`、`summary` | 便于人工识别和说明整理方式；不放入模型提示 |
| `rounds` | 可选的正整数，一轮是治疗师、患者各一条消息 |
| `source` | 可选的原始工作簿、工作表、Excel 行号和 utterance/dialogue 标识；不放入模型提示 |
| `therapist.context` | 本次活动的起点和情境，包括是否已有材料或先前学习 |
| `therapist.objectives` | 本次交流希望弄清或完成什么 |
| `therapist.materials` | 可用材料、目标答案及判断边界；没有材料时填空列表 |
| `therapist.opening_question` | 第一条治疗师发言，允许包含必要的简短材料介绍，最后聚焦一个主要问题 |
| `therapist.conversation_plan` | 如何根据回答逐步展开；不是逐轮必须照读的脚本 |
| `therapist.response_guidance` | 该任务中特别需要注意的追问、澄清、提示或反馈方式 |
| `therapist.completion_criteria` | 一个有用的结束应当明确哪些内容，也允许无改善、未解决或患者暂不继续 |

`reference_dialogue.json` 是可选的追溯材料。首批文件保留所选片段的全部原始行，包括短答与 `include=no` 行；每行有 `excel_row`、`utt_id`、说话人、原文、时间和原始标注。它不被 case 加载器读取，不进入 Therapist 或 Patient 的输入，也不会被当作另一个 case。

`source.workbook` 以 `LLM_TYB_Data_SMU_Deidentified_20261012/` 为相对根，Excel 行号包含标题行。运行不依赖原始工作簿在场；其作用是便于人工追溯。

## 与三个原始实例的对应

| case | 原文件（`02_whole_session_data/` 下） | `utterances` 的 Excel 行 | `utt_id` |
| --- | --- | --- | --- |
| `name_recall` | `Group 2/TYB2_S8_20221223_master_annotated_reviewer2.xlsx` | 42–58 | 41–57 |
| `work_task_compensation` | `Group 1/TYB1_S6_20221201_master_annotated.xlsx` | 219–258 | 218–257 |
| `name_practice_review` | `Group 1/TYB1_S7_20221205_master_annotated.xlsx` | 76–169 | 75–168 |

首例原本依赖人物图片和之前的教学。可运行版本改为明确的文字学习活动：开场提供 Tan Mei Li、Lim Chong Wei 及联想，再开始提问。完整男士姓名另由同文件第 877–891 行核对，原文保存在 `material_utterances`。这一版本用于模拟语言互动，不声称测量视觉记忆或无提示的独立回忆；患者也能在聊天历史中看到之前出现的材料。

治疗师指导保留任务目标与有效追问方式，允许当前患者有不同做法和反应。不会预设患者使用电子表格、认识原组员、完成了作业、存在某种情绪或已经进步。原始转录中描述的个人经验与临床说法不自动成为任务事实或标准答案。

## 增加一个 case

复制最接近的 case 目录，改成新的唯一 `id`，再修改 `category`、主题与 `therapist` 字段。目录可以继续分层；加载器递归寻找名字恰好是 `case.json` 的文件，按相对路径排序。

也可以从这个不依赖原始数据的完整示例开始：

```json
{
  "schema_version": 1,
  "id": "reminder_practice_review",
  "category": "practice_review",
  "title": "Reviewing a reminder strategy",
  "summary": "An authored discussion of actual reminder use and practical barriers.",
  "rounds": 10,
  "therapist": {
    "context": "Review any reminder method the patient has actually tried recently.",
    "objectives": ["Clarify what was tried and one practical experience or barrier."],
    "materials": [],
    "opening_question": "What, if anything, have you tried recently to remind yourself about daily tasks?",
    "conversation_plan": [
      "Ask for one example if a method was attempted.",
      "Explore what happened or one barrier, then discuss an optional small adjustment."
    ],
    "response_guidance": ["Do not assume the patient used a phone, forgot an appointment, or benefited."],
    "completion_criteria": ["Summarize the reported experience and any next step the patient accepts."]
  }
}
```

如填写 `source`，需要包含 `workbook`、`worksheet`、`excel_rows: [起始行, 结束行]`、`utt_ids: [起始ID, 结束ID]`、`dialog_ids: [...]`。没有原始转录时可以省略整个 `source` 和 `reference_dialogue.json`。

在代码仓库根目录执行：

```bash
python run_dialogue.py --list-cases
python run_dialogue.py --case reminder_practice_review --dry-run
CUDA_VISIBLE_DEVICES=0 python run_dialogue.py --case reminder_practice_review
```

预览只输出整理后的 Therapist system prompt 和启动指令，不加载模型，也不输出 Patient 的私有 prompt。
