# 病例资料制作记录

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

