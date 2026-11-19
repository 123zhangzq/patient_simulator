# 制作 benchmark 实例

在含有 `config.json` 的仓库根目录运行。生成使用 Python 标准库，无需模型、API key 或原始 Excel。

```bash
python construct_ins/build.py
python construct_ins/build.py --check
```

`build.py` 按病例 ID 顺序写入 UTF-8、LF 换行的两份成品文本和 `instances/manifest.json`。`--check` 重新渲染到内存，检查已提交字节和清单是否一致。每次修改病例或模板，应将更新后的成品同时提交。数据集发布时只需导出 `instances/` 和运行 benchmark 的文件。

| 要修改的内容 | 编辑位置 |
| --- | --- |
| 通用角色指令 | `templates/therapist.txt` 与 `templates/patient.txt` |
| 共同情境、活动目标、材料、固定开场、推进建议、结束条件、轮数 | `scenarios/<scenario>.json` |
| 患者资料、用于本次扮演的补充说明、此前已发生的对话 | `cases/<instance-id>.json` |
| 原始参考选段、来源切分说明 | `references/`，供制作者核对 |

新增实例时可复制相近的病例文件，令 `id` 与文件名一致，并用 `scenario` 指向一个场景文件。两位 SMART Goal 患者共用 `scenarios/smart_goal.json`，因此发布后的治疗师文本一致。病例文件保留原始资料中的来源、测量、事实及其依据。构建器只导出列明的背景、测量值、已知事实、初始经历和 `patient_role_notes`。录入者可将已经确认、与本例相关的证据边界写入这些角色说明。

`new_activity` 使用空前史。`continuation` 使用完整的治疗师／患者发言对，最后一句为患者发言。构建器把相同的 `[Earlier conversation]` 段落各写入两份成品，来源与评价说明留在制作侧。姓名回忆的六条前史由研究者编写，只交代本次活动中呈现过的材料；工作笔记和姓名练习回顾分别使用六条和十六条转录前史。SMART Goal 两例的既有自述来自患者资料，前史为空。

`[Opening utterance]` 是治疗师文本的最后一段，保存原样播出的一句开场。`[Earlier conversation]`、`[Opening utterance]` 和 `[END_SESSION]` 为运行协议保留标记。共同情境只放双方知道的信息；治疗师私有任务、材料及结束条件由场景文件中的 `therapist` 部分提供。所有后续台词由各角色在实际交谈时产生。
