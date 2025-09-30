# SymMark — From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for LLMs

**作者**：Yidan Wang, Yubing Ren*, Yanan Cao, Binxing Fang（*通讯）
**会议**：ACL 2025（Main）
**论文**：arXiv:2505.09924（v2，2025-05-16）
**代码**：GitHub/redwyd/SymMark
**参考**：见文末链接 [1]–[4]

---

## 动机（Motivation）

主流文本水印方法分为两类：logits-based（如 KGW、Unigram）与 sampling-based（如 AAR、EXP）。单一路线在可检测性、文本质量、鲁棒性与安全性之间存在不可避免的取舍。SymMark 提出将两类方法在同一生成过程中“共生”组合，通过串联（Series）、并联（Parallel）与混合（Hybrid）三种策略，将传统的权衡转化为协同。

---

## 框架概述（Series / Parallel / Hybrid）

* **Series（串联）**：每个 token 同时施加 logits 水印与采样水印，检测信号最强，但对文本质量的约束最大。
* **Parallel（并联）**：按步交替（如奇偶位）在两类水印间切换，单步只施加其一，质量更稳但灵活性不足。
* **Hybrid（混合，推荐）**：基于两种熵度量（不确定性与语义多样性）自适应选择该步使用哪类水印（或两者/都不），在检测性、质量、鲁棒性与安全性之间动态折中。

---

## 熵路由（核心机制与公式解释）

记第 (t) 步模型输出的 logits 向量为 (l_t)，其对应的概率分布为
[
p_t=\mathrm{softmax}(l_t),\quad p_{t,i}\text{ 表示第 }i\text{ 个候选的概率。}
]

**Token Entropy（不确定性）**
[
H_{\mathrm{TE}}(t)=-\sum_i p_{t,i}\log p_{t,i}.
]
含义：度量当前步概率分布的不确定性。(H_{\mathrm{TE}}) 越高，说明分布越平坦、候选越多，轻微修改 logits 对流畅度的影响越小，适宜施加 logits 水印；(H_{\mathrm{TE}}) 低则模型已“很确定”，改 logits 更可能伤害质量。

**Semantic Entropy（语义多样性）**
对 Top-(k) 候选做语义聚类，记聚成 (n) 个语义簇 (C={C_1,\ldots,C_n})，定义簇概率：
[
q_{t,j}=\sum_{i\in C_j} p_{t,i},\quad j=1,\ldots,n.
]
据此定义语义熵：
[
H_{\mathrm{SE}}(t)=-\sum_{j=1}^{n} q_{t,j}\log q_{t,j}.
]
含义：度量候选在语义层面的多样性。(H_{\mathrm{SE}}) 低表示候选语义相近，此时改采样更不易改变句子含义，适宜施加 sampling 水印；(H_{\mathrm{SE}}) 高表示语义差异大，应谨慎改变采样。

**Hybrid 路由规则（默认超参）**
阈值 (\alpha,\beta) 控制是否施加两类水印。推荐默认：Top-(k=64)、簇数 (n=10)、(\alpha\approx1.0)、(\beta\approx0.5)。流程：

1. 若 (H_{\mathrm{TE}}(t)>\alpha)，先对 (l_t) 施加 logits 水印算子 (A_w(\cdot)) 得 (\tilde l_t)，并计算 (\hat p_t=\mathrm{softmax}(\tilde l_t))；否则 (\hat p_t=\mathrm{softmax}(l_t))。
2. 若 (H_{\mathrm{SE}}(t)<\beta)，在 (\hat p_t) 上施加采样水印 (S_w(\cdot)) 采样 (y_t)；否则使用原采样策略 (S_o(\cdot)) 得到 (y_t)。

---

## 生成式（Series 与 Parallel）

* **Series**：
  [
  y_t=S_w\big(\mathrm{softmax}(A_w(l_t))\big).
  ]
* **Parallel**（以奇偶位为例）：
  [
  y_t=
  \begin{cases}
  S_o\big(\mathrm{softmax}(A_w(l_t))\big),& t\text{ 为偶数}\
  S_w\big(\mathrm{softmax}(l_t)\big),& t\text{ 为奇数}
  \end{cases}
  ]

---

## logits 水印偏置与检测统计

以 KGW/Unigram 风格为例，令当步密钥与上下文决定的 green 集合为 (G_t)，对 (i\in G_t) 的 logits 加小偏置 (\delta)：
[
l'_i = l_i + \delta\ (i\in G_t),\quad l'_i=l_i\ (i\notin G_t).
]
无须显式惩罚 red，即可在保持质量的同时提高 green 的选中概率。

经典 z-score 检测（长度 (L)，green 占比期望 (\gamma)，实际 green 数 (n_{\text{green}})）：
[
z=\frac{n_{\text{green}}-\gamma L}{\sqrt{L\gamma(1-\gamma)}}.
]
含义：无水印时 (n_{\text{green}}) 近似服从 (\mathrm{Binom}(L,\gamma))，(z) 近零；有水印时 (z) 系统性偏大，可据阈值判定。

---

## 统一检测（Unified Detection）

并行计算 logits 路与 sampling 路的检测统计（例如 logits 路用 z-score，sampling 路用基于随机序列的 p-value 或其变换），判定规则：
[
I=\big[ D_l>z_1 \big]\ \lor\ \big[ D_s>z_2 \big],
]
即任一路显著即判定为带水印（在低误报设置下稳健）。与“按组检测”（先将 token 归入两路再分别检测）相比，统一检测更简单高效。

---

## 实验结论（浓缩）

* **可检测性**：在 C4、OpenGen 与 OPT-6.7B、GPT-J-6B 等设置下，Series 与 Hybrid 的 F1/AUC 整体领先；Parallel 稳定但略低。
* **文本质量**：Hybrid 在 PPL 与下游任务上的性能降幅最小，长输出任务更稳。
* **鲁棒性**：面对编辑、复制拼接、回译与改写等攻击，Hybrid/Series 的 AUROC 明显高于单一路线（如 Unigram/AAR）。
* **安全性**：对水印窃取与伪造攻击，Hybrid 的攻击成功率更低（混合了采样随机性，使频率统计难以反推规则）。
* **效率**：生成阶段 Hybrid 需计算两类熵，开销小幅上升；检测阶段与传统方法接近。

---

## 实施要点（落地清单）

1. 接入点：在“logits 输出后、采样前”插入“熵路由 + 水印器”，不改动主推理框架。
2. logits-based：采用 Unigram/KGW 等，对 green 集合加小偏置 (\delta)（通常不对 red 施负偏置）。
3. sampling-based：实现 AAR/EXP/ITS 之一；AAR 简洁易集成。
4. 路由阈值：以 Top-(k=64)、(n=10)、(\alpha\approx1.0)、(\beta\approx0.5) 启动，使用验证集网格搜索平衡 AUROC 与 PPL/任务指标。
5. 检测端：并行计算两路统计，取并集判定；显式记录生成长度 (L)、检测阈值、误报控制策略。
6. 对抗评测：覆盖删除、同义替换、回译、模型释义等扰动，关注 FPR@TPR、ASR 与质量跌幅；可直接复刻官方仓库默认脚本与参数。

---

## 与 WatME 的对照与可叠加

WatME 通过“词汇冗余簇 + 互斥分配 red/green”降低质量损耗；SymMark 通过“熵路由 + 双范式共生”实现检测性、质量、鲁棒性与安全性的动态协同。两者可叠加：在 SymMark 的 logits 路内部采用 WatME 的语义簇互斥划分，进一步降低质量代价，尤其适用于知识召回与推理任务。

---

## 限制与注意

阈值与聚类超参对路由敏感；确定性强的场景更适合偏向 sampling 路。若攻击者可观测内部 logits 或采样随机源，将削弱安全性，需做好密钥管理与可观测面最小化。跨语言与跨 tokenizer 时需对齐语义嵌入与聚类过程。

---

* 论文 PDF 与摘要页：[arXiv（PDF）][1]、[arXiv（摘要）][2]
* ACL Anthology 正式版：[ACL Anthology][3]
* 代码仓库：[GitHub/redwyd/SymMark][4]

