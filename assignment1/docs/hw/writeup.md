# Assignment 1 书面题目

来源：`cs336_assignment1_basics.pdf`（Version **26.0.3**）。  
提交 Gradescope 时，这些题写进 `writeup.pdf`（需 typeset）。代码题交 `code.zip`，本文不收录实现细节。

数字、超参、资源上限、是否必做 leaderboard 以 PDF 原文为准。本文只抽题目和交付物，不是解法。

相关笔记：

- [`流程与注意事项.md`](./流程与注意事项.md)
- [`pdf-s2-分词器中译.md`](./pdf-s2-分词器中译.md)

---

## 怎么区分

| 类型 | 进哪 | 本文 |
|---|---|---|
| 只要文字 / 表 / 曲线 / 生成样例 | `writeup.pdf` | 全文列出 |
| 先跑实验，再写一两句观察 | `writeup.pdf`（外加你的代码与产物） | 只列书面交付 |
| 实现模块并通过 pytest | `code.zip` | 文末只给名单，不展开 |

---

## 一览

| 题号 | 节 | 分 | 书面交付 |
|---|---|---|---|
| `unicode1` | 2.1 | 1 | 三句 |
| `unicode2` | 2.2 | 3 | 编码选择、错误解码例子、非法两字节 |
| `train_bpe_tinystories` | 2.5 | 2 | 时间内存、最长 token、profile |
| `train_bpe_expts_owt` | 2.5 | 2 | 最长 token、两个 tokenizer 对比 |
| `tokenizer_experiments` | 2.7 | 4 | 压缩率、错配 tokenizer、吞吐、为何 `uint16` |
| `transformer_accounting` | 3 | 5 | 参数/显存、FLOPs 清单、占比、缩放、加长 context |
| `learning_rate_tuning` | 4.2 | 1 | 三个学习率下 loss 怎么变 |
| `adamw_accounting` | 4.3 | 2 | 显存分解、最大 batch、一步 FLOPs、训练时长 |
| `experiment_log` | 7.1 | 3 | 实验记录文档 + 日志基础设施 |
| `learning_rate` | 7.2 | 3 | 多条 LR 曲线、至少一次发散、val loss 达标 |
| `batch_size_experiment` | 7.2 | 1 | 不同 batch 曲线 + 讨论 |
| `generate` | 7.2 | 1 | ≥256 token 样例 + 流畅度评论 |
| `layer_norm_ablation` | 7.3 | 1 | 去 RMSNorm 曲线 + 评论 |
| `pre_norm_ablation` | 7.3 | 1 | post-norm vs pre-norm 曲线 |
| `no_pos_emb` | 7.3 | 1 | RoPE vs NoPE 曲线 |
| `swiglu_ablation` | 7.3 | 1 | SwiGLU vs SiLU 曲线 + 讨论 |
| `main_experiment` | 7.4 | 2 | OWT 曲线、loss 解读、生成样例 |
| `leaderboard` | 7.5 | 6 | 最终 val loss、墙钟曲线、做了什么 |

曲线题一律同时记 **step** 和 **wall-clock**。

---

## §2 Tokenizer

### unicode1 — Understanding Unicode（1 分）

(a) `chr(0)` 返回哪个 Unicode 字符？  
> 返回空字符，'\x00'


(b) 这个字符的字符串表示（`__repr__()`）和打印出来的样子有什么不同？  
> 字符串表示会显示可见的转义字符，'\x00'，打印出来是一个空字符，不可见。


(c) 这个字符出现在文本里会怎样？可在解释器里试讲义给的那几行。  
> 这个字符是零宽字符，在文本中打印出来不可见，不会像正常字符一样显示，但仍占据一个字节

### unicode2 — Unicode Encodings（3 分）

(a) 为什么更愿意在 UTF-8 字节上训 tokenizer，而不是 UTF-16 或 UTF-32？可以对不同输入比较这几种编码的输出。  
> UTF-8，更节省空间，同时ASCII字符在UTF-8保持原来的单字节形式，UTF-8和ASCII兼容。除此之外，由于UTF-8是可变长字符，可以用单字节表示所有有效的Unicode字符，避免了出现未知字符的情况，同时也方便分词器训练，因为分词器只需要处理所有单字节值即可。


(b) 讲义里有一个**错误的**解码函数（按单个字节分别 `decode`）。它为什么不对？给出一个会得到错误结果的输入字节串。讲义对 `"hello"` 的 UTF-8 调用表面上还能得到 `'hello'`。  
> 答:b'\xe4\xbd\xa0'，这是汉字‘你’的编码，在这个错误的解码函数中，它会解码错误，因为utf-8是变长编码，一个编码不一定只占一个字节。而对"hello"的调用还能得到'hello',因为这个字符串，每个字符恰好是一字节编码


(c) 给出一个**不能**解码成任何 Unicode 字符的两字节序列。  
> 例子为b'\xC1\x81'，解出码点为U+0041，是字符'A'，在utf-8中由单字节编码，由于utf-8不允许过长编码，所以这个两字节序列不能解码成任何码点

### train_bpe_tinystories — BPE Training on TinyStories（2 分）

先在 TinyStories 上训字节级 BPE，最大词表 **10,000**，加入 `<|endoftext|>`，把 vocab / merges 存盘。然后再写：

(a) 训练花了多少时间和内存？词表里最长的 token 是什么？说不说得通？  
资源：≤ 30 分钟（不需要 GPU），≤ 30 GB RAM。提示：pretokenize 多进程，并利用文档分隔符在 merge 前当特例处理，应能到 2 分钟以内。  
> 训练用时1min09s，峰值内存占用539.20MB，词表最长token是 accomplishment，合理，这应该是TinyStories内出现频率比较高的最长单词

(b) Profile 你的代码。训练过程哪一部分最耗时？  
交付：根据Profile的结果，训练最耗时的部分是预分词，大约占用猜测是TinyStories重复极高，独特 pre-token 不会随语料大幅增长，因此merge过程耗时较小

### train_bpe_expts_owt — BPE Training on OpenWebText（2 分）

先在 OpenWebText 上训，最大词表 **32,000**，vocab / merges 存盘。然后再写：

(a) 最长 token 是什么？说不说得通？  
资源：≤ 12 小时（不需要 GPU），≤ 100 GB RAM。  
> 答：训练用时57min58s，峰值内存占用7016.51 MB，词表最长token是ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ，合理，这应该是网页文本中常出现的乱码串

(b) 对比 TinyStories 上训出的 tokenizer 和 OpenWebText 上训出的。  
交付：一两句话。

### tokenizer_experiments — Experiments with tokenizers（4 分）

(a) 从 TinyStories 和 OpenWebText 各抽 10 篇文档。用你训好的两个 tokenizer（10K 和 32K）编码。各自的压缩率（bytes/token）是多少？  
> 从两个数据集的验证集中取前十篇，不剔除特殊标记。
> TinyStories tokenizer 压缩率为4.044
> OWT tokenizer 压缩率为4.513


(b) 用 TinyStories tokenizer 去切 OpenWebText 样本会怎样？比较压缩率，或定性描述。  
> 用 TinyStories tokenizer 去切 OpenWebText 样本，压缩率变成3.411098375025285，压缩率显著降低

(c) 估计 tokenizer 的吞吐（例如 bytes/second）。切完整份 The Pile（825GB 文本）要多久？  
> TinyStories tokenizer 的吞吐量为  1.204 MiB/s，切完整份 The Pile 要 194.906 h
> OWT tokenizer 的 吞吐量为 0.992 MiB/s，切完整份 The Pile 要 236.559 h

(d) 把各自的 train / development 编成 token ID。建议存成 NumPy `uint16`。为什么这个 dtype 合适？  
> uint16的数据范围是0~65535，分词器的词表大小分别为10000，32000，unit16恰好可以存下，可以在节省内存的同时，让存储更加整齐

---

## §3 模型核算

### transformer_accounting — Transformer LM resource accounting（5 分）

用本作业的结构，按 GPT-2 XL 规格：

- `vocab_size` 50,257
- `context_length` 1,024
- `num_layers` 48
- `d_model` 1,600
- `num_heads` 25
- `d_ff` 4,288（`8/3 × 1600` 最近的 64 倍数）

矩阵乘 FLOPs 规则见 PDF：\(A \in \mathbb{R}^{m \times n}\)、\(B \in \mathbb{R}^{n \times p}\) 时，\(AB\) 需要 \(2mnp\) FLOPs。

(a) 可训练参数多少？单精度浮点时，只加载模型要多少内存？  
>
> 计数依据：`TransformerLM` 中 `Embedding` 与 `lm_head` 是两个独立矩阵（未做 weight tying），
> 所有 `Linear` 无 bias；`RoPE` 的 `rotary_factor_buf` 是 persistent=False 的 buffer，不是参数。
>
> **每层参数**
>
> \[
> \underbrace{3d^2}_{W_{qkv}}+\underbrace{d^2}_{W_o}+\underbrace{3d\,d_{ff}}_{\text{SwiGLU }w_1,w_3,w_2}+\underbrace{2d}_{\text{ln1},\text{ln2}}
> =4d^2+3d\,d_{ff}+2d
> \]
>
> 代入 \(d = 1600\)、\(d_{ff} = 4288\)：
>
> | 部件 | 形状 | 每层 | ×48 层 |
> |---|---|---|---|
> | `attn.w_qkv` | \(1600\times4800\) | \(3d^2 = 7{,}680{,}000\) | \(368{,}640{,}000\) |
> | `attn.w_o` | \(1600\times1600\) | \(d^2 = 2{,}560{,}000\) | \(122{,}880{,}000\) |
> | `ffn.w1`（门控） | \(1600\times4288\) | \(6{,}860{,}800\) | \(329{,}318{,}400\) |
> | `ffn.w3`（内容） | \(1600\times4288\) | \(6{,}860{,}800\) | \(329{,}318{,}400\) |
> | `ffn.w2`（回投） | \(4288\times1600\) | \(6{,}860{,}800\) | \(329{,}318{,}400\) |
> | `ln1`, `ln2` | \(1600\) | \(3{,}200\) | \(153{,}600\) |
> | **小计** | | \(30{,}825{,}600\) | \(1{,}479{,}628{,}800\) |
>
> 外加块外参数：`token_embeddings` \(50257\times1600 = 80{,}411{,}200\)；`lm_head` \(1600\times50257 = 80{,}411{,}200\)；`ln_final` \(1600 = 1{,}600\)。
>
> **总可训练参数**
>
> \[
> N=2Vd+L\,(4d^2+3d\,d_{ff}+2d)+d
> =160{,}822{,}400+1{,}479{,}628{,}800+1{,}600
> =\mathbf{1{,}640{,}452{,}800}\approx1.64\text{B}
> \]
>
> 占比：FFN \(60.2\%\)（w1、w3、w2 各 \(20.1\%\)）、attention \(30.0\%\)、两个词表矩阵各 \(4.9\%\)、norm \(0.01\%\)。
>
> **单精度（fp32）仅加载模型权重**
>
> \[
> 1{,}640{,}452{,}800\times4\ \text{B}=6{,}561{,}811{,}200\ \text{B}
> \text{= }6.11\ \text{GiB})
> \]
>





(b) 一次 forward（序列长度为 `context_length`）需要哪些矩阵乘？总共多少 FLOPs？  
>
> 约定：batch=1，\(T=\text{context\_length}=1024\)，\(d=1600\)，\(d_{ff}=4288\)，\(H=25\)，\(d_h=d/H=64\)；
> 只计矩阵乘（\(2mnp\) 规则），causal mask 不省 \(QK^{\top}\) 的 FLOPs；
> RMSNorm / RoPE / softmax / SiLU 均为逐元素运算，不计入。
>
> **每个 Transformer Block 内（7 个矩阵乘）**&#8203;
>
> | # | 运算 | 矩阵形状 | FLOPs | 说明 |
> |---|---|---|---|---|
> | 1 | QKV 投影 \(XW_{qkv}^{\top}\) | \((1024,1600)\otimes(1600,4800)\) | \(6Td^2=\) 15,728,640,000 | 融合成一个 \(d\times3d\) 矩阵，一次算完 Q/K/V |
> | 2 | 注意力分数 \(Q_hK_h^{\top}\) | \(25\times(1024,64)\otimes(64,1024)\) | \(2T^2d=\) 3,355,443,200 | 每头一个 \(T\times T\) 分数矩阵，带 causal mask |
> | 3 | 加权求和 \(A_hV_h\) | \(25\times(1024,1024)\otimes(1024,64)\) | \(2T^2d=\) 3,355,443,200 | softmax 后的概率矩阵乘 \(V\) |
> | 4 | 输出投影 \(YW_o^{\top}\) | \((1024,1600)\otimes(1600,1600)\) | \(2Td^2=\) 5,242,880,000 | 多头拼回 \(d\) 维 |
> | 5 | SwiGLU 门控 \(XW_1^{\top}\) | \((1024,1600)\otimes(1600,4288)\) | \(2Tdd_{ff}=\) 14,050,918,400 | 升维到 \(d_{ff}\)，再进 SiLU |
> | 6 | SwiGLU 内容 \(XW_3^{\top}\) | \((1024,1600)\otimes(1600,4288)\) | \(2Tdd_{ff}=\) 14,050,918,400 | 与 SiLU 输出逐元素相乘 |
> | 7 | SwiGLU 回投 \((\cdot)W_2^{\top}\) | \((1024,4288)\otimes(4288,1600)\) | \(2Tdd_{ff}=\) 14,050,918,400 | 降回 \(d\) 维 |
>
> \[
> \text{单层}=8Td^2+4T^2d+6Tdd_{ff}=69{,}835{,}161{,}600
> \]
>
> **48 层**：\(69{,}835{,}161{,}600\times48 = 3{,}352{,}087{,}756{,}800\)
>
> **块外（1 个矩阵乘）**&#8203;
>
> | # | 运算 | 矩阵形状 | FLOPs | 说明 |
> |---|---|---|---|---|
> | 8 | 输出层 `lm_head` | \((1024,1600)\otimes(1600,50257)\) | \(2TdV=\) 164,682,137,600 | 投影到词表得到 logits |
>
> `token_embeddings` 是查表，不含矩阵乘。
>
> **总计**
>
> \[
> \text{FLOPs} = L(8Td^2+4T^2d+6Tdd_{ff}) + 2TdV
> = \mathbf{3{,}516{,}769{,}894{,}400}\approx \mathbf{3.52\ TFLOPs}
> \]
>
> 平均每 token \(3.43\) GFLOPs。
>

(c) 哪几部分最吃 FLOPs？  
>
> **FFN（SwiGLU）是绝对大头，占 57.5%**——\(w_1\)、\(w_3\)、\(w_2\) 各 19.2%，一个 block 里
> 一半以上的算力花在“升维 → 门控 → 降维”上；其次是 attention 的四个线性投影，合计 37.8%
> （其中 \(W_{qkv}\) 21.5%、\(W_o\) 7.2%）。真正的 \(T\times T\) 注意力打分只占 9.2%
> （\(QK^{\top}\) 与 \(AV\) 各 4.6%），在 1024 上下文下尚不构成瓶颈。

(d) 对 GPT-2 small（12 layer / 768 `d_model` / 12 head）、medium（24 / 1024 / 16）、large（36 / 1280 / 20）重复分析。模型变大时，各部分占总 FLOPs 的比例怎么变？  
>
> \(d_{ff}\) 按题目给 XL 的规则统一取 \(\tfrac83d\) 最近的 64 倍数：
> small 2048、medium 2752、large 3392、XL 4288。
> 因 \(3d\cdot\tfrac83d = 8d^2 = 2d\cdot 4d\)，此取法与 GPT-2 原 4d FFN 的**参数量完全相同**。
>
> ### 总量
>
> | 模型 | \(L\) | \(d\) | \(H\) | \(d_{ff}\) | 参数量 | 总 FLOPs |
> |---|---|---|---|---|---|---|
> | small | 12 | 768 | 12 | 2048 | 162.1 M | 0.2916 T |
> | medium | 24 | 1024 | 16 | 2752 | 406.5 M | 0.8302 T |
> | large | 36 | 1280 | 20 | 3392 | 833.6 M | 1.7685 T |
> | XL | 48 | 1600 | 25 | 4288 | 1640.5 M | 3.5168 T |
>
> ### 部件占比分解（占单次 forward 总 FLOPs 的百分比）
>
> | 部件 | small | medium | large | XL |
> |---|---|---|---|---|
> | \(W_{qkv}\) | 14.91 | 18.62 | 20.49 | 21.47 |
> | \(QK^{\top}\) | 6.63 | 6.21 | 5.46 | 4.58 |
> | \(AV\) | 6.63 | 6.21 | 5.46 | 4.58 |
> | \(W_o\) | 4.97 | 6.21 | 6.83 | 7.16 |
> | **attention 合计** | **33.13** | **37.25** | **38.25** | **37.78** |
> | **FFN** | **39.76** | **50.05** | **54.30** | **57.53** |
> | `lm_head` | 27.10 | 12.70 | 7.45 | 4.68 |
>
> ### 比例如何变化
>
> 模型变大时，**FFN 占比单调上升（39.8% → 57.5%），\(W_{qkv}+W_o\) 也上升（19.9% → 28.6%），
> 而 \(T^2\) 注意力项从 13.3% 降到 9.2%**——因为 \(T\) 固定、\(d\) 变大，二次项 \(4T^2d\)
> 相对线性项 \(Td^2\) 被 \(d/T\) 稀释（二次项占单层算力的比例为 \(1/(6d/T+1)\)，
> 四者依次 18.2% → 14.3% → 11.8% → 9.6%）。**`lm_head` 从 27.1% 跌到 4.7%**，
> 因为 \(Vd\) 是固定的词表开销，被越来越多的层摊薄。由于 \(d_{ff}=\tfrac83d\) 时
> FFN 与投影的死比例恒为 2:1，占比迁移几乎全部来自“`lm_head` 变小”与“二次项变小”两件事。
>

(e) GPT-2 XL 把 context 加到 16,384。一次 forward 的总 FLOPs 怎么变？各部件相对贡献怎么变？  

>
> \(T\) 变为 \(16\times\)，各项按阶数缩放：QKV、\(W_o\)、FFN、`lm_head` 属**线性项**（\(\propto T\)）
> 涨 \(16\times\)；\(QK^{\top}\) 与 \(AV\) 属**二次项**（\(\propto T^2\)）涨 \(256\times\)。
>
> \[
> \text{FLOPs} = 3.517\ \text{T}\ \longrightarrow\ \mathbf{133.58\ TFLOPs}\quad(\approx 38\times)
> \]
>
> 即 \(16\times 3.030\times10^{12}\)（线性部分）\(+\ 256\times 0.322\times10^{12}\)（二次部分）
> \(+\ 16\times 0.165\times10^{12}\)（`lm_head`）；每 token 从 3.43 GFLOPs 升到 8.15 GFLOPs，
> 参数量与显存不变，仍为 1.64B / 6.56 GB。
>
> 占比翻转：**两个 \(T^2\) 项从 9.2% 暴涨到 61.7%（各 30.9%），attention 合计 37.8% → 73.8%
> 成为主导，FFN 从 57.5% 压到 24.2%，\(W_{qkv}\) 21.5% → 9.0%，\(W_o\) 7.2% → 3.0%，
> `lm_head` 只剩 2.0%**。
>
> 一句话：总 FLOPs 由约 \(16\times\) 的线性增长变成约 \(38\times\)，瓶颈从“权重矩阵乘”切换为
> “序列长度的平方”，长上下文的主要代价落在注意力打分与加权求和上，而 FFN 反而退居次要。

---

## §4 训练组件

### learning_rate_tuning — Tuning the learning rate（1 分）

用讲义里的玩具 SGD 例子，另跑三个学习率：`1e1`、`1e2`、`1e3`，只跑 10 步。每个学习率下 loss 是衰减更快、更慢，还是发散（训练过程中变大）？  
交付：一两句话，写你观察到的行为。SGD 示例代码以 PDF 为准。

### adamw_accounting — Resource accounting for training with AdamW（2 分）

假定所有张量都是 float32。`d_ff = 8/3 × d_model`。

(a) 跑 AdamW 的峰值显存？按参数、activation、梯度、optimizer state 分解，用 `batch_size` 和模型超参（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示。activation 只算讲义列出的那些（block 里的 RMSNorm、MHA 各步、SwiGLU 各步、最后 RMSNorm、output embedding、logits 上的 cross-entropy）。  
交付：每一项的代数式，以及合计。

(b) 代入 GPT-2 XL 规格，得到只含 `batch_size` 的式子。80GB 里最大 batch 是多少？  
交付：形如 \(a \cdot \texttt{batch_size} + b\) 的式子，以及最大 batch。

(c) AdamW **一步**多少 FLOPs？  
交付：代数式 + 简短理由。

(d) MFU = 实测吞吐（token/s）相对硬件峰值 FLOP 吞吐的比。讲义此处按 **H100** 叙事：float32（实际是 TF32）峰值 495 teraFLOP/s。假设 50% MFU，在单张 H100 上把 GPT-2 XL 训 400K step、batch 1024，要多久？backward 按 forward 的两倍 FLOPs 计。  
交付：小时数 + 简短理由。

> Changelog 把不少 H100 表述改成了 B200，但本 PDF 这一问仍写 H100。数字以当前 PDF 为准。

---

## §7 实验

实验曲线都要能同时看到 **step** 和 **wall-clock**。可用 wandb 一类工具。

### experiment_log — Experiment logging（3 分）

为训练和评估搭实验追踪，能按梯度步数和墙钟画 loss 曲线。

交付：

- 实验日志基础设施代码
- 一份 experiment log（本节后面那些实验你都试了什么）

### learning_rate — Tune the learning rate（3 分，2 B200 hrs）

以讲义给的 TinyStories 起点超参为底（vocab 10k、context 256、`d_model` 512、`d_ff` 1344、4 layer / 16 head、RoPE θ = 10000、总 token 约 327,680,000）。自己调 LR、warmup、AdamW 的 β / ε、weight decay。实现正确且高效时，大约 20–30 分钟 / 1 张 B200。

(a) 扫学习率，报告最终 loss（或发散）。  
交付：多条学习率对应的曲线，并说明你的搜索策略。  
交付：TinyStories 上 per-token validation loss **≤ 1.45** 的模型。

低资源（CPU / MPS）：总 token 可降到约 40,000,000，目标 loss 可放到 2.00。MPS 不要开 TF32。其余以 PDF 蓝框为准。

(b) 有说法认为最好的学习率在「稳定边界」。发散点和你的最佳 LR 是什么关系？  
交付：递增学习率的曲线，**至少一次发散**，并分析这和收敛速度的关系。

### batch_size_experiment — Batch size variations（1 分，1 B200 hr）

batch 从 1 一直变到 GPU 显存上限，中间至少几个点（包括常见的 64、128）。必要时重新调 LR。

交付：不同 batch 的学习曲线。  
交付：几句话讨论 batch size 对训练的影响。

### generate — Generate text（1 分）

用你的 decoder 和训好的 checkpoint 生成。可调 temperature、top-p 等。

交付：至少 **256 token** 的文本（或直到第一个 `<|endoftext|>`），外加简短评论：流畅度如何，以及至少两个影响好坏的因素。

### layer_norm_ablation — Remove RMSNorm and train（1 分，0.5 B200 hrs）

去掉所有 RMSNorm 再训。原先最优 LR 下会怎样？降低 LR 能否重新稳定？

交付：去掉 RMSNorm 的曲线，以及该设置下最好 LR 的曲线。  
交付：几句话评论 RMSNorm 的作用。

### pre_norm_ablation — Implement post-norm and train（1 分，0.5 B200 hrs）

把 pre-norm 改成 post-norm 再训。公式见 PDF (25)–(28)。

交付：post-norm 与 pre-norm 对比的学习曲线。

### no_pos_emb — Implement NoPE（1 分，0.5 B200 hrs）

去掉位置信息（NoPE），与带 RoPE 的基线对比。因果 mask 的 decoder-only 理论上仍可能推断位置，这一问是实证。

交付：RoPE vs NoPE 的学习曲线。

### swiglu_ablation — SwiGLU vs. SiLU（1 分，0.5 B200 hrs）

对比 SwiGLU 和「SiLU、无 GLU」的 FFN。SiLU 那条为了对齐参数量，应设 `d_ff = 4 × d_model`（默认 SwiGLU 大约 `d_ff = 8/3 d_model` 且是 64 的倍数，且有三个矩阵）。

交付：参数量大致对齐的两条曲线。  
交付：几句话讨论发现。

在线学生 GPU 不够时，讲义建议 ablation 继续在 TinyStories 上做（用 val loss 比）。是否仍要上 OWT / leaderboard 以 PDF 为准。

### main_experiment — Experiment on OWT（2 分，2 B200 hrs）

用与 TinyStories **相同的模型结构和总训练 iteration** 在 OpenWebText 上训。学习率、batch 可能要重调。

交付：OWT 学习曲线；和 TinyStories 的 loss 差怎么解释。  
交付：生成文本（格式与 TinyStories 那题相同）。流畅度如何？模型和算力预算相同，为什么质量更差？

### leaderboard — Leaderboard（6 分，10 B200 hrs）

在 leaderboard 规则下尽量压低 validation loss：最多 **45 分钟 / 一张 B200**（0.75 B200-hour），数据只用课程提供的 OpenWebText train。其余不限。期望至少打过 naive baseline **5.0 loss**。

交付：记录到的最终 validation loss；墙钟横轴明确少于 45 分钟的学习曲线；你做了什么的说明。  
另外按 leaderboard 仓库 README 提 PR：  
[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)

---

## 文末：不进本清单的代码题

这些主要交实现和测试，书面题不展开。若 writeup 要标脚本路径，以 PDF 为准。

| 题号 | 分 | 测什么 |
|---|---|---|
| `train_bpe` | 15 | BPE 训练函数 |
| `tokenizer` | 15 | `Tokenizer` 类 |
| `linear` / `embedding` / `rmsnorm` | 1+1+1 | 底层模块 |
| `positionwise_feedforward` | 2 | SwiGLU |
| `rope` | 2 | RoPE |
| `softmax` | 1 | softmax |
| `scaled_dot_product_attention` | 5 | SDPA |
| `multihead_self_attention` | 5 | causal MHA |
| `transformer_block` / `transformer_lm` | 3+3 | block 与整模 |
| `cross_entropy` | 1 | 交叉熵 |
| `adamw` | 2 | AdamW |
| `learning_rate_schedule` | 1 | cosine + warmup |
| `gradient_clipping` | 1 | 梯度裁剪 |
| `data_loading` | 2 | `get_batch` |
| `checkpointing` | 1 | save/load |
| `training_together` | 4 | 训练脚本 |
| `decoding` | 3 | 采样（temperature / top-p） |
