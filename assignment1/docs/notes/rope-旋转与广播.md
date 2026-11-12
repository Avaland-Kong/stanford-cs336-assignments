# RoPE：旋转、预计算和广播

## 2026-11-11

### 问题是什么

实现讲义 §3.4.3 的 RoPE：位置为什么打在注意力的 Q/K 上、`token_positions` 是什么、配对方式和转角公式怎么读、表要不要预计算、和未知的 batch / head 维怎么广播。`test_rope` 已通过。

### 现象/尝试

一开始把位置编码想成加在词嵌入上。讲义用的是旋转，公式里的 \(\Theta\)、指数正负、乘除、\(2\times 2\) 的 sin 符号都容易和博客或旧译文拧反。中译一度写成正指数相乘、以及另一套旋转矩阵；对照 PDF 后改成除法，矩阵改成标准旋转。实现用相邻两维当复数再乘 \(e^{i\theta}\)，单测通过。Adapter 文档把 `token_positions` 写成 `(batch, seq)`，和 jaxtyping、真实测试不一致。

### 讨论结论

**为什么不在嵌入层加位置**

RoPE 要让 \(q_i\cdot k_j\) 只依赖相对距离 \(j-i\)。旋转必须发生在 \(W_Q、W_K\) 之后、点积之前。每一层有自己的投影，所以每层注意力都转一次。Value 不转。维度是 head dim，不是 `d_model`。

**`token_positions`**

整数位置下标 \(i\)，不是 token ID。形状 `(..., seq_len)`，和 \(x\) 的序列维对齐。测试里是 `0, 1, 2, …`。\(i=0\) 是恒等旋转。表按 `0 .. max_seq_len-1` 预存，forward 用传入的下标去切，不要假定当前窗口从 0 起。Changelog 改过 off-by-one，不要改成从 1 起。

**配对**

讲义是相邻两维一对（下标 \(2k-2\) 与 \(2k-1\)），不是跨半。跨半好写、访存看起来更顺，但槽位不同，snapshot 对不上。相对位置两种都能成立，作业只认公式 (8) 的槽位。不必构造完整 \(d\times d\) 的 \(R_i\)。

**转角**

\(\Theta\)（构造参数 `theta`）测试里是 10000，大于 1。转角 \(\theta_{i,k}\) 是弧度，可以大于 1；第一对就是 \(i\) 本身。PDF 是除：

\[
\theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d}}
\]

\(k\) 变大则每步转得更慢。正指数相乘会越来越快，和几何频率相反。指数 \((2k-2)/d\) 对应偶数 \(0,2,4,\ldots\) 再除以 \(d\)。先除再当幂，不要先 \(\Theta^{0,2,4,\ldots}\) 再除以 \(d\)。`1/\Theta^{x}` 与 \(\Theta^{-x}\) 相同。

标准旋转矩阵（已按 PDF 改正文）对应 \(e^{i\theta}\)：

\[
R_i^k=\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}
\]

旧译文把 sin 符号写反了，会当成 \(e^{-i\theta}\)。以 PDF 和 `test_rope` 为准。

**预计算**

位置轴 `0 .. max_seq_len-1` 与频率轴都是等差数列，外积得到 `(max_seq_len, d/2)` 的角，再存 cos/sin 或等价的复数因子。只在构造时算一次，不是每个 forward。RoPE 无学习参数，各层超参相同则可共用一个模块。`device` 透传给造表，不要存 `self.device`。用 `register_buffer(..., persistent=False)`，不要做成 `Parameter`。`reset_parameters` 一般不收 `device`，RoPE 也不需要这套初始化。

**精度**

接口没有 `dtype`。表用 fp32；和 `x` 相乘的提升只发生在这次运算的结果上，不会改掉原来的激活。若返回 fp32，残差流会跟着升精度，需要按 `x` 的 dtype 收回。bf16 表过不了 `atol=1e-5`。`token_positions` 保持整数。

**并行与求导**

位置维、batch 维、各 pair 之间都没有数据依赖，都可以张量化。对 pair 写 Python 循环只是写法，不是必须串行。不要原地改 `x`；往 `empty` 里切片赋值默认不在计算图上，前向可能对、训练时梯度接不上。出新张量再拼回去。

**广播**

\(x\) 是 `(..., seq_len, d_k)`，位置是 `(..., seq_len)`。序列长度必须一致；前面的维按**从右边对齐**广播，不必同 rank。`test_rope` 是 `(4, 12, d)` 配 `(12,)`。

RoPE 猜不了多出来的维是 batch 还是 head。契约与 PyTorch 其它算子相同：位置要能和 `x[..., 0]` 右对齐。`(batch, seq)` 配 `(batch, heads, seq, d_k)` 会在 head 对 batch 上失败或静默乘错。各 head 共用位置，调用方（MHA）应传入能右对齐的形状，例如 `(seq,)` 或 `(1, seq)`。Adapter 的 Args 散文把位置写成 `(batch, seq)`，以 jaxtyping 的 `... sequence_length` 和实际 fixture 为准。

### 遗留卡点

RoPE 单测已过。下一步是 SDPA / MHA：mask 在 softmax 之前；RoPE 只打在每个 head 的 Q/K 上，维度是 head dim。
