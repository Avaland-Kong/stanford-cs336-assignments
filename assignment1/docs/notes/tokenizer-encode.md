# Tokenizer encode 怎么设计

## 2026-11-03

### 问题是什么

`train_bpe` 之后要实现 `Tokenizer.encode`。公开接口有哪些、预分词要不要当构造参数、`special_tokens=None` 算不算没有 special、合并时从左到右扫还是按创建顺序、逐条扫 `merges` 会不会太慢、构造器里要不要建查找表。

### 现象/尝试

一开始把预分词、encode、decode 都想成对外接口。后来确认测试只认讲义那五个公开方法，预分词是 encode 内部步骤，应复用已有模块而不是再传进构造器。

合并一度想成：从左到右扫当前 token，能合就合，再从头扫。担心按清单逐条套规则效率很低。后来改成：只看当前 word 里的相邻对，用 merge 下标当优先级分数，反复合分数最小的那对，直到没有能合的对。两张查找表打算在构造器里建。

### 讨论结论

**公开接口和内部步骤**

测试要的是 `__init__`、`from_files`、`encode`、`encode_iterable`、`decode`。预分词不是公开方法。`get_tokenizer` 只传 `vocab`、`merges`、`special_tokens`。

预分词那条 GPT-2 regex 是作业固定规则，不是训出来的状态。训练和 encode 必须相同，靠调用同一份实现，不要把预分词函数做成构造器入参。`vocab` / `merges` 才是实例状态；`special_tokens` 因实例而异，所以要传入。

**`special_tokens=None`**

表示这次构造没有提供任何 special。编码时不按 special 切，也不往词表追加。词表里碰巧已有某段 bytes，不能据此当成 special。空列表和 `None` 对外行为应一样。赋值 `self.special_tokens = special_tokens` **不会**把 `None` 收成空列表，要自己收；否则后面按列表遍历会出错。

**encode 三段，路径要分开**

1. 按已注册 special 切开，保序。普通段和 special 段都要。
2. 普通段：同一条预分词，每个 pre-token 变成 UTF-8 单字节序列，只在内部 merge。
3. 按 `merges` 的创建顺序应用合体，再查成 token ID。

special 段不预分词、不 merge，整段对应词表里那一个 ID。对 `<|endoftext|>` 跑 regex 会被切碎。

**合并规则：创建顺序，不是最左优先**

`merges` 是有序清单。训练按频率（平局按原始 bytes）**选出**下一对；编码不再比频率、不再比字典序，只照清单执行。一对 `(A, B)` 表示当前相邻的 `A` 然后 `B` 拼成 `A+B`。

从左到右「能合就合」等于最左边那对优先。当前序列里若同时有靠左的后学 pair 和靠右的先学 pair，会合错。不变量：在所有**当前存在**且出现在清单里的相邻对中，选**创建更早**的；同一项出现多次时才从左到右、不重叠。合完邻居变了，再按同一优先级看，不是重新从左抓第一个能合的。

新拼出来的块只能出现在更晚学会的 merge 里，所以贪心取当前最小 rank 不会冒出更小的分数，也不会把早该合的对漏掉。

**不要扫全部 merge 清单**

清单可能上万项，一个 pre-token 往往很短。问「当前这几个邻居在不在表里、下标是多少」，不要问「每一条规则能不能用」。构造时扫一遍 `merges`，建成 pair → 创建下标。没有规则的哨兵大于任何合法下标即可（词表大小在数值上通常够用，概念上表示不能合）。

**同分**

不同 pair 不会撞同一下标。会碰到的是同一 pair 在 word 里出现多次，尤其是重叠。找最小值时从左扫到右、只有更小才更新，最左自然留下，不必再检测「有几个重复」。

**两张表放构造器**

只取决于这份 `vocab` / `merges`，和某次输入无关：

- pair → 创建下标
- bytes → token ID

`encode` 里只切段、预分词、合并、查表。追加 special 时注意不要误改调用方还在用的那份 `vocab`。

**保序与并行**

按 special 切开后逐片做到底（普通片走完预分词+合并+查 ID，special 片直接查 ID），顺序就是原文顺序。并行、大文件切块放在 worker / 切块层，不要编进这段合并。`encode_iterable` 是对可迭代对象逐段 encode、不要先拼成整份，那是调用方式，不是合并算法。

作业测试更卡正确性和 `encode_iterable` 内存，没有训练那种硬速度门槛。先把顺序和 special 路径做对，再用讲义 `'the cat ate'` 和 GPT-2 fixture 核对。

相关笔记：[`预分词前预处理.md`](./预分词前预处理.md)（训练 vs encode 的切段差异）、[`bpe-merge-表更新.md`](./bpe-merge-表更新.md)（训练时单次合体在一个 word 上长什么样）。

### 遗留卡点

encode / `encode_iterable` / `decode` 尚未写。重叠 special 仍以测试为准（更长的整段保留）。建议先用手算：一个同时含「靠左后学 pair」和「靠右先学 pair」的短序列，以及一个会重叠的短 word，确认与「按清单从早到晚套」一致，再接到 `get_tokenizer`。
