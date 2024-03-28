# graduation-project
🤗CQU计算机学院毕业设计——基于MindSpore的Falcon大模型迁移与性能研究

项目部分为华为昇思MindSpore社区开源任务，基于MindSpore的一个NLP易用开源库MindNLP https://github.com/mindlab-ai/mindnlp

src目录为falcon的MindSpore实现，实验证明该版本实现与HuggingFace中模型实现在容纳误差为$`10^{-3}`$的前提下两者等效，完整模型已上传至modelscope社区 https://www.modelscope.cn/models/mindnlp/falcon-rw-1b/summary ，包括模型实现、所需配置文件以及转换好的预训练权重文件。
单元测试代码见test目录。

train_falcon为在MindNLP框架中基于falcon-rw-1b预训练模型进行微调的代码，数据集为GLUE基准数据集中的MRPC语料，任务是语义匹配。

FlashAttention kernel用CUDA编写，参考了代码https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu，目前为只支持静态block size与FP32数据格式
