import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 数据
flash_1_fwd_1 = np.array([1757, 7208, 14648, 34132])
flash_2_fwd_1 = np.array([1667, 5837, 9227, 33470])
manual_fwd_1 = np.array([20654, 23915, 27506, 43266])

flash_1_fwd_2 = np.array([80656, 92199, 299383, 1366994])
flash_2_fwd_2 = np.array([67266, 91592, 323513, 1232401])
manual_fwd_2 = np.array([65050, 125752, 273406, 579964])

flash_1_fwdbwd = np.array([19245, 58564, 148523, 391909, 1454829]) # 4910635
flash_2_fwdbwd = np.array([16758, 50082, 147481, 380131, 1387472]) # 4974145
manual_fwdbwd = np.array([69251, 96867, 173127, 257154, 527448]) # 1140624

seq_len_fwd_1 = [1, 16, 32, 64]
seq_len_fwd_2 = [128, 256, 512, 1024]
seq_len_fwdbwd = [32, 64, 128, 256, 512] # 1024


# 将微秒(us)转换为毫秒(ms)
flash_1_fwd_1 = flash_1_fwd_1 / 1000
flash_2_fwd_1 = flash_2_fwd_1 / 1000
manual_fwd_1 = manual_fwd_1 / 1000

flash_1_fwd_2 = flash_1_fwd_2 / 1000
flash_2_fwd_2 = flash_2_fwd_2 / 1000
manual_fwd_2 = manual_fwd_2 / 1000

flash_1_fwdbwd = flash_1_fwdbwd / 1000
flash_2_fwdbwd = flash_2_fwdbwd / 1000
manual_fwdbwd = manual_fwdbwd / 1000


# 绘图
def plot_bar_and_line(seq_len, data1, data2, data3, title, labels):
    x = np.arange(len(seq_len))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width, data1, width, label=labels[0], color="#a5a58d")
    rects2 = ax1.bar(x, data2, width, label=labels[1], color="#6d6875")
    rects3 = ax1.bar(x + width, data3, width, label=labels[2], color="#b5838d")

    ax1.set_xlabel("序列长度", fontsize=16)
    ax1.set_ylabel("时延(ms)", fontsize=16)
    ax1.set_title(title, fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_len)
    ax1.legend(loc='upper left', fontsize=13)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 添加折线图
    ax2 = ax1.twinx()
    ax2.plot(x - width, data1, color="#a5a58d", linestyle="--", marker="o")
    ax2.plot(x, data2, color="#6d6875", linestyle="--", marker="o")
    ax2.plot(x + width, data3, color="#b5838d", linestyle="--", marker="o")

    ax2.set_ylim(ax1.get_ylim())
    plt.yticks(fontsize=16)
    fig.tight_layout()
    # plt.show()
    plt.savefig(title + ".jpeg", dpi=1200)

matplotlib.rcParams["font.family"] = "SimSun"

# 绘制三张图
plot_bar_and_line(
    seq_len_fwd_1,
    manual_fwd_1,
    flash_1_fwd_1,
    flash_2_fwd_1,
    "MindSpore框架下注意力正向计算时延",
    ["MindSpore手动实现算法", "注意力优化算法1", "注意力优化算法2"],
)
plot_bar_and_line(
    seq_len_fwd_2,
    manual_fwd_2,
    flash_1_fwd_2,
    flash_2_fwd_2,
    "MindSpore框架下注意力正向计算时延（续）",
    ["MindSpore手动实现算法", "注意力优化算法1", "注意力优化算法2"],
)
plot_bar_and_line(
    seq_len_fwdbwd,
    manual_fwdbwd,
    flash_1_fwdbwd,
    flash_2_fwdbwd,
    "MindSpore框架下注意力正向反向计算时延",
    ["MindSpore手动实现算法", "注意力优化算法1", "注意力优化算法2"],
)
