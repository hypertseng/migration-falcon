import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["font.family"] = "SimSun"
# 提取的数据
train_loss = [
    0.5578398527922453,
    0.3285983216528799,
    0.19387631218937226,
    0.10808298946206087,
    0.08010279171347358,
    0.047642045031445736,
    0.022214503070108253,
    0.00996485359009055,
    0.016036289190155227,
    0.016734968862762118,
]
train_acc = [
    0.7219193020719739,
    0.8647764449291166,
    0.9252998909487459,
    0.9621046892039259,
    0.9716466739367503,
    0.982824427480916,
    0.9929116684841875,
    0.9964558342420938,
    0.9956379498364231,
    0.9934569247546347,
]
train_f1 = [
    0.8045228056726714,
    0.901195219123506,
    0.9448025785656728,
    0.9720603015075377,
    0.978972907399919,
    0.9872495446265938,
    0.9947411003236246,
    0.9973721447341823,
    0.9967637540453075,
    0.9951475940153659,
]
eval_loss = [
    0.5473138314706308,
    0.777088447853371,
    0.7137975339536313,
    0.6379048382794416,
    0.8745307215937862,
    0.8515983864113137,
    0.9174031151665581,
    0.9698032449792933,
    0.8814156850179037,
    0.9664180896900318,
]
eval_acc = [
    0.736231884057971,
    0.7808695652173913,
    0.8168115942028985,
    0.8179710144927537,
    0.7681159420289855,
    0.8359420289855073,
    0.8388405797101449,
    0.8318840579710145,
    0.832463768115942,
    0.8336231884057971,
]
eval_f1 = [
    0.832535885167464,
    0.8546153846153847,
    0.8743038981702467,
    0.8588129496402876,
    0.8044965786901271,
    0.8802369868810834,
    0.8841666666666667,
    0.8788638262322475,
    0.8773865082732287,
    0.874617737003058,
]

epochs = list(range(1, len(train_loss) + 1))

plt.figure(figsize=(14, 6))

# 绘制训练数据
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="训练集损失", marker="o", color="blue")
plt.plot(epochs, train_acc, label="训练集准确率", marker="o", color="green")
plt.plot(epochs, train_f1, label="训练集F1值", marker="o", color="purple")
plt.xlabel("Epochs", fontsize=16, fontname="Times New Roman")
plt.ylabel("值", fontsize=16)
plt.title("训练指标", fontsize=20)
plt.legend()

# 绘制验证数据
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_loss, label="验证集损失", marker="o", color="red")
plt.plot(epochs, eval_acc, label="验证集准确率", marker="o", color="orange")
plt.plot(epochs, eval_f1, label="验证集F1值", marker="o", color="brown")
plt.xlabel("Epochs", fontsize=16, fontname="Times New Roman")
plt.ylabel("值", fontsize=16)
plt.title("验证指标", fontsize=20)
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("LoRA_record.jpeg", dpi=1200)
