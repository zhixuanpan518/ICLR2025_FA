import matplotlib.pyplot as plt


with open('attack_result/2_train_adv_accuracy.txt','r') as file:
    sub_acc = [float(line.strip()) for line in file]
with open('attack_result/10_train_adv_accuracy.txt','r') as file:
    super_acc = [float(line.strip()) for line in file]

with open('attack_result/2_train_adv_loss.txt','r') as file:
    sub_loss = [float(line.strip()) for line in file]
with open('attack_result/10_train_adv_loss.txt','r') as file:
    super_loss = [float(line.strip()) for line in file]


# 绘制 local_grad_norms 和 global_grad_norms 的曲线图

plt.figure(figsize=(10, 6))
plt.plot(sub_acc, label="10", color="blue")
plt.plot(super_acc, label="2", color="red")
plt.xticks(ticks=[0, 5, 10, 15, 20])
plt.xlabel("Perturbation Radius")
plt.ylabel("Accuracy")
plt.title("Adversarial Robustness")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("figure/Train Accuracy Adversarial Robustness", dpi=300)


plt.figure(figsize=(10, 6))
plt.plot(sub_loss, label="10", color="blue")
plt.plot(super_loss, label="2", color="red")
plt.xticks(ticks=[0, 5, 10, 15, 20])
plt.xlabel("Perturbation Radius")
plt.ylabel("Loss")
plt.title("Adversarial Robustness")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("figure/Train Loss Adversarial Robustness", dpi=300)




with open('attack_result/10_test_adv_accuracy.txt','r') as file:
    sub_acc = [float(line.strip()) for line in file]
with open('attack_result/2_test_adv_accuracy.txt','r') as file:
    super_acc = [float(line.strip()) for line in file]

with open('attack_result/10_test_adv_loss.txt','r') as file:
    sub_loss = [float(line.strip()) for line in file]
with open('attack_result/2_test_adv_loss.txt','r') as file:
    super_loss = [float(line.strip()) for line in file]



plt.figure(figsize=(10, 6))
plt.plot(sub_acc, label="10", color="blue")
plt.plot(super_acc, label="2", color="red")
plt.xticks(ticks=[0, 5, 10, 15, 20])
plt.xlabel("Perturbation Radius")
plt.ylabel("Accuracy")
plt.title("Adversarial Robustness")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("figure/Test Accuracy Adversarial Robustness", dpi=300)


plt.figure(figsize=(10, 6))
plt.plot(sub_loss, label="10", color="blue")
plt.plot(super_loss, label="2", color="red")
plt.xticks(ticks=[0, 5, 10, 15, 20])
plt.xlabel("Perturbation Radius")
plt.ylabel("Loss")
plt.title("Adversarial Robustness")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("figure/Test Loss Adversarial Robustness", dpi=300)
