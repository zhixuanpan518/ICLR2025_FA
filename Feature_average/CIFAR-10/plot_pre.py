import matplotlib.pyplot as plt


# with open('pre_attack_result/10_train_adv_accuracy.txt','r') as file:
#     sub_acc = [float(line.strip()) for line in file]
# with open('pre_attack_result/train_adv_accuracy.txt','r') as file:
#     linear_acc = [float(line.strip()) for line in file]
# with open('pre_attack_result/2_train_adv_accuracy.txt','r') as file:
#     super_acc = [float(line.strip()) for line in file]

# with open('pre_attack_result/10_train_adv_loss.txt','r') as file:
#     sub_loss = [float(line.strip()) for line in file]
# with open('pre_attack_result/2_train_adv_loss.txt','r') as file:
#     super_loss = [float(line.strip()) for line in file]
# with open('pre_attack_result/train_adv_loss.txt','r') as file:
#     linear_loss = [float(line.strip()) for line in file]

# # 绘制 local_grad_norms 和 global_grad_norms 的曲线图

# plt.figure(figsize=(10, 6))
# plt.plot(sub_acc, label="10", color="blue")
# plt.plot(super_acc, label="2", color="red")
# # plt.plot(linear_acc, label="linear", color="green")
# plt.xticks(ticks=[0, 5, 10, 15, 20])
# plt.xlabel("Perturbation Radius")
# plt.ylabel("Accuracy")
# plt.title("Adversarial Robustness")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("figure/Pre Train Accuracy Adversarial Robustness", dpi=300)


# plt.figure(figsize=(10, 6))
# plt.plot(sub_loss, label="10", color="blue")
# plt.plot(super_loss, label="2", color="red")
# # plt.plot(linear_loss, label="linear", color="green")
# plt.xticks(ticks=[0, 5, 10, 15, 20])
# plt.xlabel("Perturbation Radius")
# plt.ylabel("Loss")
# plt.title("Adversarial Robustness")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("figure/Pre Train Loss Adversarial Robustness", dpi=300)




with open('pre_attack_result/10_test_adv_accuracy.txt','r') as file:
    sub_acc = [float(line.strip()) for line in file]
sub_acc = sub_acc[:21]
with open('pre_attack_result/2_test_adv_accuracy.txt','r') as file:
    super_acc = [float(line.strip()) for line in file]
super_acc = super_acc[:21]
# with open('pre_attack_result/test_adv_accuracy.txt','r') as file:
#     linear_acc = [float(line.strip()) for line in file]

# with open('attack_result/10_test_adv_loss.txt','r') as file:
#     sub_loss = [float(line.strip()) for line in file]
# with open('attack_result/2_test_adv_loss.txt','r') as file:
#     super_loss = [float(line.strip()) for line in file]
# with open('pre_attack_result/test_adv_loss.txt','r') as file:
#     linear_loss = [float(line.strip()) for line in file]

plt.rcParams['axes.facecolor'] = '#eaeaf2'    
plt.plot(sub_acc, label="10", color="blue")
plt.plot(super_acc, label="2", color="red")
plt.xlabel("Perturbation Radius", fontsize=20) #,fontsize=13)
plt.ylabel('Robust Test Accuracy', fontsize=16) #r'$\operatorname{max}_{r}\frac{\langle w_{i,r}, u_i \rangle}{\|u_i\|} \quad or \quad \operatorname{max}_{r}\frac{\langle w_{i,r}, v_i \rangle}{\|v_i\|}, \quad i = 1,2$')#,fontsize=13)
plt.title("CIFAR-10 (Pre-Trained)",fontsize=20)
xticks = [0, 4, 8, 12, 16, 20]
xtick_labels = ['0.0', '0.8', '1.6', '2.4', '3.2', '4.0']
plt.xticks(xticks, xtick_labels)
#plt.yscale("log")
  #'gainsboro'#'azure'#'lavender'#'#f0f0f0'
plt.grid(color = 'white')
# plt.grid()
plt.legend(loc = 'lower left', fontsize=16)
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.tight_layout()
#保存pdf
plt.savefig("figure/CIFAR10.pdf",format="pdf")
plt.show()



# plt.figure(figsize=(10, 6))

# # plt.plot(linear_acc, label="linear", color="green")
# plt.xticks(ticks=[0, 10, 20, 30, 40])
# plt.xlabel("Perturbation Radius")
# plt.ylabel("Accuracy")
# plt.title("Adversarial Robustness")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("figure/Pre Test Accuracy Adversarial Robustness.pdf", format='pdf')


# plt.figure(figsize=(10, 6))
# plt.plot(sub_loss, label="10", color="blue")
# plt.plot(super_loss, label="2", color="red")
# # plt.plot(linear_loss, label="linear", color="green")
# plt.xticks(ticks=[0, 10, 20, 30, 40])
# plt.xlabel("Perturbation Radius")
# plt.ylabel("Loss")
# plt.title("Adversarial Robustness")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("figure/Pre Test Loss Adversarial Robustness", dpi=300)
