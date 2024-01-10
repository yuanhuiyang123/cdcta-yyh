import numpy as np
from matplotlib import pyplot as plt


def plot_abundance(ground_truth, estimated, em, save_dir):

    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')

    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
    plt.tight_layout()

    plt.savefig(save_dir + "abundance.png")
def plot_estimated_abundance(estimated, em, save_dir):
    for i in range(em):
        plt.figure(figsize=(340/100, 340/100), dpi=100)  # 因为matplotlib以英寸为单位，需要按照dpi转换为340像素大小
        plt.imshow(estimated[:, :, i], cmap='jet', aspect='auto')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(f"{save_dir}/abundance_endmember{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_endmembers(target, pred, em, save_dir):

    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i], label="Extracted")
        plt.plot(target[:, i], label="GT")
        plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_dir + "end_members.png")
