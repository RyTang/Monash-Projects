import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def get_loss_values(filename="outputs/sentiments_train_metric_log_avg.json"):
    total_loss, box_reg_loss, classifier_loss = [], [], []
    with open(filename, "r") as json_file:
        logs = json.loads(json_file.read())
        for epoch in logs:
            log = logs[epoch]
            total_loss.append(float(log["loss"]))
            box_reg_loss.append(float(log["loss_box_reg"]))
            classifier_loss.append(float(log["loss_classifier"]))
    return total_loss, box_reg_loss, classifier_loss


def plot_loss_values(
    total_loss: List[float], box_reg_loss: List[float], classifier_loss: List[float]
):
    # create plot
    fig, ax = plt.subplots()
    # create lines
    (total_loss_line,) = ax.plot(total_loss, label="Total Loss", marker="o")
    (box_reg_loss_line,) = ax.plot(box_reg_loss, label="Box Reg Loss", marker="o")
    (classifier_loss_line,) = ax.plot(
        classifier_loss, label="Classifier Loss", marker="o"
    )
    # set x tick to start from 1
    plt.xticks(np.arange(len(total_loss)), np.arange(1, len(total_loss)+1))
    # set x, y label
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss value")
    # set title
    ax.set_title("Average Loss Values for Text Faster RCNN")
    # create legend
    ax.legend(loc="upper right")
    # show the plot
    plt.show()


if __name__ == "__main__":
    total_loss, box_reg_loss, classifier_loss = get_loss_values()
    plot_loss_values(total_loss, box_reg_loss, classifier_loss)
