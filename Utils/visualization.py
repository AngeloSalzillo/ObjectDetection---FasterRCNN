import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_detections(image, boxes, labels, scores, idx_to_class):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        class_name = idx_to_class.get(label, "Unknown")
        ax.text(
            xmin,
            ymin - 10,
            f"{class_name}: {score:.2f}",
            color="red",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    ax.axis("off")
    plt.show()
