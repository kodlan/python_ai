import numpy as np
import csv
import matplotlib.pyplot as plt

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

def generate(count, filename):
    np.random.seed(1)

    x = np.random.multivariate_normal([0, 0], [[1, .75], [0.75, 1]], count)
    y = np.random.multivariate_normal([1, 5], [[1, .75], [0.75, 1]], count)

    points = np.vstack((x, y)).astype(np.float32)
    labels = np.hstack((np.zeros(count), np.ones(count)))

    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], c=labels, alpha=.4)
    plt.show()

    csv_writer(points, filename)

if __name__ == "__main__":
    generate(3000, "train.csv")
    generate(200, "test.csv")
