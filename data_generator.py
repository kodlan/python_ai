import numpy as np
import csv
import matplotlib.pyplot as plt

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


if __name__ == "__main__":

    x = np.arange(-3.14, 3.14, 0.01)
    y = np.sin(x)

    plt.scatter(x, y)
    plt.show()

    # generate list of [x, y] values
    data = map(lambda (i,j): [i, j], zip(x, y))

    csv_writer(data, "output.csv")