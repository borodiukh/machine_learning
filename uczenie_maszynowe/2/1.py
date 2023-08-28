import csv
import matplotlib.pyplot as plt

with open('data2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    column_two = []
    column_three = []
    for row in csv_reader:
        column_two.append(float(row[1]))
        column_three.append(float(row[2]))
    print(column_two)
    print(column_three)

    plt.plot(column_two, column_three, 'o')
    plt.show()
