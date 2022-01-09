from numpy.core.fromnumeric import size
import xlrd
import numpy as np
import matplotlib.pyplot as plt

MAX_ERROR = 0.045


def get_data(data_filename):
    work_Book = xlrd.open_workbook(data_filename)
    sheet = work_Book.sheet_by_name('Sheet1')
    x_list = []
    y_list = []
    for i in range(1, sheet.nrows):
        cells = sheet.row_values(i)
        x_list.append(round(float(cells[1]), 2))
        y_list.append(round(float(cells[2]), 2))

    return [x_list, y_list]


def preprocessing(x, y):
    n = len(x)
    x_new1 = []
    y_new1 = []
    temp = 0
    for i in range(n):
        if x[i]-temp >= 50:
            x_new1.append(x[i])
            y_new1.append(y[i])
            temp = x[i]
    return [x_new1, y_new1]


def cluster(y):
    y_temp = y.copy()
    y_temp.sort()
    partition_id_new = 1
    while True:
        partition_id_old = partition_id_new
        mean_left = np.mean(y_temp[:partition_id_old])
        mean_right = np.mean(y_temp[partition_id_old:])
        for i in range(len(y_temp)):
            if abs(mean_left-y_temp[i]) < abs(mean_right-y_temp[i]) and \
                    abs(mean_left-y_temp[i+1]) >= abs(mean_right-y_temp[i+1]):
                break
        partition_id_new = i+1
        if partition_id_new == partition_id_old:
            break

    return y_temp[partition_id_new]


def initial_classification(x, y):
    y_record1 = []
    for i in range(1, len(x)-1):
        temp = ((x[i]-x[i-1])*(y[i+1]-y[i])-(x[i+1]-x[i])*(y[i]-y[i-1])) / \
            ((x[i]-x[i-1])*(x[i+1]-x[i])*((x[i]-x[i-1])+(x[i+1]-x[i])))
        y_record1.append(abs(temp))
    return y_record1


def get_lines_indexes(x_label1):
    list_of_line_index1 = []
    temp_list = []
    for i in range(len(x_label1)):
        if x_label1[i] == 0:
            temp_list.append(i)
        else:
            if len(temp_list) > 2:
                list_of_line_index1.append(temp_list)
            temp_list = []
    return list_of_line_index1


def get_ab_of_lines(list_of_line_index1):
    reslt = []
    plt.plot(x_data, y_data, '-b')
    for i in range(len(list_of_line_index1)):
        x_column = np.array(
            x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]]).reshape(-1, 1)
        a = np.column_stack((x_column, np.ones((len(x_column), 1))))
        b = np.array(y_new[list_of_line_index1[i][0]                     :list_of_line_index1[i][-1]]).reshape(-1, 1)
        XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        y_predicted = x_column*XY[0][0]+XY[1][0]

        # break lines
        delta = abs(y_predicted-b)
        if len(delta) > 3:
            counter1 = 0
            for j in range(1, len(delta)-1):
                if delta[j] > delta[j-1] and delta[j] > delta[j+1] and delta[j] >= 5*MAX_ERROR:
                    x_label[list_of_line_index1[i][j]] = 1
                    plt.plot(x_column[j], y_predicted[j], 'og', markersize=7)
                    print('break lines----------------')
                    counter1 += 1
            if counter1 >= 1:
                return None

        # remove and add points
        counter2 = 0
        counter3 = 0
        index_h1 = list_of_line_index1[i][0]
        index_h2 = list_of_line_index1[i][-1]
        if len(delta) > 3:
            while True:
                index_h1 = list_of_line_index1[i][0]-counter2
                d1 = abs(x_new[index_h1]*XY[0][0]+XY[1][0]-y_new[index_h1])
                print(d1)
                if d1 > 3*MAX_ERROR and (index_h2-index_h1) > 3:
                    x_label[index_h1] = 1
                    counter2 += 1
                    break
                elif d1 < 1.5*MAX_ERROR:
                    x_label[index_h1] = 0
                    counter2 += 1
                else:
                    break

        if len(delta) > 3:
            while True:
                index_h2 = list_of_line_index1[i][-1]+counter3
                d2 = abs(x_new[index_h2]*XY[0][0]+XY[1][0]-y_new[index_h2])
                print(d2)
                if d2 >1.5*MAX_ERROR and (index_h2-index_h1) > 3:
                    x_label[index_h2] = 1
                    counter3 += 1
                    break
                elif d1 < 0.5*MAX_ERROR:
                    x_label[index_h2] = 0
                    counter3 += 1
                else:
                    break

        plt.plot(x_new[list_of_line_index1[i][0]:list_of_line_index1[i]
                 [-1]], (y_predicted.T).tolist()[0], '-r')
        reslt.append([XY[0][0], XY[1, 0]])
    if counter2+counter3 > 0:
        print('remove and add points---------------')
        return None
    plt.show()
    return reslt


def check_equal_a():
    couter1 = 0
    for i in range(len(ab_of_each_line)-1):
        if abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0]) < 1e-3:
            x_label[list_of_line_index[i][-1]:list_of_line_index[i+1][0]] = 0
            couter1 += 1
            print(i)
    return couter1


if __name__ == '__main__':
    [x_data, y_data] = get_data('input.xls')
    [x_new, y_new] = preprocessing(x_data, y_data)
    # n_points = len(x_data)
    x_label = np.zeros(len(x_new), dtype='int')
    y_record = initial_classification(x_new, y_new)
    t_v = cluster(y_record)
    # plt.plot(x_data, y_data)
    for i in range(len(y_record)):
        if abs(y_record[i]) > t_v:
            # plt.plot(x_new[i+1], y_new[i+1], 'or')
            x_label[i+1] = 1
    # plt.show()

    while True:
        while True:
            list_of_line_index = get_lines_indexes(x_label)
            ab_of_each_line = get_ab_of_lines(list_of_line_index)
            print("---------while_________")
            if ab_of_each_line != None:
                break
        break
        print('-----------')
        if check_equal_a() == 0:
            break

    print('_________end___________')
