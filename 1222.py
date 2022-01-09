import xlrd
import numpy as np
import matplotlib.pyplot as plt

MAX_ERROR = 0.045  # 抬落道最大误差限值


def get_data(data_filename):
    """从excel中读取数据"""
    work_Book = xlrd.open_workbook(data_filename)
    sheet = work_Book.sheet_by_name('Sheet1')
    x_list = []
    y_list = []
    for i in range(1, sheet.nrows):
        cells = sheet.row_values(i)
        # 保留两位小数
        x_list.append(round(float(cells[1]), 2))
        y_list.append(round(float(cells[2]), 2))
    return [x_list, y_list]


def preprocessing(x, y):
    """前处理——间距取50m，间距50m以内的点暂时不考虑"""
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
    """利用求得的二阶差商聚类——通过二阶差商将点分为两类。聚类方法为一维聚类"""
    y_temp = y.copy()  # 深拷贝，防止改变原链表里的数据
    y_temp.sort()
    partition_id_new = 1  # 当面口述吧
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

    return y_temp[partition_id_new]  # 返回聚类的分界限值


def initial_classification(x, y):
    """计算并记录二阶差商"""
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
        if x_label1[i] == 0 and i < len(x_label1)-1:
            temp_list.append(i)
        else:
            if len(temp_list) > 1:
                list_of_line_index1.append(temp_list)
            temp_list = []
    return list_of_line_index1


def plot_figure():
    plt.plot(x_data, y_data, '.k')  # 画出所有数据点
    for i in range(len(list_of_line_index)):
        plt.plot(x_new[list_of_line_index[i][0]:list_of_line_index[i]
                       [-1]+1], predicted_ys[i], '-r')  # 画出每一段的拟合直线
    plt.show()


def get_ab_of_lines(list_of_line_index1):
    reslt = []
    deltas = []
    predict_y = []
    for i in range(len(list_of_line_index1)):
        # 最小二乘开始
        x_column = np.array(
            x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        a = np.column_stack((x_column, np.ones((len(x_column), 1))))
        b = np.array(y_new[list_of_line_index1[i][0]                     :list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        # 最小二乘结束
        y_predicted = x_column*XY[0][0]+XY[1][0]

        delta = abs(y_predicted-b)
        if len(delta) > 2:
            for j in range(1, len(delta)-1):
                if delta[j] > delta[j-1] and delta[j] > delta[j+1] and delta[j] >= MAX_ERROR:
                    x_label[list_of_line_index1[i][j]] = 1
                    return None, None, None

        reslt.append([XY[0][0], XY[1][0]])
        deltas.append(delta)
        predict_y.append((y_predicted.T).tolist()[0])
    return reslt, deltas, predict_y


def check_equal_a():
    for i in range(len(ab_of_each_line)-1):
        if abs(ab_of_each_line[i+1][0]-ab_of_each_line[i][0]) < 3e-3:
            x_column = np.array(
                x_new[list_of_line_index[i][0]:list_of_line_index[i+1][-1]+1]).reshape(-1, 1)
            a = np.column_stack((x_column, np.ones((len(x_column), 1))))
            b = np.array(y_new[list_of_line_index[i][0]                         :list_of_line_index[i+1][-1]+1]).reshape(-1, 1)
            XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
            y_predicted = x_column*XY[0][0]+XY[1][0]
            delta = abs(y_predicted-b)
            if max(delta) < MAX_ERROR:
                x_label[list_of_line_index[i][0]
                    :list_of_line_index[i+1][-1]+1] = 0
                return 0
    return 1


def stable_fit():
    for i in range(len(delatas_of_lines)):
        for j in range(len(delatas_of_lines[i])):
            if delatas_of_lines[i][j] > MAX_ERROR:
                x_label[list_of_line_index[i][j]] = 1
                return None
    return 1


def circle_fit():
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        x_l = x_new[list_of_line_index[i][-1]]
        x_r = x_new[list_of_line_index[i+1][0]]
        if x_intersect > x_l and x_intersect < x_r:
            beta = abs(
                np.arctan(ab_of_each_line[i][0])-np.arctan(ab_of_each_line[i+1][0]))/2
            theta = (
                np.arctan(ab_of_each_line[i][0])+np.arctan(ab_of_each_line[i+1][0]))/2+np.pi/2
            if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                A, B = -np.cos(theta)/np.cos(beta), -np.sin(theta)/np.cos(beta)
            else:
                A, B = np.cos(theta)/np.cos(beta), np.sin(theta)/np.cos(beta)
            l_index = list_of_line_index[i][-1]+1
            r_index = list_of_line_index[i+1][0]
            n = r_index-l_index
            radius = (A*(n*x_intersect-sum(x_new[l_index:r_index]))+B*(n*y_intersect-sum(y_new[l_index:r_index]))) /\
                (A*A+B*B-1)/n
            x0 = x_intersect-A*radius
            y0 = y_intersect-B*radius
            a = ab_of_each_line[i][0]
            b = ab_of_each_line[i][1]
            x_l = (x0-a*b+a*y0)/(1+a*a)
            y_l = (a*x0+a*a*y0+b)/(1+a*a)
            a = ab_of_each_line[i+1][0]
            b = ab_of_each_line[i+1][1]
            x_r = (x0-a*b+a*y0)/(1+a*a)
            y_r = (a*x0+a*a*y0+b)/(1+a*a)
            x_list = np.arange(int(x_l), int(x_r), 1)
            print(radius)
            print(x_list)
            if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                y_list = y0+pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            else:
                y_list = y0-pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            plt.plot(x_list, y_list, '-g')
            plt.plot([x_l,x_r],[y_l,y_r],'-.b')


if __name__ == '__main__':
    [x_data, y_data] = get_data('input.xls')
    x_new, y_new = x_data, y_data
    x_label = np.zeros(len(x_new), dtype='int')
    x_label[0] = 1
    x_label[-1] = 1
    while True:
        while True:
            list_of_line_index = get_lines_indexes(x_label)
            ab_of_each_line, delatas_of_lines, predicted_ys = get_ab_of_lines(
                list_of_line_index)
            if ab_of_each_line != None:
                if stable_fit() == 1:
                    if check_equal_a() == 1:
                        break
        circle_fit()
        plot_figure()
        break

    print('_________end___________')
