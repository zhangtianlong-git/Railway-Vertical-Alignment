import xlrd
import numpy as np
import matplotlib.pyplot as plt

MAX_ERROR = 0.045  # 抬落道最大误差限值
LIST_ITERATION_ORDER = [1,2,3]
def update_list_order(x):
    global LIST_ITERATION_ORDER
    LIST_ITERATION_ORDER = LIST_ITERATION_ORDER[1:]
    LIST_ITERATION_ORDER.append(x)


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
    """
    x_label1是聚类后划分得到的点标签，label=1代表划为曲线段，label=2代表直线段。
    此函数的目标，比如 x_label = [0 0 0 0 1 1 0 0 0 1 1 1 1 0 0 0 0],得到直线段的下标列表组
    返回如下形式的列表 list_of_line_index1 = [[1 2 3 4], [7 8 9], [14 15 16 17]]
    这样可以方便索引到每条直线上的每个点
    """
    list_of_line_index1 = []
    temp_list = []
    for i in range(len(x_label1)):
        if x_label1[i] == 0 and i < len(x_label1)-1:
            temp_list.append(i)
        else:
            if len(temp_list) > 1 and len(temp_list)<len:
                list_of_line_index1.append(temp_list)
            temp_list = []
    return list_of_line_index1


def get_ab_of_lines(list_of_line_index1):
    """
    获得每条直线的a和b，其中y=ax+b。注意这个函数用到了全局变量x_new、y_new和list_of_line_index。
    最后每条直线的a和b也用链表返回，格式为 [[a1 b1],[a2 b2],[a3 b3]......] 拟合方法为最小二乘法。
    此函数还包括异常点的打断，见程序的第二段
    """
    reslt = []
    plt.plot(x_data, y_data, '-b')  # 画出所有数据点
    for i in range(len(list_of_line_index1)):
        # 最小二乘开始
        x_column = np.array(
            x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        a = np.column_stack((x_column, np.ones((len(x_column), 1))))
        b = np.array(y_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        # 最小二乘结束
        y_predicted = x_column*XY[0][0]+XY[1][0]

        # 检测是否存在聚类效果不好导致直线拟合有问题的地方，当拟合直线存在拐点，
        # 并且该拐点大于最大阈值的几倍时，将该点的x_label设为1，函数直接返回None，
        # 重新计算各新分线段的a和b，直到没有异常点才返回非None的结果链表
        delta = abs(y_predicted-b)
        if len(delta) > 2:
            for j in range(1, len(delta)-1):
                if delta[j] > delta[j-1] and delta[j] > delta[j+1] and delta[j] >= MAX_ERROR:
                    print(
                        '-------------------------------------------delta is %f' % (delta[j]))
                    x_label[list_of_line_index1[i][j]] = 1
                    plt.plot(x_column[j], y_predicted[j],
                             'og', markersize=7)  # 画出异常点
                    print('break lines----------------')
                    # update_list_order(list_of_line_index[i][j])
                    # if np.std(LIST_ITERATION_ORDER) == 0:
                    #     if len(list_of_line_index1[i])<len(list_of_line_index1[i+1]):
                    #         x_label[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1]=1
                    #     else:
                    #         x_label[list_of_line_index1[i+1][0]:list_of_line_index1[i+1][-1]+1]=1
                    #     return None
                    return None

        plt.plot(x_new[list_of_line_index1[i][0]:list_of_line_index1[i]
                       [-1]+1], (y_predicted.T).tolist()[0], '-r')  # 画出每一段的拟合直线
        reslt.append([XY[0][0], XY[1][0]])
    print(LIST_ITERATION_ORDER)
    plt.show()  # 显示图像，注释后不显示图像
    return reslt


def check_equal_a():
    """
    用到了全局变量x_label，如果相邻直线的坡度差小于千分之一，将它们之间的x_label全设为0
    """
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = a_plus_1 = ab_of_each_line[i][0]
        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        x_l = x_new[list_of_line_index[i][-1]+1]
        x_r = x_new[list_of_line_index[i+1][0]-1]
        y_up = max(y_new[list_of_line_index[i][-1]+1:list_of_line_index[i+1][0]])
        y_low = min(y_new[list_of_line_index[i][-1]+1:list_of_line_index[i+1][0]])
        if_x_in = x_intersect>=x_l and x_intersect<=x_r
        if_y_in = y_intersect>=y_up or y_intersect<=y_low
        # if abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0]) < 1e-3:
        #     x_label[list_of_line_index[i][0]:list_of_line_index[i+1][-1]] = 0
        #     return 1
        if not (if_x_in and if_y_in):
            x_label[list_of_line_index[i][0]:list_of_line_index[i+1][-1]] = 0
            return 1
    return 0


def stable_fit():
    for i in range(len(list_of_line_index)):
        # 最小二乘开始
        x_column = np.array(
            x_new[list_of_line_index[i][0]:list_of_line_index[i][-1]+1]).reshape(-1, 1)
        a = np.column_stack((x_column, np.ones((len(x_column), 1))))
        b = np.array(y_new[list_of_line_index[i][0]:list_of_line_index[i][-1]+1]).reshape(-1, 1)
        XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        # 最小二乘结束
        y_predicted = x_column*XY[0][0]+XY[1][0]
        delta = abs(y_predicted-b)
        std_err = np.std(delta)
        if std_err > 0:
            for j in range(1, len(delta)-1):
                if delta[j] >= MAX_ERROR:
                    x_label[list_of_line_index[i][j]] = 1
                    return None
    return True


if __name__ == '__main__':
    [x_data, y_data] = get_data('input.xls')
    x_new, y_new = x_data, y_data
    # [x_new, y_new] = preprocessing(x_data, y_data)
    x_label = np.zeros(len(x_new), dtype='int')
    # y_record = initial_classification(x_new, y_new)
    # t_v = cluster(y_record) # 聚类阈值，大于阈值的暂定为曲线段，lable=1
    # for i in range(len(y_record)):
    #     if abs(y_record[i]) > t_v:
    #         x_label[i+1] = 1
    # plt.show()

    # 两层while，内层为直线段拟合、打断、重构；外层为合并相似坡度
    while True:
        while True:
            list_of_line_index = get_lines_indexes(x_label)
            ab_of_each_line = get_ab_of_lines(list_of_line_index)
            print("---------while_________")
            if ab_of_each_line != None:
                stable_fit()
                break
        # break
        if check_equal_a() == 0:
            break

    print('_________end___________')
