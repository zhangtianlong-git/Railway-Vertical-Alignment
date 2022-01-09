from numpy.core.fromnumeric import mean
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xw

MAX_ERROR = 0.04  # 抬落道最大误差限值


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
            if len(temp_list) > 2:
                list_of_line_index1.append(temp_list)
            temp_list = []
    return list_of_line_index1


def plot_figure():
    for i in range(len(list_of_line_index)):
        plt.plot(x_new[list_of_line_index[i][0]:list_of_line_index[i]
                       [-1]+1], predicted_ys[i], '-r',linewidth=2)  # 画出每一段的拟合直线
    plt.plot(x_new, y_new, '.k')  # 画出所有数据点
    plt.get_current_fig_manager().window.geometry("1500x800+120+100")
    plt.ylim([41,53])
    plt.xlim([97000,103000])
    # plt.plot(x_plot_break,y_plot_break,'og',markersize = 7)
    plt.show()


def get_ab_of_lines(list_of_line_index1, para=1,final = False):
    reslt = []
    deltas = []
    predict_y = []
    for i in range(len(list_of_line_index1)):
        # 最小二乘开始
        x_column = np.array(
            x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        a = np.column_stack((x_column, np.ones((len(x_column), 1))))
        b = np.array(y_new[list_of_line_index1[i][0]
                     :list_of_line_index1[i][-1]+1]).reshape(-1, 1)
        XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        #最小二乘结束
        if XY[0][0]>6e-3 and final == True:
            XY[0][0] = 6e-3
            XY[1][0] = mean(b-x_column*XY[0][0])
        elif XY[0][0]<-6e-3 and final == True:
            XY[0][0]=-6e-3
            XY[1][0] = mean(b-x_column*XY[0][0])


        # if i>0 and i < len(list_of_line_index1)-1 and final == True:
        #     x_column_n = np.array(
        #     x_new[list_of_line_index1[i+1][0]:list_of_line_index1[i+1][-1]+1]).reshape(-1, 1)
        #     a_n = np.column_stack((x_column_n, np.ones((len(x_column_n), 1))))
        #     b_n = np.array(y_new[list_of_line_index1[i+1][0]:list_of_line_index1[i+1][-1]+1]).reshape(-1, 1)
        #     XY_n = np.dot(np.linalg.inv(np.dot(a_n.T, a_n)), np.dot(a_n.T, b_n))

        #     x_column_l = np.array(
        #     x_new[list_of_line_index1[i-1][0]:list_of_line_index1[i-1][-1]+1]).reshape(-1, 1)
        #     a_l = np.column_stack((x_column_l, np.ones((len(x_column_l), 1))))
        #     b_l = np.array(y_new[list_of_line_index1[i-1][0]:list_of_line_index1[i-1][-1]+1]).reshape(-1, 1)
        #     XY_l = np.dot(np.linalg.inv(np.dot(a_l.T, a_l)), np.dot(a_l.T, b_l))

        #     if XY_n[0][0]-XY[0][0]<-10e-3:
        #         XY[0][0] -= 1e-4
        #         print(i)
        #         XY[1][0] = mean(b-x_column*XY[0][0])
        #     if XY[0][0]-XY_l[0][0]<-10e-3:
        #         XY[0][0] += 0.5e-4
        #         XY[1][0] = mean(b-x_column*XY[0][0])


        y_predicted = x_column*XY[0][0]+XY[1][0]
        # plt.plot(x_column,y_predicted,'-r',linewidth=2)

        delta = abs(y_predicted-b)
        if len(delta) > 2:
            for j in range(1, len(delta)-1):
                if delta[j] > delta[j-1] and delta[j] > delta[j+1] and delta[j] >= para*MAX_ERROR:
                    x_label[list_of_line_index1[i][j]] = 1
                    # plt.plot(x_new,y_new,'.k')
                    # plt.plot(x_plot_break,y_plot_break,'og',markersize = 7)
                    # plt.plot(x_new[list_of_line_index1[i][j]],y_new[list_of_line_index1[i][j]],'ob',markersize = 7)
                    # x_plot_break.append(x_new[list_of_line_index1[i][j]])
                    # y_plot_break.append(y_new[list_of_line_index1[i][j]])
                    # plt.get_current_fig_manager().window.geometry("1500x800+120+100")
                    # plt.ylim([41,53])
                    # plt.xlim([97000,103000])
                    # plt.show()
                    return None, None, None
        reslt.append([XY[0][0], XY[1][0]])
        deltas.append(delta)
        predict_y.append((y_predicted.T).tolist()[0])
    return reslt, deltas, predict_y


def check_equal_a(para=1, thre=3e-3, final=False):
    for i in range(len(ab_of_each_line)-1):
        if abs(ab_of_each_line[i+1][0]-ab_of_each_line[i][0]) < thre:
            x_column = np.array(
                x_new[list_of_line_index[i][0]:list_of_line_index[i+1][-1]+1]).reshape(-1, 1)
            a = np.column_stack((x_column, np.ones((len(x_column), 1))))
            b = np.array(y_new[list_of_line_index[i][0]:list_of_line_index[i+1][-1]+1]).reshape(-1, 1)
            XY = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
            if XY[0][0]>6e-3 and final == True:
                print(XY[0][0])
                XY[0][0] = 6e-3
                XY[1][0] = mean(b-x_column*XY[0][0])
            elif XY[0][0]<-6e-3 and final == True:
                print(XY[0][0])
                XY[0][0]=-6e-3
                XY[1][0] = mean(b-x_column*XY[0][0])
            y_predicted = x_column*XY[0][0]+XY[1][0]
            delta = abs(y_predicted-b)
            if max(delta) < para*MAX_ERROR:
                x_label[list_of_line_index[i][0]:list_of_line_index[i+1][-1]+1] = 0
                return 0
    return 1


def stable_fit():
    for i in range(len(delatas_of_lines)):
        if delatas_of_lines[i][0] > MAX_ERROR:
            x_label[list_of_line_index[i][0]] = 1
            return None
        if delatas_of_lines[i][-1] > MAX_ERROR:
            x_label[list_of_line_index[i][-1]] = 1
            return None
        a = ab_of_each_line[i][0]
        b = ab_of_each_line[i][1]
        delta_l = abs(x_new[list_of_line_index[i][0]-1] *
                      a+b-y_new[list_of_line_index[i][0]-1])
        delta_r = abs(x_new[list_of_line_index[i][-1]+1] *
                      a+b-y_new[list_of_line_index[i][-1]+1])
        if list_of_line_index[i][0]-2 >= 0 and delta_l < MAX_ERROR and x_label[list_of_line_index[i][0]-2] == 1:
            x_label[list_of_line_index[i][0]-1] = 0
            return None
        if list_of_line_index[i][-1]+2 < len(x_label) and x_label[list_of_line_index[i][-1]+2] == 1 and delta_r < MAX_ERROR:
            x_label[list_of_line_index[i][-1]+1] = 0
            return None
    return 1


def circle_fit():
    change = 0
    old_x_label = x_label.copy()
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        x_l = x_new[list_of_line_index[i][-1]]
        x_r = x_new[list_of_line_index[i+1][0]]
        if abs(a_plus_1-a)>3e-3:
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
            radius = -(A*(n*x_intersect-sum(x_new[l_index:r_index])) +
                       B*(n*y_intersect-sum(y_new[l_index:r_index]))) /\
                (A*A+B*B-1)/n
            radius *= 1.4
            if radius > 0 and radius < 10000:
                radius = 10000
            if radius > 40000:
                radius = 40000
            # radius=(radius%100+3)*100
            x0 = x_intersect+A*radius
            y0 = y_intersect+B*radius
            a = ab_of_each_line[i][0]
            b = ab_of_each_line[i][1]
            x_l = (x0-a*b+a*y0)/(1+a*a)
            a = ab_of_each_line[i+1][0]
            b = ab_of_each_line[i+1][1]
            x_r = (x0-a*b+a*y0)/(1+a*a)
            if radius > 0:
                left_index = list_of_line_index[i][0]
                right_index = list_of_line_index[i+1][-1]+1
                for i in range(left_index, right_index):
                    if x_new[i] < x_l and x_label[i] == 1 and old_x_label[i] == x_label[i]:
                        x_label[i] = 0
                        if sum(x_label[left_index:right_index]) == 0:
                            x_label[i] = 1
                        else:
                            print('i is %d, x is %f'%(i,x_new[i]))
                            change = 1
                    elif x_new[i] > x_l and x_new[i] < x_r and x_label[i] == 0 and old_x_label[i] == x_label[i]:
                        x_label[i] = 1
                        if sum(x_label[left_index:right_index])==0:
                            x_label[i] = 0
                        else:
                            print('i is %d, x is %f'%(i,x_new[i]))
                            change = 1
                    elif x_new[i] > x_r and x_label[i] == 1 and old_x_label[i] == x_label[i]:
                        x_label[i] = 0
                        if sum(x_label[left_index:right_index])==0:
                            x_label[i] = 1
                        else:
                            print('i is %d, x is %f'%(i,x_new[i]))
                            change = 1
    if change == 0:
        return 1
    else:
        return 0


def circle_fit_plot():
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        x_l = x_new[list_of_line_index[i][-1]]
        x_r = x_new[list_of_line_index[i+1][0]]
        if abs(a_plus_1-a)>3e-3:
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
            radius = -(A*(n*x_intersect-sum(x_new[l_index:r_index])) +
                       B*(n*y_intersect-sum(y_new[l_index:r_index]))) /\
                (A*A+B*B-1)/n
            radius *= 1.4
            if radius > 0 and radius < 10000:
                radius = 10000
            if radius > 40000:
                radius = 40000
            # radius=(radius%100+3)*100
            x0 = x_intersect+A*radius
            y0 = y_intersect+B*radius
            a = ab_of_each_line[i][0]
            b = ab_of_each_line[i][1]
            x_l = (x0-a*b+a*y0)/(1+a*a)
            a = ab_of_each_line[i+1][0]
            b = ab_of_each_line[i+1][1]
            x_r = (x0-a*b+a*y0)/(1+a*a)
            x_list = np.arange(x_l, x_r, 1)
            if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                y_list = y0+pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            else:
                y_list = y0-pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            if radius > 5000:
                plt.plot(x_list, y_list, '-g',linewidth=3)


def merge_shor_lines():
    state = 0
    for i in range(len(ab_of_each_line)-2):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect_l = (b_plus_1-b)/(a-a_plus_1)
        b_plus_1 = ab_of_each_line[i+2][1]
        b = ab_of_each_line[i+1][1]
        a_plus_1 = ab_of_each_line[i+2][0]
        a = ab_of_each_line[i+1][0]
        x_intersect_r = (b_plus_1-b)/(a-a_plus_1)
        if (x_intersect_r-x_intersect_l) < 350:
            x_label[list_of_line_index[i+1][0:]] = 1
            state = 1
    if state == 0:
        return 1
    else:
        return 0


def points_owners():
    change = 0
    for i in range(len(ab_of_each_line)-1):
        if abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0]) < 3e-3:
            b_plus_1 = ab_of_each_line[i+1][1]
            b = ab_of_each_line[i][1]
            a_plus_1 = ab_of_each_line[i+1][0]
            a = ab_of_each_line[i][0]
            x_intersect = (b_plus_1-b)/(a-a_plus_1)
            x_l_l = x_new[list_of_line_index[i][0]]
            x_l_r = x_new[list_of_line_index[i][-1]]
            x_r_l = x_new[list_of_line_index[i+1][0]]
            x_r_r = x_new[list_of_line_index[i+1][-1]]
            if x_intersect > x_l_r and x_intersect < x_r_l and list_of_line_index[i+1][0]-list_of_line_index[i][-1] > 2:
                l_index = list_of_line_index[i][-1]+1
                r_index = list_of_line_index[i+1][0]
                temp_list = x_new[l_index:r_index]
                temp_list_t = (abs(np.array(temp_list)-x_intersect)).tolist()
                find_index = temp_list_t.index(min(temp_list_t))+l_index
                x_label[find_index] = 1
                x_label[l_index:find_index] = 0
                x_label[find_index+1:r_index] = 0
                print('middle')
                change = 1
            elif x_intersect > x_l_l and x_intersect < x_l_r:
                l_index = list_of_line_index[i][0]
                r_index = list_of_line_index[i][-1]+1
                temp_list = x_new[l_index:r_index]
                temp_list_t = (abs(np.array(temp_list)-x_intersect)).tolist()
                find_index = temp_list_t.index(min(temp_list_t))+l_index
                x_label[find_index] = 1
                x_label[l_index:find_index] = 0
                x_label[find_index+1:list_of_line_index[i+1][0]] = 0
                print('left')
                plt.plot(x_new[find_index],y_new[find_index],'og',markersize=7)
                plot_figure()
                change = 1
            elif x_intersect > x_r_l and x_intersect < x_r_r:
                l_index = list_of_line_index[i+1][0]
                r_index = list_of_line_index[i+1][-1]+1
                temp_list = x_new[l_index:r_index]
                temp_list_t = (abs(np.array(temp_list)-x_intersect)).tolist()
                find_index = temp_list_t.index(min(temp_list_t))+l_index
                x_label[find_index] = 1
                x_label[list_of_line_index[i][-1]+1:find_index] = 0
                x_label[find_index+1:r_index] = 0
                print('right')
                change = 1
    if change == 0:
        return 1
    else:
        return 0


def merge_short(final = False):
    change = 0
    for i in range(1, len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect_next = (b_plus_1-b)/(a-a_plus_1)

        b_plus_1 = ab_of_each_line[i][1]
        b = ab_of_each_line[i-1][1]
        a_plus_1 = ab_of_each_line[i][0]
        a = ab_of_each_line[i-1][0]
        x_intersect_last = (b_plus_1-b)/(a-a_plus_1)

        if x_intersect_next-x_intersect_last < 350 and abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0])+abs(ab_of_each_line[i][0]-ab_of_each_line[i-1][0]) > 3e-3:
            x_label[list_of_line_index[i][0]:list_of_line_index[i][-1]+1] = 1
            change = 1
        
        if abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0]) > 3e-3 and ab_of_each_line[i][0]*ab_of_each_line[i+1][0]<0 and x_new[list_of_line_index[i][-1]]-x_new[list_of_line_index[i][0]]>800 and final == True:
            for j in range(len(list_of_line_index[i])):
                if x_new[list_of_line_index[i][-1]]-x_new[list_of_line_index[i][j]] <450:
                    x_label[list_of_line_index[i][j]] = 1
                    print(x_new[list_of_line_index[i][j]])
                    break
        
        if x_intersect_next-x_intersect_last < 350 and final == True:
            if len(list_of_line_index[i-1])>len(list_of_line_index[i+1]):
                for j in range(len(list_of_line_index[i-1])):
                    if x_new[list_of_line_index[i-1][-1]]-x_new[list_of_line_index[i-1][j]] <150:
                        x_label[list_of_line_index[i-1][j]] = 1
                        x_label[list_of_line_index[i-1][j]+1:list_of_line_index[i][0]] = 0
                        break
            else:
                for j in range(len(list_of_line_index[i+1])):
                    if x_new[list_of_line_index[i+1][j]]-x_new[list_of_line_index[i+1][0]] > 150:
                        x_label[list_of_line_index[i+1][j]] = 1
                        x_label[list_of_line_index[i][0]:list_of_line_index[i+1][j]] = 0
                        break
    if change == 0:
        return 1
    else:
        return 0

def plot_short():
    for i in range(1, len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect_next = (b_plus_1-b)/(a-a_plus_1)

        b_plus_1 = ab_of_each_line[i][1]
        b = ab_of_each_line[i-1][1]
        a_plus_1 = ab_of_each_line[i][0]
        a = ab_of_each_line[i-1][0]
        x_intersect_last = (b_plus_1-b)/(a-a_plus_1)

        if x_intersect_next-x_intersect_last < 350:
            print('x_intersect_next-x_intersect_last is %f'%(x_intersect_next-x_intersect_last))
            plt.plot(x_new[list_of_line_index[i][0]:list_of_line_index[i]
                       [-1]+1], predicted_ys[i], '^g',markersize = 10)  # 画出每一段的拟合直线
        if abs(ab_of_each_line[i][0]-ab_of_each_line[i+1][0]) > 3e-3:
            plt.plot(x_new[list_of_line_index[i][0]:list_of_line_index[i]
                       [-1]+1], predicted_ys[i], 'oy',markersize = 5)  # 画出每一段的拟合直线

def output():
    bpd = []
    radiuses = []
    delta_alpha = []
    alpha = []
    y_predicts = []
    y_predicts.append(0)
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        for j in range(len(list_of_line_index[i])):
            y_predicts.append(predicted_ys[i][j])
        if abs(a_plus_1-a)>3e-3:
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
            radius = -(A*(n*x_intersect-sum(x_new[l_index:r_index])) +
                       B*(n*y_intersect-sum(y_new[l_index:r_index]))) /\
                (A*A+B*B-1)/n
            radius *= 1.4
            if radius > 0 and radius < 10000:
                radius = 10000
            if radius > 40000:
                radius = 40000
            x0 = x_intersect+A*radius
            y0 = y_intersect+B*radius
            a = ab_of_each_line[i][0]
            b = ab_of_each_line[i][1]
            x_l = (x0-a*b+a*y0)/(1+a*a)
            a = ab_of_each_line[i+1][0]
            b = ab_of_each_line[i+1][1]
            x_r = (x0-a*b+a*y0)/(1+a*a)
            x_list = np.arange(x_l, x_r, 1)
            if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                y_list = y0+pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            else:
                y_list = y0-pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            radiuses.append(radius)
            for j in range(list_of_line_index[i][-1]+1,list_of_line_index[i+1][0]):
                if x_new[j]<=x_l:
                    y_predicts.append(x_new[j]*ab_of_each_line[i][0]+ab_of_each_line[i][1])
                elif x_new[j] > x_l and x_new[j] < x_r:
                    if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                        y_temp = y0+pow(pow(radius, 2)-pow(x_new[j]-x0, 2), 0.5)
                    else:
                        y_temp = y0-pow(pow(radius, 2)-pow(x_new[j]-x0, 2), 0.5)
                    y_predicts.append(y_temp)
                else:
                    y_predicts.append(x_new[j]*ab_of_each_line[i+1][0]+ab_of_each_line[i+1][1])
        else:
            radiuses.append(0)
            for j in range(list_of_line_index[i][-1]+1,list_of_line_index[i+1][0]):
                if x_new[j]<=x_intersect:
                    y_predicts.append(x_new[j]*ab_of_each_line[i][0]+ab_of_each_line[i][1])
                elif x_new[j]>x_intersect:
                    y_predicts.append(x_new[j]*ab_of_each_line[i+1][0]+ab_of_each_line[i+1][1])

        delta_alpha.append(ab_of_each_line[i+1][0]-ab_of_each_line[i][0])
        bpd.append([x_intersect,y_intersect])
        alpha.append(ab_of_each_line[i][0])
    
    alpha.append(ab_of_each_line[-1][0])
    for j in range(len(list_of_line_index[-1])):
        y_predicts.append(predicted_ys[-1][j])
    y_predicts.append(0)
    return bpd,radiuses,alpha,delta_alpha,y_predicts


def output_fixed():
    bpd = []
    radiuses = []
    delta_alpha = []
    alpha = []
    y_predicts = []
    y_predicts.append(0)
    for i in range(len(ab_of_each_line)-1):
        b_plus_1 = ab_of_each_line[i+1][1]
        b = ab_of_each_line[i][1]
        a_plus_1 = ab_of_each_line[i+1][0]
        a = ab_of_each_line[i][0]
        if a >6e-3:
            a = 6e-3
            xs = np.array(x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1])
            ys = np.array(y_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1])
            b = mean(ys-xs*a)
        if a < -6e-3:
            a = -6e-3
            xs = np.array(x_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1])
            ys = np.array(y_new[list_of_line_index1[i][0]:list_of_line_index1[i][-1]+1])
            b = mean(ys-xs*a)

        x_intersect = (b_plus_1-b)/(a-a_plus_1)
        y_intersect = (a*b_plus_1-a_plus_1*b)/(a-a_plus_1)
        # for j in range(len(list_of_line_index[i])):
        #     y_predicts.append(predicted_ys[i][j])
        if abs(a_plus_1-a)>3e-3:
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
            radius = -(A*(n*x_intersect-sum(x_new[l_index:r_index])) +
                       B*(n*y_intersect-sum(y_new[l_index:r_index]))) /\
                (A*A+B*B-1)/n
            radius *= 1.4
            if radius > 0 and radius < 10000:
                radius = 10000
            if radius > 40000:
                radius = 40000
            x0 = x_intersect+A*radius
            y0 = y_intersect+B*radius
            a = ab_of_each_line[i][0]
            b = ab_of_each_line[i][1]
            x_l = (x0-a*b+a*y0)/(1+a*a)
            a = ab_of_each_line[i+1][0]
            b = ab_of_each_line[i+1][1]
            x_r = (x0-a*b+a*y0)/(1+a*a)
            x_list = np.arange(x_l, x_r, 1)
            if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                y_list = y0+pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            else:
                y_list = y0-pow(pow(radius, 2)-pow(x_list-x0, 2), 0.5)
            radiuses.append(radius)
            for j in range(list_of_line_index[i][-1]+1,list_of_line_index[i+1][0]):
                if x_new[j]<=x_l:
                    y_predicts.append(x_new[j]*ab_of_each_line[i][0]+ab_of_each_line[i][1])
                elif x_new[j] > x_l and x_new[j] < x_r:
                    if ab_of_each_line[i][0] > ab_of_each_line[i+1][0]:
                        y_temp = y0+pow(pow(radius, 2)-pow(x_new[j]-x0, 2), 0.5)
                    else:
                        y_temp = y0-pow(pow(radius, 2)-pow(x_new[j]-x0, 2), 0.5)
                    y_predicts.append(y_temp)
                else:
                    y_predicts.append(x_new[j]*ab_of_each_line[i+1][0]+ab_of_each_line[i+1][1])
        else:
            radiuses.append(0)
            for j in range(list_of_line_index[i][-1]+1,list_of_line_index[i+1][0]):
                if x_new[j]<=x_intersect:
                    y_predicts.append(x_new[j]*ab_of_each_line[i][0]+ab_of_each_line[i][1])
                elif x_new[j]>x_intersect:
                    y_predicts.append(x_new[j]*ab_of_each_line[i+1][0]+ab_of_each_line[i+1][1])

        delta_alpha.append(ab_of_each_line[i+1][0]-ab_of_each_line[i][0])
        bpd.append([x_intersect,y_intersect])
        alpha.append(ab_of_each_line[i][0])
    
    alpha.append(ab_of_each_line[-1][0])
    for j in range(len(list_of_line_index[-1])):
        y_predicts.append(predicted_ys[-1][j])
    y_predicts.append(0)
    return bpd,radiuses,alpha,delta_alpha,y_predicts


if __name__ == '__main__':
    [x_data, y_data] = get_data('input.xls')
    for i in range(7):
        x_data.insert(0,x_data[0]-50)
        y_data.insert(0,y_data[0]+0.01*i)
    x_data.append(x_data[-1]+50)
    y_data.append(y_data[-1])
    x_new, y_new = x_data, y_data
    x_label = np.zeros(len(x_new), dtype='int')
    x_label[0] = 1
    x_label[-1] = 1
    x_plot_break=[]
    y_plot_break=[]
    while True:
        list_of_line_index = get_lines_indexes(x_label)
        ab_of_each_line, delatas_of_lines, predicted_ys = get_ab_of_lines(
            list_of_line_index)
        if ab_of_each_line != None:
            if stable_fit() == 1:
                if check_equal_a() == 1:
                    break
    # plt.clf()
    # plot_figure()
    

    points_owners()
    while True:
        list_of_line_index = get_lines_indexes(x_label)
        ab_of_each_line, delatas_of_lines, predicted_ys = get_ab_of_lines(
            list_of_line_index, 3)
        if ab_of_each_line != None:
            if check_equal_a(para=3, thre=2e-3):
                if points_owners() == 1:
                    if merge_short() == 1:
                        break
    # plot_figure()

    while True:
        list_of_line_index = get_lines_indexes(x_label)
        ab_of_each_line, delatas_of_lines, predicted_ys = get_ab_of_lines(
            list_of_line_index, 5)
        if ab_of_each_line != None:
            if check_equal_a(para=5, thre=5e-4):
                circle_fit_plot()
                plot_figure()
                if circle_fit() == 1:
                    if points_owners() == 1:
                        if merge_short() == 1:
                            break
    # circle_fit_plot()
    # plot_figure()

    merge_short(final=True)
    while True:
        list_of_line_index = get_lines_indexes(x_label)
        ab_of_each_line, delatas_of_lines, predicted_ys = get_ab_of_lines(
            list_of_line_index, 5,final=True)
        if ab_of_each_line != None:
            if check_equal_a(para=5, thre=5e-4,final=True):
                if circle_fit() == 1:
                    if points_owners() == 1:
                            break
    circle_fit_plot()
    plot_short()
    plot_figure()
    circle_fit_plot()
    
    bpd,radiuses,alpha,delta_alpha,y_predicts = output()
    sum = 0
    for i in range(8,len(x_new)-1):
        sum+=(y_new[i]-y_predicts[i])*(y_new[i]-y_predicts[i])
    print(sum)
    bpd_x = []
    bpd_y = []
    for i in range(len(bpd)):
        bpd_x.append(bpd[i][0])
        bpd_y.append(bpd[i][1])
        p_bpd, = plt.plot(bpd[i][0],bpd[i][1],'ob',markersize=8)
    plt.plot(bpd_x,bpd_y,'-.b')
    p_predicts, = plt.plot(x_new[1:-1],y_predicts[1:-1],'-r')
    p_exact, = plt.plot(x_new, y_new, '^g', markersize = 5)
    plt.legend([p_bpd, p_predicts, p_exact], ["Grade Change Points", "Optimized Line","Data Points"], loc='upper right')  # 画出所有数据点
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.text(100000,70,r'$\sum_{i=1}^n (\hat y_i-y_i)^2  = %.2f m^2$'% (sum),  fontsize=15)
    plt.show()

    

    workbook = xw.Workbook('bpd.xlsx')  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['变坡点x', '变坡点y', '半径','坡长','坡度','坡度差']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头

    i = 3  # 从第二行开始写入数据
    for j in range(len(bpd)):
        insertData = [round(bpd[j][0],2),round(bpd[j][1],2), int(radiuses[j])]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 2

    i = 4  # 从第二行开始写入数据
    for j in range(len(bpd)-1):
        insertData = [round(bpd[j+1][0]-bpd[j][0],2)]
        row = 'D' + str(i)
        worksheet1.write_row(row, insertData)
        i += 2
    
    i = 2  # 从第二行开始写入数据
    for j in range(len(alpha)):
        insertData = [int(alpha[j]*100000)/100]
        row = 'E' + str(i)
        worksheet1.write_row(row, insertData)
        i += 2
    
    i = 3  # 从第二行开始写入数据
    for j in range(len(delta_alpha)):
        insertData = [int(delta_alpha[j]*100000)/100]
        row = 'F' + str(i)
        worksheet1.write_row(row, insertData)
        i += 2
    workbook.close()  # 关闭表

    workbook = xw.Workbook('line_points.xlsx')  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['x坐标', 'y坐标-实际', 'y坐标-重构']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头

    i = 2  # 从第二行开始写入数据
    for j in range(len(x_new)):
        insertData = [round(x_new[j],2),round(y_new[j],2), round(y_predicts[j],2)]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1

    workbook.close()  # 关闭表