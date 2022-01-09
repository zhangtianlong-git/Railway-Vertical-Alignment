import numpy as np
import matplotlib.pyplot as plt


class Node:
    """节点定义"""

    def __init__(self, x, y, r=0):
        self.x = x
        self.y = y
        self.r = r
        self.prev = None
        self.next = None

    def get_BPD_alpha(self, return_degree=False):
        """变坡点的角度值，默认返回弧度制，return_degree设为True返回角度"""
        alpha = np.arccos(((self.x-self.prev.x)*(self.next.x-self.x)+(self.y-self.prev.y)*(self.next.y-self.y)) /
                          np.linalg.norm([self.x-self.prev.x, self.y-self.prev.y]) /
                          np.linalg.norm([self.next.x-self.x, self.next.y-self.y]))
        if return_degree == False:
            return alpha
        else:
            return np.degrees(alpha)

    def getX_of_both_side(self):
        """获取变坡点左右圆曲线和直线段的交点x坐标，用于初步划分节点所属区段"""
        # 变坡点到左切点的距离
        d1 = self.r*abs(np.tan(self.get_BPD_alpha()/2))
        # 变坡点与上个变坡点的距离
        d2 = np.linalg.norm([self.x-self.prev.x, self.y-self.prev.y])
        # 运用相似三角形
        x_temp_1 = self.x-(self.x-self.prev.x)*d1/d2
        # 同上
        d1 = self.r*abs(np.tan(self.get_BPD_alpha()/2))
        d2 = np.linalg.norm([self.x-self.next.x, self.y-self.next.y])
        x_temp_2 = self.x+(self.next.x-self.x)*d1/d2
        return [x_temp_1, x_temp_2]

    def getab_of_both_sides(self):
        """获取变坡点左右直线方程的a和b，y=ax+b"""
        left_a = (self.y-self.prev.y)/(self.x-self.prev.x)
        left_b = self.y-left_a*self.x
        right_a = (self.y-self.next.y)/(self.x-self.next.x)
        right_b = self.y-right_a*self.x
        return left_a, left_b, right_a, right_b

    def getXY_of_circle_center(self):
        """获取变坡点所对应的曲线段圆心坐标"""
        left_a, left_b, right_a, right_b = self.getab_of_both_sides()
        if left_a*right_a >= 0:
            raise ValueError("变坡点左右斜率不正确！")
        # 圆心到左右直线距离为r，联立方程组，利用矩阵运算求出圆心坐标
        a = np.array([(left_a, -1), (right_a, -1)])
        if left_a > 0:
            b = np.array([[self.r*np.linalg.norm([left_a, 1])-left_b],
                         [self.r*np.linalg.norm([right_a, 1])-right_b]])
        else:
            b = np.array([[-self.r*np.linalg.norm([left_a, 1])-left_b],
                         [-self.r*np.linalg.norm([right_a, 1])-right_b]])
        XY = np.dot(np.linalg.inv(a), b)
        return [XY[0][0], XY[1][0]]


class NodeList:
    """节点链表"""

    def __init__(self):
        self._head = None

    def length(self):
        """链表长度"""
        cur = self._head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def is_empty(self):
        """判断链表是否为空"""
        return self._head == None

    def travel(self):
        """遍历链表"""
        cur = self._head
        while cur != None:
            print('...')
            cur = cur.next

    def travel_and_getY(self, x_list):
        """遍历链表，通过x值判断点所属的区段，计算对应的y"""
        y_list = []
        for each in x_list:
            cur = self._head.next
            while True:
                # 从第一个变坡点开始，查询x所属区段是否为当前变坡点左侧的直线段，用左侧的直线方程求y
                if (each >= cur.prev.x) and (each < cur.getX_of_both_side()[0]):
                    left_a, left_b, right_a, right_b = cur.getab_of_both_sides()
                    y_list.append(left_a*each+left_b)
                    break
                # 查询x所属区段是否为当前变坡点所对应的曲线段，用曲线方程计算y
                elif (each >= cur.getX_of_both_side()[0]) and (each < cur.getX_of_both_side()[1]):
                    left_a, left_b, right_a, right_b = cur.getab_of_both_sides()
                    XY = cur.getXY_of_circle_center()
                    d_temp = pow(pow(cur.r, 2)-pow(each-XY[0], 2), 0.5)
                    if left_a > 0:
                        y_list.append(XY[1]+d_temp)
                    else:
                        y_list.append(XY[1]-d_temp)
                    break
                # 不满足以上，跳到下个节点开始筛查
                else:
                    cur = cur.next
                    # 如果是最后一个变坡点，则用其右侧的直线方程计算y
                    if cur.next == None:
                        left_a, left_b, right_a, right_b = cur.prev.getab_of_both_sides()
                        y_list.append(right_a*each+right_b)
                        break
        return y_list

    def append(self, x, y, r=0):
        """尾部插入元素"""
        node = Node(x, y, r)
        if self.is_empty():
            # 如果是空链表，将_head指向node
            self._head = node
        else:
            # 移动到链表尾部
            cur = self._head
            while cur.next != None:
                cur = cur.next
            # 将尾节点cur的next指向node
            cur.next = node
            # 将node的prev指向cur
            node.prev = cur

    def get_BPD_node(self, node_id):
        """获取变坡点，id从1开始"""
        if self.length() > 2 and node_id < self.length()-1:
            cur = self._head
            for i in range(node_id):
                cur = cur.next
        else:
            raise IndexError("变坡点的id索引出现错误")
        return cur


if __name__ == "__main__":
    li = NodeList()
    # 输入起点、终点和变坡点，变坡点输入参数为：x，y，r，其中r为圆曲线半径
    li.append(0, 0)
    li.append(500, 100, 600)
    li.append(1200, -50, 600)
    li.append(1500, 0, 600)
    li.append(2200, -200)
    # 以5m为间距,获得轨道竖曲线y坐标
    x_list = np.arange(0, 2200, 5)
    y_list = li.travel_and_getY(x_list)
    # 以40m为间距生成测试样本，考虑随机性，在y方向上添加均值为0，标准差为10正态随机值
    x_out = np.arange(0, 2200, 40)
    y_out_temp = li.travel_and_getY(x_out)
    y_out = y_out_temp + np.random.normal(0, 0.02, len(x_out))
    # 作简图
    plt.plot(x_list, y_list)
    plt.plot(x_out, y_out, 'ro')
    plt.axis('equal')
    plt.ylim(-210.110)
    plt.xlim(0.2200)
    plt.show()
    # 结果输出到txt
    file = open('output.txt', 'w')
    file.write(u'x坐标,' + u'y坐标,' + u'y坐标+随机误差,'+'\n')
    for i in range(len(x_out)):
        file.write("%.2f" % (x_out[i]) + ',' + "%.4f" %
                   (y_out_temp[i]) + ',' + "%.4f" % (y_out[i]) + '\n')
    file.close()
