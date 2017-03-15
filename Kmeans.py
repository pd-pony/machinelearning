from numpy import *
import time
import matplotlib.pyplot as plt


# 计算欧式距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))
    # 0ρ = sqrt( (x1-x2)^2+(y1-y2)^2 )　|x| = √( x2 + y2 )
    # power 对列表计算2次方  求和后开方


# 初始化质心随机样本
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape  # 获取数据集合的行列总数
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        # uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]  # 行数

    clusterAssment = mat(zeros((numSamples, 2)))  #

    clusterChanged = True  # 停止循环标志位

    ## step 1: init 初始化k个质点
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each 行
        for i in range(numSamples):
            minDist = 100000.0  # 设定一个极大值
            minIndex = 0
            ## for each centroid
            ## step 2: 寻找最接近的质心
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                # 将centroids（k个初始化质心）的j行和dataset（数据全集）的i行 算欧式距离，返回数值型距离
                if distance < minDist:
                    # 找距离最近的质点，记录下来。
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster # 跟新这个簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # clusterAssment 是一个n行2列的矩阵  Assment 评估
                clusterAssment[i, :] = minIndex, minDist ** 2  # 赋值为 新的质点标号

        ## step 4: update centroids
        for j in range(k):
            # 属于j这个质点的所有数值的平均值算出成为新的质点
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print  ('Congratulations, cluster complete!')
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("Sorry! Your k is too large! please contact Zouxy")
        return 1

    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  # 设定颜色

    # draw the centroids
    # 画出质点，用特殊图型
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__ == '__main__':
    ## step 1: 加载数据
    print ("step 1: load data...")
    dataSet = []
    fileIn = open('./data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(' ')
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        # 再按照数据之间的空格作为分隔符。
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        # 返回加入到dataset中的每组数据为一个列表。形成二维数组
    ## step 2: 开始聚类...
    print ("step 2: clustering...")
    dataSet = mat(dataSet)
    # mat 函数，将数组转化为矩阵

    k = 5
    centroids, clusterAssment = kmeans(dataSet, k)

    ## step 3: show the result
    print ("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)