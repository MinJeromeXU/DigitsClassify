'''
kNN: k Nearest Neighbors
# 数据集中每个数字的写法不一样有X中写法，每一种写法对应一个文件将X种写法归为一类x为样本数
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)          
@Output: the most popular class label
'''
from numpy import *
import operator
from os import listdir
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 将原数据集的每一行减去测试集（测试集只有1*M，需要构成一个n*M矩阵）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 将距离从小到大排序后返回对应于原数据集中的索引位置（哪一行）
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):  # 取出距离最小的前k个距离对应的每个样本集对应的所属类，
        voteIlabel = labels[sortedDistIndicies[i]]
        # 统计前k个样本集中属于同一个类的个数{类别名：个数}
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 个数最大的则为该测试所属的类
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # sorted返回的是一个列表[(类别名：个数)]
    # 只返回类别名
    return sortedClassCount[0][0]
# 将一个32*32的图像矩阵转换为一个1*102的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    # load the training set 解析路径下的所有文件列表
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 文件名格式：0_0.txt  第一个0代表类标签
        #从文件名中获取类标签，从文件内容中获取一个（1*1024）的向量，
        # m个文件就有构成m*1024矩阵
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #hwLabels就是一个m*1的向量作为类标识
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % ((errorCount) / float(mTest)*100))
handwritingClassTest()
#额外说明该分类器是样本集特征都是0或者1 比如两个样本集：x1=[0,1,0,1,1,0],x2=[1,0,0,0,1,1] ，
# 做距离计算不会相差太大 x1-x2=[0-1,1-0,...,0-1]六个特征数值范围都是0,1其中一个，可以直接计两者距离
#但是当特征中数值范围差距比较大时需要做归一化处理
#比如y1=[a1,b1],y2=[a2,b2],a1,a2数字范围为[2,10],
# 但是b1,b2范围是[1,1000]相差较大是需要做归一化处理便于计算，
# 这个不只是数字分类器可以是其他的分类器
# newValue=(oldValue-min)/(max-min)

# 该KNN分类器的缺点：1、样本不平衡，比如类别0的样本数是188，而类别2的样本数是86
#                    2、计算量比较大（改用kd树可以减少计算量），该课题还好。需求的内存也比较大。
# 优点：1、时间复杂度为O(n),简单理解只有一层for循环，2、可用于非线性分类

# K的取值问题：在分类时较大的K值能够减小噪声的影响。但会使类别之间的界限变得模糊。
# 一个较好的K值可通过各种启发式技术来获取，
# 比如，交叉验证。另外噪声和非相关性特征向量的存在会使K近邻算法的准确性减小
# 本例用的k=3  增大或者减小结果错误率都比3高
# 可以用一个for循环对k在一个范围值内画出错误率的变化曲线

