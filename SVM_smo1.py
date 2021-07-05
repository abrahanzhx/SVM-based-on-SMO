import numpy as np
import random
import pandas as pd


def getData1(fileName):

    train = pd.read_excel(fileName)
    dataArr = train.loc[:, '样本属性1':'样本属性10'].values
    classLabels = train['类标'].values

    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    return dataMat, labelMat


def getData2(ngFileName, psFileName):
    train_ng = pd.read_excel(ngFileName, header=None)
    train_ps = pd.read_excel(psFileName, header=None)

    ng_X = np.array(train_ng)
    ps_X = np.array(train_ps)

    ng_y = np.full((len(ng_X), 1), -1)
    ps_y = np.ones((len(ps_X), 1))

    data_X = np.concatenate((ng_X, ps_X), axis=0)
    data_y = np.concatenate((ng_y, ps_y), axis=0)
    data_np = np.concatenate((data_X, data_y), axis=1)
    np.random.shuffle(data_np)

    data_shu_x = data_np[:,0:118]

    data_shu_y = data_np[:,118]

    X_mat = np.mat(data_shu_x)
    y_mat = np.mat(data_shu_y).T

    return X_mat, y_mat

def getData2test(ngFileName, psFileName):
    test_ng = pd.read_excel(ngFileName, header=None)
    test_ps = pd.read_excel(psFileName, header=None)

    test_ng_X = np.array(test_ng)
    test_ps_X = np.array(test_ps)

    test_ng_y = np.full((len(test_ng_X), 1), -1)
    test_ps_y = np.ones((len(test_ps_X), 1))

    data_X = np.concatenate((test_ng_X, test_ps_X), axis=0)
    data_y = np.concatenate((test_ng_y, test_ps_y), axis=0)

    X_mat = np.mat(data_X)
    y_mat = np.mat(data_y)

    # return X_mat, y_mat
    test_ng_X_mat = np.mat(test_ng_X)
    test_ps_X_mat = np.mat(test_ps_X)
    test_ng_y_mat = np.mat(test_ng_y)
    test_ps_y_mat = np.mat(test_ps_y)

    return X_mat, y_mat, test_ng_X_mat, test_ps_X_mat, test_ng_y_mat, test_ps_y_mat


class optStruct:

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 数据标签
        self.C = C  # 松弛变量
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。


def calcEk(oS, k):

    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def updateEk(oS, k):

    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存


def selectJrand(i, m):

    j = i  # 选择一个不等于i的j
    while j == i:
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):

    maxK = -1
    maxDeltaE = 0
    Ej = 0  # 初始化
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i:
                continue  # 不计算i,浪费时间
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, oS.m)  # 随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  # 计算Ej
    return j, Ej  # j,Ej


def clipAlpha(aj, H, L):

    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):

    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T  # 线性核 内核分母
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):

    oS = optStruct(dataMatIn, classLabels, C, toler)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            iter += 1
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i + 1, alphaPairsChanged))
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            iter += 1
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i + 1, alphaPairsChanged))
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif alphaPairsChanged == 0:  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas


def calcWs(X, labelMat, alphas):

    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 计算准确率
def score(dataMatrix, labelMat, w):
    trueCount = 0
    for i in range(labelMat.shape[0]):
        fx = float(w.T * dataMatrix[i, :].T + b)
        if np.sign(fx) == labelMat[i]:
            trueCount += 1
    return float(trueCount) / labelMat.shape[0]


if __name__ == '__main__':
    # X_train, y_train = getData1('Data1.xlsx')
    # test_X, test_y = getData1('Data1.t.xlsx')

    X_train, y_train = getData2('Data2_ng.xlsx', 'Data2_ps.xlsx')
    test_X, test_y, test_ng_X_mat, test_ps_X_mat, test_ng_y_mat, test_ps_y_mat = getData2test('Data2.t_ng.xlsx', 'Data2.t_ps.xlsx')
    # test_X,test_y = getData2test('Data2.t_ng.xlsx', 'Data2.t_ps.xlsx')
    b, alphas = smoP(X_train, y_train, 1, 0.001, 40)
    print('b=', float(b))
    # print('alphas=',alphas)
    w = calcWs(X_train, y_train, alphas)
    print('w=', w)
    print("训练集分类准确率: %f" % score(X_train, y_train, w))
    print("测试集分类准确率: %f" % score(test_X, test_y, w))
    print("负类测试集分类准确率: %f" % score(test_ng_X_mat, test_ng_y_mat, w))
    print("正类测试集分类准确率: %f" % score(test_ps_X_mat, test_ps_y_mat, w))