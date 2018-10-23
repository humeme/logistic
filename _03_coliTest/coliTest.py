#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 18:43
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : coliTest.py
# @Software: PyCharm

from numpy import *
from stocGradAscent1 import *

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        # 这里处理得到的也是一个向量
        currLine = line.strip().split('\t')
        lineArr =[]
        #print ('currLine=', currLine[1])
        # 循环把文本中的数据读入 lineArr 中
        for i in range(21):
            # 这里读进去的只是一行行的向量，并没有组成数组
            lineArr.append(float(currLine[i]))
        #print ('lineArr=', lineArr)
        # 在这里组成了一个列表
        trainingSet.append(lineArr)
       # print('type(trainingSet)=',type(trainingSet))
       # print ('trainingSet=', trainingSet)
		# 将类标签添加到一个包中
        trainingLabels.append(float(currLine[21]))
    # 随机梯度上升得到权重值
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)
    # 计算错误率以及错误的个数
    errorCount = 0; numTestVec = 0.0
    # 处理测试数据的每一行，读到列表中
    for line in frTest.readlines():
        # 计算有多少个测试数据
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 对测试集进行分类操作并计算分类错的个数
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    # 错误率
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

# 计算 10次 平均错误率
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()