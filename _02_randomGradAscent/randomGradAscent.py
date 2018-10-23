#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 17:16
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : randomGradAscent.py
# @Software: PyCharm

from numpy import *
from sigmoid import sigmoid
from loadDataSet import loadDataSet
from plotBestFit import plotBestFit


# 函数说明：这里是改良版的优化算法
def stocGradAscent0(dataMatrix, classLabels):
	dataMatrix = array(dataMatrix)
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)   #initialize to all ones
	for i in range(m):
		# 一行一行数据处理，这里处理的是数值
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	dataMatrix = array(dataMatrix)
	m,n = shape(dataMatrix)
	#print ('m= ', m)
	weights = ones(n)   #initialize to all ones
	# 迭代150次后返回weights
	for j in range(numIter):
		# 从 0 ~ m 组成一个列表
		dataIndex = list(range(m))
		#print ('dataIndex= ',dataIndex)
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
			# 随机选出一个下标用于提取出一行数据或者类标签
			randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			# print ('weights= ',weights)
			del(dataIndex[randIndex])
	return weights

if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	weights = stocGradAscent1(dataArr, labelMat)
	# print ('weights= ', weights)
	plotBestFit(weights)