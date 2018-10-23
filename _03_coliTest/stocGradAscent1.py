#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 18:45
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : stocGradAscent1.py
# @Software: PyCharm

from numpy import *

def sigmoid (inX):
	return 1.0 / (1 + exp (-inX))

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