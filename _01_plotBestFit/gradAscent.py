#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 11:35
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : gradAscent.py
# @Software: PyCharm

from numpy import *

# sigmoid函数，压缩值到（0，1）
def sigmoid(inX):
	return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn, classLabels):
	# 将数组转换成矩阵，可进行相关的矩阵运算
	dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
	# 对类标签包数组转为矩阵，然后转置
	labelMat = mat(classLabels).transpose() #convert to NumPy matrix
	# 读取数据包的行和列
	m,n = shape(dataMatrix)
	# 向目标移动的步长
	alpha = 0.001
	# 迭代次数
	maxCycles = 500
	weights = ones((n,1))
	for k in range(maxCycles):              #heavy on matrix operations
		h = sigmoid(dataMatrix*weights)     #matrix mult
		error = (labelMat - h)              #vector subtraction
		weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
	return weights.getA()