#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/20 21:51
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : gradAscent.py
# @Software: PyCharm

from numpy import *

# 对文本进行处理，读出数据包和标签包
def loadDataSet():
	# 初始化两个包，一个存放数据的包，一个存放类别的包
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	# 循环处理文件中的每一行
	for line in fr.readlines():
		# strip（）是默认去掉首尾空格，split（）默认以空格，标点等为对象劈开
		lineArr = line.strip().split()
		# 添加文本第一列和第二列，以向量的形式，最后构成数组
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		# 把文本第三列添加到标签包
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

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
	# 用来存放三个权重值
	weights = ones((n,1))
	# W := W + α * X(tanspose) * (Y - g(X*θ))
	for k in range(maxCycles):              #heavy on matrix operations
		# 求sigmoid函数， g(X*θ)
		h = sigmoid(dataMatrix*weights)     #matrix mult
		# 求出 Y - g(X*θ)
		error = (labelMat - h)              #vector subtraction
		# W := W + α * X(tanspose) * (Y - g(X * θ))
		weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
	return weights

if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	print (gradAscent(dataArr, labelMat))