#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 11:28
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : plotBestFit.py
# @Software: PyCharm

from numpy import *
from loadDataSet import loadDataSet
from gradAscent import gradAscent
import matplotlib.pyplot as plt

def plotBestFit(weights):
	# 以列表形式读取出来
	dataMat, labelMat = loadDataSet()

	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	#print (n, type(shape))
	xcord1 = [];  ycord1 = []
	xcord2 = [];  ycord2 = []
	for i in range(n):
		# 1 为正样本，0 为负样本
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	x2 = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, x2)
	plt.xlabel('X1'); plt.ylabel('X2'); plt.title('data')
	plt.show()

if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	weights = gradAscent(dataArr, labelMat)
	plotBestFit(weights)