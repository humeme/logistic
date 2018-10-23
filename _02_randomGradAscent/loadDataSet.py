#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 19:25
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : loadDataSet.py
# @Software: PyCharm

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