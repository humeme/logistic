#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 17:17
# @Author  : Humeme
# @Site    : 湘潭大学
# @File    : sigmoid.py
# @Software: PyCharm

from numpy import *

def sigmoid (inX):
	return 1.0 / (1 + exp (-inX))