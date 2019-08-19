#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/16 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/16  下午3:00 modify by jade
from jade import *
from datasetopeation.jadeClassifyTFRecords import CreateClassifyTFRecorder
def createDataSet():
    XMLTOPROTXT("/home/jade/Data/sdfgoods/sdfgoods10.xlsx",protxt_name="sdfgoods")
    CreateClassifyTFRecorder("/home/jade/Data/sdfgoods/sdfgoods10","sdfgoods","/home/jade/Data/sdfgoods/sdfgoods.prototxt")


if __name__ == '__main__':
    createDataSet()
