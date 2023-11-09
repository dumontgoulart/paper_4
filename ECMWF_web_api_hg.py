# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2020

@author: tbr910
"""
import sys
import os
from datetime import datetime
import numpy as np


import seaborn as sns
import pandas as pd


from ecmwfapi import ECMWFService
from ecmwfapi import ECMWFDataServer

###################################################################### MARS ARCHIVE ##########################################################################
os.chdir(r'/gpfs/work3/0/einf4318/ecmwf/data') ## Dir has to be tbr910, for some weird reason
################################# SEAS forecast ############################################### 

server = ECMWFService("mars")

lon_min, lat_min, lon_max, lat_max = -88.549805,20.303418,-58.447266,31.391158
start_date = '2017-08-31'
step_ini, step_end, step_ratio = 0, 2, 1
grid = 0.25

# this is working
server.execute(
    {
    "class": "od",
    "date": f"{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}",
    "expver": "1",
    "levtype": "sfc",
    "param": "228.128",
    "number": f"1/to/2",
    "step": f"{step_ini}/to/{step_end}/by/{step_ratio}",
    "stream": "enfo",
    "time": "00",
    "type": "pf",
    "grid": f"{grid}/{grid}",
    "area": f"{int(lat_min)}/{int(lon_min)}/{int(lat_max)}/{int(lon_max)}",
    "format": "netcdf",
    },
    "target_test_cl.nc")


# Now with ensemble forecast

lon_min, lat_min, lon_max, lat_max = -88.549805,20.303418,-58.447266,31.391158
start_date = '2017-09-06'
step_ini, step_end, step_ratio = 0, 24, 1
number = 1
grid = 0.25
vars = 'tp'

dict_vars = {'tp': '228.128', 'msl': '151.128', 'u10': '165.128', 'v10': '166.128', 't2m': '167.128'}

server.execute(
    {
    "class": "od",
    "date": f"{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}",
    "expver": "1",
    "levtype": "sfc",
    "number": f"1/to/{number}",
    "param": f"{dict_vars[vars]}",
    "step": f"{step_ini}/to/{step_end}/by/{step_ratio}",
    "stream": "enfo",
    "time": "00",
    "type": "pf",
    "grid": f"{grid}/{grid}",
    "area": f"{int(lat_min)}/{int(lon_min)}/{int(lat_max)}/{int(lon_max)}",
    "format": "netcdf",
    },
    f"ecmwf_eps_pf_{vars}_n{number}_s{step_end}_{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}.nc")













# # hindcast request 
# retrieve,
# class=od,
# date=2022-12-29,
# expver=1,
# hdate=2021-12-29,
# levtype=sfc,
# number=1/2/3/4/5/6/7/8/9/10,
# param=228.128,
# step=0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360/366/372/378/384/390/396/402/408/414/420/426/432/438/444/450/456/462/468/474/480/486/492/498/504/510/516/522/528/534/540/546/552/558/564/570/576/582/588/594/600/606/612/618/624/630/636/642/648/654/660/666/672/678/684/690/696/702/708/714/720/726/732/738/744/750/756/762/768/774/780/786/792/798/804/810/816/822/828/834/840/846/852/858/864/870/876/882/888/894/900/906/912/918/924/930/936/942/948/954/960/966/972/978/984/990/996/1002/1008/1014/1020/1026/1032/1038/1044/1050/1056/1062/1068/1074/1080/1086/1092/1098/1104,
# stream=enfh,
# time=00:00:00,
# type=pf,
# target="output"




# def tigge_pf_request (date, target):              
#     server.retrieve({
        
#         "class": "ti",
#         "dataset": "tigge",
#         "date": date,
#         "expver": "prod",
#         "grid": "F640", 
#         "levtype": "sfc",
#         "number": "1/to/50",#/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50
#         "origin": "ecmf",
#         "param": "228228",
#         "step": "6/TO/36/BY/6", # 0 is time step zero, so zero accumulated precipitation. Accumulated precipitation (tp) is compared to initialization time (i.e. step 36 is accumulated tp from ini to step 36).
#         "time": '00/12',
#         "area": " 54.109311/2.064148/50.794060/7.788420",    # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
#         "type": "pf", # perturbed forecast fp= forecast probability , cf=control forecast,
#         "target": target,
#     })


# retrieve,
# class=od,
# date=['2021-07-09', '2021-07-10', '2021-07-11', '2021-07-12', '2021-07-13', '2021-07-14', '2021,07-15']
# expver=1,
# levtype=sfc,
# param=228.128,
# step=
# stream=oper,
# time=00:00:00, 12:00:00
# type=fc,
# target="output"

# Bounding box in EPSG = 4326: 
# North: 50.95
# East: 6.15
# South: 50.60
# West: 5.65

# #####################################################################################################################################################################
# ###################################################################### TIGGE ARCHIVE ##########################################################################
# server = ECMWFDataServer()

# def retrieve_tigge_data():
    
# # old request setting
#     # date_list=pd.date_range('2010-01-01','2010-01-03', 
#     #           freq='D').strftime("%Y-%m-%d").tolist()#MS
#     # #date_range= []
#     # #forbidden_dates= ['2015-09-01','2014-03-01'] ## damaged tapes
#     # # for i in range(len(date_list)-1): 
#     # #     if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1] :
#     # #         month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
#     # #         date_range.append(month_range)

#     # forbidden_dates= ['2015-09-01','2014-03-01'] ## damaged tapes
#     # times = ['00', '12']
#     # for date in date_list:
#     #   if  date!=forbidden_dates[0] and date!=forbidden_dates[1]:  
#     #     for time in times:
#     #         target = 'C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF/tigge_files/12_h_runs/ecmwf_tigge_%s_%s.grb' % (date[:10], time)
#     #         print ('REQUEST ACTIVE FOR: %s' % (target))
#     #         tigge_pf_request(date,time, target)

# # new request setting
#     date_list=pd.date_range('2020-06-01','2020-07-01', 
#               freq='MS').strftime("%Y-%m-%d").tolist()#MS
#     date_range= []
#     for i in range(len(date_list)-1): 
#         # if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1]:
#             month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
#             date_range.append(month_range)
  
#     for date in date_range:
#         target = 'C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF/tigge_files/ecmwf_tigge_%s.grib' % (date[:10])
#         print ('REQUEST ACTIVE FOR: %s' % (target))
#         tigge_pf_request(date, target)               
        
# def tigge_pf_request (date, target):              
#     server.retrieve({
        
#         "class": "ti",
#         "dataset": "tigge",
#         "date": date,
#         "expver": "prod",
#         "grid": "F640", 
#         "levtype": "sfc",
#         "number": "1/to/50",#/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50
#         "origin": "ecmf",
#         "param": "228228",
#         "step": "6/TO/36/BY/6", # 0 is time step zero, so zero accumulated precipitation. Accumulated precipitation (tp) is compared to initialization time (i.e. step 36 is accumulated tp from ini to step 36).
#         "time": '00/12',
#         "area": " 54.109311/2.064148/50.794060/7.788420",    # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
#         "type": "pf", # perturbed forecast fp= forecast probability , cf=control forecast,
#         "target": target,
#     })

# retrieve_tigge_data()