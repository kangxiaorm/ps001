# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:22:09 2020

@author: Administrator
"""
import sys
import os
sys.path.append("D:/cjx_code/fleet_rs_v_2.0")
os.chdir('D:/cjx_code/fleet_rs_v_2.0')

from ridesharing.engine import SimulatorEngine
from ridesharing.manager import VehicleManager
from ridesharing.strategy import NearestSharing,APART,PDTL,AM,MTShare,DeepPool
from ridesharing.simulator import RoadsNetworkBasedSimulator
from ridesharing.requests import *

import networkx as nx
import pandas as pd 
import osmnx as ox
import pickle

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


# name = "wh_ridesharing"
# path3="data/wuhan_roads.gpkl"
# path6='data/wuhan_ods.pkl'
# path_demand = "data/wuhan_demand_demsc.pkl"
# path_regions = "data/wuhan_regions.pkl"

name = "sf_ridesharing"
path3="data/SanFrancisco_roads.pkl"
path6='data/SanFrancisco_ods.pkl'
path_demand = "data/SanFrancisco_demand_demsc.pkl"
path_regions = "data/SanFrancisco_regions.pkl"


g=nx.read_gpickle(path3)
ods=pd.read_pickle(path6)
requests=ods.copy()
timestamp=requests["pick_up_time"]
requests["timestamp"]=timestamp

vm=VehicleManager(g,cache_size=20000,name=name) # 初始化车辆信息
vm.uniformed_initialize_vehicles()
engine=SimulatorEngine(vm,name=name)
simulator=RoadsNetworkBasedSimulator(engine=engine)
simulator.set_requests(requests)

demands = pd.read_pickle(path_demand)
with open(path_regions,'rb') as f:
    regions = pickle.load(f)

rds = list(simulator.requests_dates)

# sf
# train_dates = rds[0:17]
# test_dates = rds[17:]

# wh
# train_dates = rds[0:8]
# train_dates = rds[4:8]
test_dates = rds[8:9]

# ns = NearestSharing()
# apart = APART()
# pdtl = PDTL(regions, demands)
# am = AM(regions, demands, name)
# mtshare = MTShare()
deeppool=DeepPool()

simulator.simulating(name,[deeppool],test_dates,cache_dir="./ridesharing_results")
# simulator.simulating(name,[am],train_dates,cache_dir="./ridedispatching_results")

