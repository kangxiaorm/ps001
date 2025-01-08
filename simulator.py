# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:21:41 2020

@author: Administrator
"""
import pandas as pd 
from .engine import SimulatorEngine
from .manager import VehicleManager
from multiprocessing import Pool
import networkx as nx

class RoadsNetworkBasedSimulator(object):
    
    
    def __init__(self,requests=None,engine=None,cache_dir="..//ridesharing_results",**kwargs):
        
        self.strategys={}

        if isinstance(engine,SimulatorEngine):
            self.engine=engine
            self.engine.set_cache_dir(cache_dir)
        else:
            vm=VehicleManager(kwargs.get("roads"),kwargs.get("vehicles",{}),kwargs.get("regions",{}),
                              kwargs.get("name",""),cache_size=kwargs.get("cache_size",10000))
            self.engine=SimulatorEngine(vm,name=kwargs.get("name",""),cache_dir=cache_dir)
        
        if isinstance(requests,pd.DataFrame):
            self.set_requests(requests)
        else:
            self.requests=None
            
        
    def set_requests(self,requests:pd.DataFrame):
        if isinstance(requests,pd.DataFrame):
            pass
        
            
            
        else:
            raise ValueError("Invalid requests data format")
        
        reqs=requests.set_index(requests["timestamp"])
        reqs=reqs.sort_index()
        self.requests=reqs
        self.requests["request_date"]=reqs.timestamp.astype(str).apply(lambda x:x[:10])
        self.requests_dates=self.requests["request_date"].unique()
        
    def add_simulate_strategy(self,strategy):
        name=strategy.__class__.__name__
        self.strategys[name]=strategy

    def simulating(self,name:str,strategys:list,dates:list,cache_dir="..//results_1114_1120",print_unit=0.1):
        
        self.engine.set_cache_dir(cache_dir)
        ds=set(dates)
        rds=set(self.requests_dates)    # 21 days
        if ds.issubset(rds) is False:
            raise ValueError("dates must be subset of dates requests data contains")
        self.engine.vehicle_manager.uniformed_initialize_vehicles()
        for s in strategys:
            s.clear()
            for date in dates:
                print("processing strategy:{0} date:{1}".format(s.__class__.__name__,date))
                reqs=self.requests[self.requests["request_date"]==date]
                reqs=reqs.sort_index()
                
                self.engine.name="{0}[{1}]".format(name,date)
                self.engine.reinitialize()
                self.engine.set_requests(reqs)
                self.engine.set_strategy(s)
                self.engine.set_date(date)
                # try:
                #     self.engine.run(report=print_unit)
                # except Exception as e:
                #     print(e)

                self.engine.run(report=print_unit)

    @staticmethod
    def run_strategy(strategy,date,roads_path,ods_path,vehicles={},name='',
                     cache_size=10000,cache_dir="..//ridesharing_results",print_unit=0.1):
        
        roads=nx.read_gpickle(roads_path)
        ods=pd.read_pickle(ods_path)
        ods1=ods.copy()
        ods1["timestamp"]=ods1["pick_up_time"]
        vm=VehicleManager(roads,vehicles=vehicles,name=name,cache_size=cache_size)
        engine=SimulatorEngine(vm,strategy,name="{0}_{1}".format(date,name),cache_dir=cache_dir)
        ods2=ods1[ods1["o_date"]==date]
        engine.set_requests(ods2)
        engine.run(report=print_unit)
        
        
    def multiprocessing_simulating(self,strategys:list,dates:list,roads_path,ods_path,
                                   pool_size=4,print_unit=0.1,vehicle_num=500,capacity=4,
                                   cache_size=10000,cache_dir="..//ridesharing_results"):
        pool=Pool(pool_size)
        vhs=self.engine.vehicle_manager.uniformed_initialize_vehicles(vehicle_num,capacity)
        for s in strategys:
            for date in dates:
                print("tast->{0}-{1} start simulating".format(s.__class__.__name__,date))
                pool.apply_async(RoadsNetworkBasedSimulator.run_strategy,(s,date,roads_path,ods_path,vhs,
                                                    self.engine.name,cache_size,cache_dir,print_unit))
        
        pool.close()
        pool.join()
       
        print("Finish multiprocessing simulating")