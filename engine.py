# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:27:26 2020

@author: Administrator
"""
import os
import networkx as nx
import osmnx as ox
import time

import geopandas as gpd
from shapely.geometry import Point,LineString,Polygon
import pandas as pd 
from rtree.index import Index as RIndex
from collections import deque,defaultdict
from ridesharing.datastructure import Vehicle,Passenger
from queue import Queue, Empty
import logging
from ridesharing.manager import VehicleManager,gps_distance
from ridesharing.statistic import passengers_analysis
import pickle


class SimulatorEngine(object):
    
    vehicles={}
    regions={}
    passengers={}
    
    def __init__(self,vehicle_manager:VehicleManager,
                 sharing_strategy=None,name='',
                 start_time='',end_time='',freq='s',cache_dir="..//result"):
        self.logger=logging.getLogger(name)
        self.name=name
        self.logger=logging.getLogger(self.__class__.__name__)
        self.vehicle_manager:VehicleManager=vehicle_manager
        
#        self.event_engine=EventEngine2(timer_on=False,name="RideSharingSimulator:{0}".format(name))
        self.strategy=sharing_strategy
        self.start_datetime=start_time
        self.end_datetime=end_time
        self.freq=freq
     
        self.set_cache_dir(cache_dir)
        self.is_request_init=False
        self.requests=None
        self.__moving_events=defaultdict(dict)
        self.__passenger_events=defaultdict(dict)
        self.shortest_path_cache={}
        self.vehicle_nodes_predict=defaultdict(dict)
        
    def get_moving_events(self):
        return self.__moving_events
    
    def get_passenger_events(self):
        return self.__passenger_events
    
    def reinitialize(self):
        self.__moving_events.clear()
        self.__passenger_events.clear()
        self.vehicle_nodes_predict.clear()
        vhs=self.vehicle_manager.reinitialize_vehicles()
        return vhs
    
    def set_cache_dir(self,dir_):
        self.cache_dir=os.path.abspath(dir_)
        
    def set_requests(self,requests:pd.DataFrame):
        req=requests.copy()
        if isinstance(requests,pd.DataFrame) is False:
            raise ValueError("requests data must be pandas.DataFrame type")
        norm_cols=['id',"o_x","o_y","d_x","d_y","timestamp"]
        for ncol in norm_cols:
            if ncol not in req.columns:
                raise ValueError("Unnormalized requests data! Not contain column:{0}".format(ncol))
        if req.index.size<1:
            raise ValueError("have no data in requests data")
        r0=req.iloc[0]
        ts=r0.timestamp
        if isinstance(ts,pd.Timestamp) is False:
            raise ValueError("The attribute timestamp must be type of pandas.Timestamp")
        

        cols=["source","target","source_dist","target_dist"]
        
        
        f=[col in req.columns for col in cols]
        self.is_request_init=False
        ids=req["id"]
        if ids.is_unique:
            pass
        else:
            ids=list(range(len(ids)))
            req["id"]=ids
            
        if all(f):
            acols=norm_cols+cols
            self.is_request_init=True
            
        else:
            acols=norm_cols
        self.requests=req[acols]
        ts=self.requests["timestamp"]
        self.requests=self.requests.set_index(ts)

    def get_request_date_range(self):
        unique_time=self.requests["timestamp"].value_counts()
        unique_time.sort_index(inplace=True)
        start=unique_time.index[0]
        end=unique_time.index[-1]

        start_time = self.date + " 00:00:00"
        end_time = self.date + " 23:59:59"
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)

        return start,end
    
    def __init_requests(self,requests:pd.DataFrame):
        print("requests data preprocessing")
        g=self.vehicle_manager.roads
        o_nearest_nodes=ox.get_nearest_nodes(g,self.requests["o_x"],self.requests["o_y"],method="balltree")
        d_nearest_nodes=ox.get_nearest_nodes(g,self.requests["d_x"],self.requests["d_y"],method="balltree")
        self.requests["source"]=o_nearest_nodes
        self.requests["target"]=d_nearest_nodes
        o_dist=[gps_distance(req[["o_x","o_y"]].values.tolist(),(g.nodes[req["source"]]['x'],g.nodes[req["source"]]['y'])) for t,req in self.requests.iterrows()]
        d_dist=[gps_distance(req[["d_x","d_y"]].values.tolist(),(g.nodes[req["target"]]['x'],g.nodes[req["target"]]['y'])) for t,req in self.requests.iterrows()]
        self.requests["source_dist"]=o_dist
        self.requests["target_dist"]=d_dist
        self.requests["id"]=range(self.requests.index.size)   
        ts=self.requests["timestamp"]
        self.requests=self.requests.set_index(ts)
        print("complete requests data preprocessing")
        
        print("complete initialize requests")
        self.is_request_init=True
        
    
#    def calculate_shortest_path_cache(self,cache_num=2000):
#        source_count=self.requests["source"].value_counts()
#        target_count=self.requests["target"].value_counts()
#        vc=pd.concat([source_count,target_count],axis=1)
#        vc1=vc.sum(axis=1)
#        vc1=vc1.sort_values()
#        cache_nodes=vc1.index[:cache_num].tolist()
#        for n in cache_nodes:
#            pass
        
    def save_results(self,filename):
        try:
            ts=pd.Timestamp.now()
            date=ts.strftime("%Y%m%d")
            time=ts.strftime("%H%M%S")
            sname=self.strategy.__class__.__name__
            path=os.path.join(self.cache_dir,sname,'vehicles',
                              "{0}_{1}_{2}.pkl".format(filename, date, time))
            dir_=os.path.dirname(path)
            if os.path.exists(dir_) is False:
                os.makedirs(dir_)
            with open(path,'wb') as f:
                pickle.dump(self.vehicle_manager.vehicles,f)
            path=os.path.join(self.cache_dir,sname,'results',"{0}_{1}_{2}.pkl".format(filename, date, time))
            
            dir_=os.path.dirname(path)
            if os.path.exists(dir_) is False:
                os.makedirs(dir_)
            with open(path,'wb') as f:
                pickle.dump(self.results,f)
        except Exception as e :
            self.logger.exception(e)
            print(e,"save_results")

    def load_travel_time(self,time_id:int):
        w="travel_time_"+time_id+".pkl"
        path=os.path.join(self.cache_dir,w)
        with open(path,'rb') as f:
            d=pickle.load(f)
        self.travel_times=d
        return d
    
    def filter_requests(self):
        # print("filter invalid requests processing...")
        self.requests=self.requests[self.requests["source"]!=self.requests["target"]]

        in_degree_of_zero_set = set()
        out_degree_of_zero_set = set()

        for node, indegree in self.vehicle_manager.roads.in_degree:
            if indegree == 0:
                in_degree_of_zero_set.add(node)
        for node, outdegree in self.vehicle_manager.roads.out_degree:
            if outdegree == 0:
                out_degree_of_zero_set.add(node)

        self.requests = self.requests[~self.requests["source"].isin(in_degree_of_zero_set)]
        self.requests = self.requests[~self.requests["target"].isin(out_degree_of_zero_set)]

        # 存在回路，回路中的点是否要过滤？

        # print("complete filtering invalid requests")
        
    def run(self,is_save=True,report=0.1):
#        
        stime=pd.Timestamp.now()
        if self.strategy is None:
            raise ValueError("Not initialize ride sharing strategy!")
        if len(self.vehicle_manager.vehicles)==0:
            raise ValueError("Not initialize vehicles")
        if self.requests is None or len(self.requests.index)==0:
            raise ValueError("Not set requests data")
        if self.is_request_init is False:
            self.__init_requests(self.requests)
        
        self.filter_requests()

        start,end=self.get_request_date_range()
        self.config(start,end,self.freq)
        unit=1/report
        n=len(self.datetime_index)
        # print("Start simulating...")
        cur_tid=-1
        p=0
        step=100/report
        now=pd.Timestamp.now()
        for i,t in enumerate(self.datetime_index):
            
            self.timestamp=t
            p1=int(i/n*100*unit)
            if p1!=p:
                p=p1
                p0=i/n*100
                now1=pd.Timestamp.now()
                delta=now1-now
                delta=delta*(step-p)
                now=now1
                print("complete: {0} % timestamp:{1} last time:{2}".format(p0,t,delta))
#            tid=self.get_time_id(t)
#            if cur_tid!=tid:
#                cur_tid=tid
#                print("current time id:{0}\n load travel time of time id:{0}".format(cur_tid))
#                self.load_travel_time(cur_tid)
#                print("load travel time success")
#           print(i,t)
#             if t >= pd.Timestamp("2013-09-09 8:00:00"):
#                 self.process(t)
#             if t >= pd.Timestamp("2008-06-06 9:00:00") and t <= pd.Timestamp("2008-06-06 15:00:00"):
#                 self.process(t)
            self.process(t)

        etime=pd.Timestamp.now()
        run_delta=etime-stime
        info=self.calculate_statictis()
        info["run_time_delta"]=run_delta
        info["strategy_name"]=self.strategy.__class__.__name__
        info["start_timestamp"]=self.start_datetime
        info["end_timestamp"]=self.end_datetime
        if is_save:
            self.save_results(self.name)
        return info
            
#        for t in self.datetime_index:
#            self.event_engine.put(Event(name=t))
        
    def calculate_statictis(self):
        d={}
        try:
            ps,u_ps=passengers_analysis(self.vehicle_manager)
            d["passenger_records"]=ps
            d["u_passenger_records"]=u_ps
            d["served_ratio"]=round(ps.index.size/(self.requests.index.size-u_ps.index.size)*100,3)
        except:
            pass
        self.results=d
        
        return d
    
    def process(self,timestamp):
        
        self.__on_timestamp(timestamp)
        
        for vid,events in self.__moving_events.items():
            vh=self.vehicle_manager.vehicles[vid]
            while timestamp in events:
                node,location,ts=events.pop(timestamp)
                self.on_vehicle_arrive_node(vh,ts,node)
            
    def __on_timestamp(self,timestamp:pd.Timestamp):
        ts=timestamp
        if ts in self.requests.index:
            reqs:pd.DataFrame=self.requests.loc[[ts]]
            for t,req in reqs.iterrows():
                o=req["source"]
                d=req["target"]
                od_path,od_cost=self.vehicle_manager.shortest_travel_path_cost(o,d,timestamp)
                if od_path is not None:
                    r=req.to_dict()
                    r["path"]=[]
                    r["shortest_path"]=od_path
                    r["shortest_cost"]=od_cost
                    r["status"] = "unassigned"
                    self.strategy.on_requests(r,timestamp,self)
        
    def estimate_vehiles_location_node(self,timestamps:list):
        nodes={t:{} for t in timestamps}
        
        for t in timestamps:
            for vid,predict in self.vehicle_nodes_predict.items():
                k=list(predict.keys())[0]
                nodes[vid]=k
                for node,ts in predict.items():
                    
                    if ts<=t:
                        nodes[t][vid]=node
                    else:
                        break
        return nodes
        
    def append_pick_up_passenger_event(self,vehicle:Vehicle,passenger,):
        o=passenger["source"]
        handlers:dict=self.__passenger_events[(vehicle.id,o)]
        pid=passenger["id"]
        if pid not in handlers:
            handlers[pid]=(self.on_pick_up_passenger,passenger)
        else:
            print("Exists pick up event! vehicle:{0} passenger id:{1} node:{2}".format(vehicle.id,pid,o))
    
    def append_drop_off_passenger_event(self,vehicle:Vehicle,passenger):
        d=passenger["target"]
        handlers:dict=self.__passenger_events[(vehicle.id,d)]
        pid=passenger["id"]
        if pid not in handlers:
            handlers[pid]=(self.on_drop_off_passenger,passenger)
        else:
            print("Exists get off event! vehicle:{0} passenger id:{1} node:{2}".format(vehicle.id,pid,d))

    def set_strategy(self,strategy):
        strategy.batch_timestamp = None
        strategy.whole_batch_timestamp = None
        strategy.waiting_requests.clear()
        strategy.waiting_requests_batch.clear()

        self.strategy=strategy

    def set_date(self, date):
        self.date=date
        
    def config(self,start,end,freq=None):
        if freq is None:
            freq=self.freq
        else:
            self.freq=freq
        self.start_datetime=start
        self.end_datetime=end
        self.datetime_index=pd.date_range(start=start,end=end,freq=freq)
        self.time_delta=pd.Timedelta(1,unit=freq)

    def shortest_path_cost(self,source,target,weight:str):
        if source in self.travel_times:
            if target in self.travel_times[target]:
                return self.travel_times[source][target]
        return nx.dijkstra_path_length(self.vehicle_manager.roads,source,target,weight=weight)

    def on_vehicle_location_changed(self,vehicle:Vehicle,timestamp:pd.Timestamp,cur_location):
        
#        self.vehicle_manager.update_vehicle_location(vehicle,cur_location,vehicle.location)
        vehicle.update_location(cur_location,timestamp)

    def on_vehicle_arrive_node(self,vehicle:Vehicle,timestamp:pd.Timestamp,cur_node):
        self.strategy.on_vehicle_arrive_road_node(vehicle,timestamp,cur_node)
        vehicle.update_node(cur_node)
        location=(self.vehicle_manager.roads.nodes[cur_node]['x'],self.vehicle_manager.roads.nodes[cur_node]['y'])
        self.on_vehicle_location_changed(vehicle,timestamp,location)
        k=(vehicle.id,cur_node)
        
        if k in self.__passenger_events:
            
            events:dict=self.__passenger_events.pop(k)
            rids=list(events.keys())
            for rid in rids:
                handler,passenger=events.pop(rid)
                handler(vehicle,passenger,timestamp)
        
        
        if vehicle.next_node is None:
            self.on_vehicle_empty(vehicle,timestamp)
        else:
            self.simulate_movement(vehicle,timestamp)
        
    def on_vehicle_empty(self,vehicle,timestamp):
        pass
    
    def on_vehicle_fully(self,vehicle,timestamp):
        pass
    
    def on_pick_up_passenger(self,vehicle:Vehicle,passenger:dict,timestamp:pd.Timestamp):
        self.logger.info("on_pick_up_passenger: vehicle id:{0} carry num:{1},waiting passenger num:{2} timestamp:{3}".format(
                vehicle.id,vehicle.carry_num,vehicle.waiting_num,timestamp))
        self.strategy.on_pick_up_passenger(vehicle,passenger,timestamp)
        
        vehicle.pick_up_passenger(passenger,timestamp)
        self.append_drop_off_passenger_event(vehicle,passenger)

    def on_drop_off_passenger(self,vehicle:Vehicle,passenger:dict,timestamp:pd.Timestamp):
        self.logger.info("on_get_off_passenger: vehicle id:{0} carry num:{1},waiting passenger num:{2} timestamp:{3}".format(
                vehicle.id,vehicle.carry_num,vehicle.waiting_num,timestamp))
        vehicle.drop_off_passenger(passenger,timestamp)
        
    def simulate_movement(self,vehicle:Vehicle,timestamp:pd.Timestamp):
        if vehicle.next_node is not None:
            try:
                k=(vehicle.cur_node,vehicle.next_node,0)
                e=self.vehicle_manager.roads.edges[k]
                w="travel_time_{0}".format(self.get_time_id(timestamp))
                cost=e[w]
                delta=pd.Timedelta(cost,unit=self.freq)
                ts=timestamp+delta
                n=self.vehicle_manager.roads.nodes[vehicle.next_node]
                location=(n['x'],n['y'])
                self.append_movement(vehicle,ts,vehicle.next_node,location)
            except Exception as e1:
                print(e1)
                print("vid:{0} time:{1} cur_node:{2} k:{3}".format(vehicle.id,timestamp,vehicle.cur_node,k))
                raise ValueError("simulate_movement error")
        
    def append_movement(self,vehicle:Vehicle,timestamp:pd.Timestamp,node:int,location):
        ts=timestamp.ceil(self.freq)
        vid=vehicle.id
        handlers = self.__moving_events[vid]
        
        if ts not in handlers:
            handlers[ts]=(node,location,timestamp)
        else:
            print("time in vehicle movement handlers! vid:{0} timestamp:{1}".format(vid,timestamp))
    
    def update_vehicle_schedule(self,vehicle:Vehicle,timestamp:pd.Timestamp,schedule:list):
        vehicle.update_schedule(schedule)
        targets=[s[1] for s in schedule] #each s is a tuple whicle including (request id,road node, node type(pick up of drop off))

        cur=vehicle.cur_node
        path=[cur]
        for i in range(0,len(targets)):
            nex = targets[i]
            spath, scost = self.vehicle_manager.shortest_travel_path_cost(cur, nex, timestamp)
            if spath is None:
                return None
            spath1 = spath[1:]
            path.extend(spath1)
            cur = nex
        self.update_vehicle_path(vehicle,timestamp,path)
    
    def update_vehicle_path(self,vehicle:Vehicle,timestamp:pd.Timestamp,path:list):
        if path[0] == vehicle.cur_node:
            vehicle.update_path(path)
        else:
            print("path[0] != cur_node")
            path0,cost0=self.vehicle_manager.shortest_travel_path_cost(vehicle.cur_node,path[0],timestamp)
            path0.extend(path[1:])
            vehicle.update_path(path0)
            path=path0
    
        self.__moving_events[vehicle.id].clear()
        self.on_vehicle_arrive_node(vehicle,timestamp,path[0])
        ts=timestamp
        path_time={path[0]:ts}
        cur=path[0]
        for i in range(1,len(path)):
            nex=path[i]
            tid=self.get_time_id(ts)
            e=self.vehicle_manager.roads.edges[cur,nex,0]
            cost=e["travel_time_{0}".format(tid)]
            ts+=pd.Timedelta(cost,unit='s')
            path_time[nex]=ts
            cur=nex
        self.vehicle_nodes_predict[vehicle.id]=path_time

    def assign_vehicle_to_passenger(self,passenger,vehicle:Vehicle,timestamp:pd.Timestamp,**kwargs):
        vehicle.add_waiting_passenger(passenger)
        self.append_pick_up_passenger_event(vehicle,passenger)
#        self.append_passenger_event((vehicle.id,passenger["target"]),self.on_get_off_passenger,vehicle,passenger)

    @staticmethod
    def get_time_id(timestamp:pd.Timestamp,unit=60):
        t0=pd.Timestamp(timestamp.date())
        time_id = int((timestamp-t0).total_seconds()/60/unit)
        return time_id
    

    
    
    
