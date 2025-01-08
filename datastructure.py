# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:15:57 2020

@author: Administrator
"""
from collections import deque,defaultdict
import pandas as pd 
import logging
from enum import Enum
import networkx as nx

class VehicleStatus(Enum):
    waiting=0
    pick_up_moving=1
    get_off_moving=2
    
WAINTING=0
PICK_UP_MOVING=1
GET_OFF_MOVING=2
    
class Vehicle(object):
    
    
    def __init__(self,id_,capacity:int,location:list,status=0,available=True,company='',cur_node=-1):
        self.id=id_
        self.company=''
        self.__capacity=capacity
        self.__location=location
        self.__trajectory=[]
        self.__status=status
        self.passengers={}
        self.passenger_records=[]
        self.travel_distance=0
        self.travel_time=0
        self.loaded_distance=0
        self.loaded_time=0
        self.fully_loaded_time=0
        self.fully_loaded_distance=0
        self.travel_path=[]
        self.schedule=deque()
        self.path=deque()
    
        self.waiting_passengers={}
        self.__avail=available
        self.cur_node=cur_node
#        self.targets_graph=nx.DiGraph()
       
    
    @property
    def initial_info(self):
        return {"id":self.id,"capacity":self.capacity,"location":self.location,"status":self.status,"available":self.available,
                "company":self.company,"cur_node":self.cur_node}
        
    @property
    def status(self):
        return self.__status
    
    @property
    def cur_edge(self):
        return self.cur_node,self.cur_target
        
    @property
    def cur_path(self):
        cur_node,cur_target=self.cur_edge
        if cur_target is None:
            return []
        else:
            i=self.path.index(cur_node)
            ii=self.path.index(cur_target)
            return self.path[i,ii+1]
    
    @property
    def targets(self):
        return [node for rid,node in self.schedule]
    
    @property
    def cur_target(self):
        targets=self.targets
        if len(targets)>0:
           return targets[0]
        else:
            return None

    @status.setter
    def status(self,value):
        if isinstance(value,int):
            if value in (WAINTING,PICK_UP_MOVING,GET_OFF_MOVING):
                self.__status=value
            else:
                raise ValueError("vehicle status must in {1:waiting,2:pick_up_moving,3:get_off_moving}")
        else:
            raise ValueError("vehicle status must be int type")

    @property
    def available(self):
        return self.__avail
    
    @available.setter
    def available(self,value):
        if value is True:
            self.__avail=True
        else:
            self.__avail=False
    
    @property
    def waiting_num(self):
        return len(self.waiting_passengers)
    
    @property
    def avail_capacity(self):
        return self.__capacity-self.waiting_num-self.carry_num
    
    @property
    def carry_num(self):
        return len(self.passengers)
    

    @property
    def capacity(self):
        return self.__capacity
    @property
    def location(self):
        return self.__location
    
    @location.setter
    def location(self,value):
        
        self.__location=value
    
    @property
    def servered_num(self):
        return len(self.passenger_records)
    
    @property
    def trajectory(self):
        return self.__trajectory.copy()
    
        
    def is_fully(self):
        if self.cur_passenger_num==self.__capacity:
            return True
        else:
            return False
        
    def is_empty(self):
        if self.cur_passenger_num==0:
            return True
        else:
            return False
        
    def is_available(self):
        return self.__avail
    
    
    def reset_records(self):
        self.loaded_distance=0
        self.fully_loaded_distance=0
        self.loaded_time=0
        self.fully_loaded_time=0
        self.travel_distance=0
        self.travel_time=0
        
    @property 
    def next_node(self):
        
        if len(self.path)>1:
            return self.path[1]
        else:
            return None
        
    def set_cur_node(self,cur_node):
        self.cur_node=cur_node
        
    def update_node(self,cur_node):
        if len(self.schedule)>0:  
            pass
        if cur_node==self.next_node:
            self.path.popleft()
            for i,p in self.passengers.items():
                p["path"].append(cur_node)
            self.cur_node=cur_node
        elif cur_node==self.cur_node:
            pass
        elif cur_node!=self.cur_node and cur_node!=self.next_node:
            if len(self.path)>0:
                self.cur_node=self.path[0]
        
      
        
            
    def update_location(self,location:list,timestamp):
        
        self.__trajectory.append((self.location[0],self.location[1],timestamp,self.carry_num,self.cur_node))
        self.location=location
        
    def update_schedule(self,schedule:list):
        self.schedule.clear()
        self.schedule.extend(schedule)
    def update_path(self,path:list):
        self.path.clear()
        self.path.extend(path)
        
    def get_cost_of_target(self,index:int):
        edges=self.schedule[:index+1]
        costs=[self.targets_graph.edges[e]["cost"] for e in edges]
        return sum(costs)
        
        
    
    def get_schedule(self):
        return self.schedule

    def pick_up_passenger(self,passenger:dict,timestamp:pd.Timestamp):
        k=(passenger["id"],self.cur_node,0)
        if k in self.schedule:
            self.schedule.remove(k)
            
        passenger["path"]=[self.cur_node]
        passenger["vehicle_id"]=self.id
        if passenger["id"] in self.waiting_passengers:
            p=self.waiting_passengers.pop(passenger["id"])
        if self.avail_capacity>0:
            self.passengers[passenger["id"]]=passenger
            passenger["pick_up_time"]=timestamp
            if self.avail_capacity==0:
                self.__avail=False
        else:
            raise ValueError("Have no capacity for picking up passengers")
        
        
    
    def drop_off_passenger(self,passenger,timestamp:pd.Timestamp):
        k=(passenger["id"],self.cur_node,1)
        if k in self.schedule:
            self.schedule.remove(k)
        if passenger["id"] in self.passengers:
            p=self.passengers.pop(passenger["id"])
            self.__avail=True
            p["drop_off_time"]=timestamp
            self.passenger_records.append(p)
 
            
        else:
            raise ValueError("Not contain passenger:{0}".format(passenger["id"]))
    
    def add_waiting_passenger(self,passenger):
        self.waiting_passengers[passenger["id"]]=passenger
        
    def remove_waiting_passenger(self,id_):
        if id_ in self.waiting_passengers:
            self.waiting_passengers.pop(id_)
        

    def clear(self):
        self.waiting_passengers.clear()
        self.passengers.clear()
        self.passenger_records.clear()
        self.__trajectory.clear()
        self.schedule.clear()
        self.path.clear()
     
        
    def to_dict(self):
        return {"id":self.id,"cur_node":self.cur_node,"trajectory":self.__trajectory,"passenger_records":self.passenger_records}
        
    


FEMALE=0
MALE=1
class Passenger(object):
    
    
    def __init__(self,id_='',target=-1,source=0,timestamp:pd.Timestamp='',cost=0):
        self.id=id_
        self.request_time=timestamp
        self.target=target
        self.source=source
        self.travel_path=path
        self.shortest_cost=cost
        self.pick_up_time=None
        self.get_off_time=None
        
        
        
    @property
    def o_location(self):
        return (self.o_x,self.o_y)
    
    @property
    def d_location(self):
        return (self.d_x,self.d_y)
    
    def __getitem__(self,key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return None
    def keys(self):
        return list(self.__dict__.keys())
        
    def __setitem__(self,key,value):
        self.__dict__[key]=value
    