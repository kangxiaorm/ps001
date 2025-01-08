# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:20 2020

@author: Administrator
"""
import pandas as pd
from ridesharing.manager import VehicleManager,get_weight_of_path
import networkx as nx 
import osmnx as ox
import os 


def get_time_id(timestamp:pd.Timestamp,unit=60):
    t0=pd.Timestamp(timestamp.date())
    time_id = int((timestamp-t0).total_seconds()/60/unit)
    return time_id

def statistic_analysis(vehicle_manager:VehicleManager,requests:pd.DataFrame):
    servered_nums={i:len(vh.passenger_records) for i,vh in vehicle_manager.vehicles.items()}
    total_servered_num=sum(servered_nums.values())
    served_percentage=round(total_servered_num/requests.index.size*100,3)
    
def responding_ratio(vehicle_manager:VehicleManager,requests_num):
    servered_nums={i:len(vh.passenger_records) for i,vh in vehicle_manager.vehicles.items()}
    total_servered_num=sum(servered_nums.values())
    servered_percentage=round(total_servered_num/requests_num.index.size*100,3)
    serverd_distribution=pd.Series(servered_nums)
    serverd_distribution=serverd_distribution.sort_values(ascending=False)
    return servered_nums,servered_percentage


def cal_passenger_sharing_time(served_passengers:pd.DataFrame):
    
    share_times={}
    vhs=served_passengers.vehicle_id.unique()
    for vh in vhs:
        print("processing vehicle:{0}".format(vh))
        vh_passengers=served_passengers[served_passengers.vehicle_id==vh]
        vh_passengers=vh_passengers.sort_values("pick_up_time")
        vh_ps={}
        pick_events=pd.DataFrame({"timestamp":vh_passengers.pick_up_time,"type":1})
        drop_events=pd.DataFrame({"timestamp":vh_passengers.drop_off_time,"type":0})
        events=pick_events.append(drop_events)
        events.sort_values("timestamp",inplace=True)
        
        for i,row in events.iterrows():
            ts=row["timestamp"]
            tp=row["type"]
            if tp==1:
                p={"id":i,"ptime":ts,"type":tp,"st":0,"share_ts":None}
                if len(vh_ps)==0:
#                    p["smode"]=0
                    pass
                else:
#                    p["smode"]=1
                    p["share_ts"]=ts
                    for id_,v in vh_ps.items():
                        if v["share_ts"] is None:
                            v["share_ts"]=ts
                vh_ps[i]=p
                
            else:
                p=vh_ps.pop(i)
                if p["share_ts"] is not None:
                    delta=ts-p["share_ts"]
                    s=delta.total_seconds()
                    p["st"]+=s
                    
                share_times[p["id"]]=p["st"]
                
                for id_,v in vh_ps.items():
                    
                    if len(vh_ps)==1:
                        delta=ts-v["share_ts"]
                        v["st"]+=delta.total_seconds()
                        v["share_ts"]=None
                    else:
                        pass
       
            
    served_passengers["share_time"]=pd.Series(share_times)
    return served_passengers

        
    
def passengers_analysis(vehicle_manager:VehicleManager,to_pandas=True):
    
    passengers=[]
    unserved_passengers=[]
    for vid,vh in vehicle_manager.vehicles.items():
        print("processing vehicle id:{0}".format(vid))
        for passenger in vh.passenger_records:
            passenger["vehicle_id"]=vid
            delta=passenger["drop_off_time"]-passenger["pick_up_time"]
            passenger["travel_time"]=round(delta.total_seconds(),3)
            passenger["shortest_cost"]=round(passenger["shortest_cost"],3)
            cost,ts=vehicle_manager.calculate_path_travel_time(passenger["path"],passenger["pick_up_time"])
            passenger["expacted_cost"]=round(cost,3)
            passenger["travel_distance"]=vehicle_manager.calculate_path_distance(passenger["path"])
            passenger["shortest_distance"]=vehicle_manager.calculate_path_distance(passenger["shortest_path"])
            passenger["additional_distance"]=passenger["travel_distance"]-passenger["shortest_distance"]
            passenger["additional_cost"]=round(passenger["travel_time"]-passenger["shortest_cost"],3)
            passenger["waiting_time"]=(passenger["pick_up_time"]-passenger["timestamp"]).total_seconds()
            passengers.append(passenger)
        for unserved_passenger in vh.passengers.values():
            unserved_passengers.append(unserved_passenger)

            
    if to_pandas:
        passengers1=pd.DataFrame(passengers)
        unserved_passengers1=pd.DataFrame(unserved_passengers)
        passengers1=passengers1.set_index(passengers1["id"])
        unserved_passengers1 = unserved_passengers1.set_index(unserved_passengers1['id'])
        cal_passenger_sharing_time(passengers1)
        return passengers1, unserved_passengers1
    return passengers, unserved_passengers

def vehicles_analysis(vehicle_manager:VehicleManager):
    pass


class RidesharingAnalysis:
    
    
    def __init__(self,requests:pd.DataFrame,served_passengers:pd.DataFrame,unserved_passengers,root_dir=".//risharing_analysis",batch=5):
            
        self.root_dir=os.path.abspath(root_dir)
        
        self.served_passengers=served_passengers

        # 去除最后一个batch未处理的请求
        served_end = served_passengers["timestamp"].max()
        start = requests["timestamp"].value_counts().sort_index().index[0]
        end = max(start + pd.Timedelta(minutes=60*24-batch), served_end)
        requests = requests[requests["timestamp"] <= end]

        # 去除访问上车点但未访问下车点的请求
        requests = requests[~requests["id"].isin(unserved_passengers["id"].values)]

        self.total_requests=requests
    
    def analysing(self,root_dir=None,is_save=True):
        if root_dir is None:
            root_dir=self.root_dir
        group = self.group_by_time(self.served_passengers,"timestamp",60)
        
        r={
                "responding_ratio":self.total_responding_ratio(),
                "sharing_ratio":self.total_sharing_ratio(),
                "mean_waiting_time":self.total_mean_waiting_time(),
                "additional_distance_ratio":self.total_distance_additional_ratio(),
                "additional_time_ratio":self.total_time_additional_ratio(),
                "hourly_responding_ratio":self.hourly_responding_ratio(),
                "hourly_sharing_ratio":self.hourly_sharing_ratio(group=group),
                "hourly_mean_waiting_time":self.hourly_mean_waiting_time(group=group),
                "hourly_additional_distance_ratio":self.hourly_distance_additional_ratio(group=group),
                "hourly_additional_time_ratio":self.hourly_time_additional_ratio(group=group)
                }
        return r
    
    
    def responding_ratio(self,requests_num,served_passenger_num,round_=3):
        try:
            return round(served_passenger_num/requests_num,round_)
        except:
            return 0
    
    @staticmethod
    def get_time_range(time_unit=60):
        tmins=60*24
        t0=pd.Timestamp("00:00:00")

    @staticmethod
    def group_by_time(data:pd.DataFrame,sname:str,time_unit=60):
        reqs=data.copy()
        tmins=60*24
        tintervals=int(tmins/time_unit)
        tids=reqs[sname].apply(lambda x: get_time_id(x,unit=time_unit))
        TID="tid"
        reqs[TID]=tids
        group={i:reqs[reqs[TID]==i] for i in range(tintervals) }
        return group
        
    def hourly_responding_ratio(self,time_unit:int=60):
        req_group=self.group_by_time(self.total_requests,"timestamp",time_unit)
        ps_group=self.group_by_time(self.served_passengers,"timestamp",time_unit)
        ratio={i:self.responding_ratio(req_group[i].index.size,ps_group[i].index.size) for i in req_group.keys()}
        return ratio

    def get_numbers_of_request(self,time_unit:int=60):
        req_group = self.group_by_time(self.total_requests, "timestamp", time_unit)
        count={i:req_group[i].index.size for i in req_group.keys()}
        return count
    
    def total_responding_ratio(self):
        return self.responding_ratio(self.total_requests.index.size,self.served_passengers.index.size)

    def sharing_ratio(self,served_passengers:pd.DataFrame):
        d=served_passengers[served_passengers.share_time>0]
        try:
            return round(d.index.size/served_passengers.index.size,5)
        except:
            return 0
    
    def vehicles_served_num(self):
        return self.served_passengers.index.size
    
    def hourly_sharing_ratio(self,group=None, time_unit:int=60):
        group=self.group_by_time(self.served_passengers,"timestamp",time_unit) if group is None else group
        gr={h:self.sharing_ratio(ps) for h,ps in group.items()}
        return gr

    def total_sharing_ratio(self):
        return self.sharing_ratio(self.served_passengers)
    
    def total_mean_waiting_time(self):
        return self.served_passengers["waiting_time"].mean()
    
    def hourly_mean_waiting_time(self,group=None,sname="waiting_time",time_unit:int=60):
        group=self.group_by_time(self.served_passengers,"timestamp",time_unit) if group is None else group
        gr={h:ps[sname].mean() for h,ps in group.items()}
        return gr

    def hourly_mean_share_time(self,group=None,sname="share_time"):
        group=self.group_by_time(self.served_passengers,"timestamp") if group is None else group
        gr={h:ps[sname].mean() for h,ps in group.items()}
        return gr
    
    def additional_ratio(self,actual_cost,shortest_cost,round_=3):
        return round(actual_cost/shortest_cost,round_)
    
    def total_distance_additional_ratio(self):
        return self.additional_ratio(self.served_passengers.travel_distance.mean(),self.served_passengers.shortest_distance.mean())
    
    def hourly_distance_additional_ratio(self,group=None):
        group=self.group_by_time(self.served_passengers,"timestamp") if group is None else group
        gr={h:self.additional_ratio(ps.travel_distance.mean(),ps.shortest_distance.mean()) for h,ps in group.items()}
        return gr

    def total_time_additional_ratio(self):
        return self.additional_ratio(self.served_passengers.travel_time.mean(),self.served_passengers.shortest_cost.mean())

    def hourly_time_addition(self,group=None,time_unit:int=60):
        group = self.group_by_time(self.served_passengers, "timestamp",time_unit) if group is None else group
        gr = {h: ps.additional_cost.mean() for h, ps in group.items()}
        return gr

    def hourly_time_additional_ratio(self,group=None):
        group=self.group_by_time(self.served_passengers,"timestamp") if group is None else group
        gr={h:self.additional_ratio(ps.travel_time.mean(),ps.shortest_cost.mean()) for h,ps in group.items()}
        return gr

    def hourly_numbers_of_vehicles(self,group=None,time_unit:int=60):
        group = self.group_by_time(self.served_passengers, "timestamp",time_unit) if group is None else group
        gr = {h: len(ps.vehicle_id.unique()) for h, ps in group.items()}
        return gr

    def hourly_rides_per_driver(self,group=None,time_unit:int=60):
        group = self.group_by_time(self.served_passengers, "timestamp",time_unit) if group is None else group
        try:
            gr = {h: ps.index.size / len(ps.vehicle_id.unique()) for h, ps in group.items()}
            return gr
        except:
            return 0
    
