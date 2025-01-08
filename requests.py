# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:31:32 2020

@author: Administrator
"""
import pandas as pd 

def generate_requests_by_ods(ods:pd.DataFrame):
    o_x=ods["StartX"].values.tolist()
    o_y=ods["StartY"].values.tolist()
    d_x=ods["EndX"].values.tolist()
    d_y=ods["EndY"].values.tolist()
    
    time_index=ods["StartDate"].astype(str)+' '+ods["StartTime"].astype(str).values.tolist()
    timestamp=pd.to_datetime(time_index)
    requests=zip(o_x,o_y,d_x,d_y,timestamp)
    return list(requests)

def normalize_requests(req:list):
    reqs=pd.DataFrame(req,columns=['o_x','o_y',"d_x",'d_y',"timestamp"])
    reqs=reqs.set_index(reqs["timestamp"])
    reqs.sort_index(inplace=True)
    return reqs
    

def extract_ridesharing_od(passenger_records:pd.DataFrame):
    vids=passenger_records.vehicle_id.unique()
    ods=[]
    for vid in vids:
        ps=passenger_records[passenger_records["vehicle_id"]==vid]
        ps=ps.sort_values("pick_up_time")
        if ps.index.size>1:
            p0=ps.iloc[0]
            od=p0.to_dict()
            
            for i in range(1,ps.index.size):
               
                p2=ps.iloc[i]
                if p2.pick_up_time>od["drop_off_time"]:
                      ods.append(od)
                      od=p2.to_dict()
                else:
                    od["target"]=p2["target"]
                    od["drop_off_time"]=p2["drop_off_time"]
    ods_df=pd.DataFrame(ods)

#    ids=range(ods_df.index.size)
    ods_df.set_index(ods_df["id"],inplace=True)
    return ods_df



def get_unserved_ods(passenger_records:pd.DataFrame,od_requests:pd.DataFrame):
    
    served_ids=passenger_records["id"].unique()
    unserved_ids=set(od_requests["id"])-set(served_ids)
    od_df=od_requests.set_index(od_requests["id"])
    unserved_ods=od_df.loc[list(unserved_ids)]
    return unserved_ods

def merge_ods(passenger_records:pd.DataFrame,requests:pd.DataFrame):
    unserved_ods=get_unserved_ods(passenger_records,requests)
    eods=extract_ridesharing_od(passenger_records)
    df=eods
    df1=pd.concat([df,unserved_ods])
    return df1

    

    