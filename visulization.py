# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:18:41 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
import seaborn as sns
from ridesharing.manager import VehicleManager
import folium
import webbrowser as wb
import os
import math

def get_vehicles_location(vehicle_manager:VehicleManager):
    nodes={k:v.cur_node for k,v in vehicle_manager.vehicles.items()}
    locations={k:(g.nodes[v]['x'],g.nodes[v]['y']) for k,v in nodes.items()}
    return locations

def build_vehicles_location_folium(vehicle_location:dict):
    fk=list(vehicle_location.keys())[0]
    loc=list(vehicle_location[fk])
    loc.reverse()
    m=folium.Map(location=loc,zoom_start=14)
    
    for k,v in vehicle_location.items():
        v1=list(v)
        v1.reverse()
        folium.Marker(v1,popup='<i>id:{0}</i>'.format(k)).add_to(m)
    return m

def show_vehicles_locations(vehicle_manager:VehicleManager,path=".//vehicle_locations.html"):
    locations=get_vehicles_location(vehicle_manager)
    m=build_vehicles_location_folium(locations)
    m.save(path)
    p=os.path.abspath(path)
    wb.open(p)
    return m

def to_utm(vehicle_locations:dict):
    locs={k:wgs84toWebMercator(*v) for k,v in vehicle_locations.items()}
    return locs

def wgs84toWebMercator(lon,lat):
    x =  lon*20037508.342789/180
    y =math.log(math.tan((90+lat)*math.pi/360))/(math.pi/180)
    y = y *20037508.34789/180
    return x,y
