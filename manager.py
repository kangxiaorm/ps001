# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:05:48 2020

@author: Administrator
"""
import sys
curPath = "D:/cjx_code/fleet_rs_v_1.0"
if curPath not in sys.path:
    sys.path.append(curPath)

import networkx as nx
import osmnx as ox
import geopandas as gpd
from ridesharing.datastructure import Vehicle
from rtree.index import Index as RIndex
import pandas as pd
from haversine import haversine, Unit
import math
from collections import deque, defaultdict
from shapely.geometry import Point, LineString, Polygon
import numpy as np


EARTH_RADIUS = 6371000
LONGITUDE_CIRCUS = 31544206


def gps_distance(pt1, pt2):
    r = haversine((pt1[1], pt1[0]), (pt2[1], pt2[0]), Unit.METERS)
    return r


def meters_to_latitude(meters: float):
    circu = 2*math.pi*EARTH_RADIUS
    unit = 360/circu
    return unit*meters


def meters_to_longitude(meters: float, latitude: float):
    circu = 2*math.pi*EARTH_RADIUS*math.cos(latitude)
    unit = 360/circu
    return unit*meters


def get_time_id(timestamp: pd.Timestamp, unit=60):
    t0 = pd.Timestamp(timestamp.date())
    time_id = int((timestamp-t0).total_seconds()/60/unit)
    return time_id


def get_weight_of_path(g: nx.MultiDiGraph, path: list, weight: str):
    v = 0
    for i in range(0, len(path)-1):

        node = path[i]
        next_node = path[i+1]
        k = (node, next_node, 0)
        w = g.edges[k][weight]
        v += w
    return v


def get_edges_of_path(g: nx.MultiDiGraph, path: list, weight: str):
    es = [(path[i], path[i+1], 0) for i in range(0, len(path)-1)]
    return es


def get_reachable_nodes(g: nx.MultiDiGraph, source_node, threshold: float, weight: str, is_sort=True):

    costs, paths = nx.single_source_dijkstra(
        g, source_node, cutoff=threshold, weight=weight)

    return costs, paths

#    visited=defaultdict(dict)
#    stack=deque([source_node])
#    sum_cost=0
#    costs=defaultdict(list)
#    costs[source_node].append((0,source_node,))
#    while len(stack)>0:
#        cur=stack[-1]
#        suc_nodes=g.successors(cur)
#        k=tuple(stack)
#        not_visited=[n for n in suc_nodes if visited[k].get(n,False) is False]
#        if len(not_visited)>0:
#            for node in not_visited:
#                k=(cur,node,0)
#                cost=g.edges[k][weight]
#                visited[k][node]=True
#                if sum_cost+cost<=threshold:
#                    stack.append(node)
#                    sum_cost+=cost
#                    costs[node].append((sum_cost,tuple(stack)))
#                    break
#        else:
#            pop=stack.pop()
#            if len(stack)>0:
#                cur=stack[-1]
#                k=(cur,pop,0)
#                cost=g.edges[k][weight]
#                sum_cost-=cost
#
#    d=dict(costs)
#    if is_sort is True:
#        for k,v in d.items():
#            d[k]=sorted(v,key=lambda x:x[0])
#    return d


def get_reverse_reachable_nodes(g: nx.MultiDiGraph, target_node, threshold: float, weight: str, is_sort=True):

    visited = {n: {} for n in g.nodes}
    stack = deque([target_node])
    sum_cost = 0
    costs = defaultdict(list)
    costs[target_node].append((0, (target_node,)))
    while len(stack) > 0:
        cur = stack[0]
        pre_nodes = g.predecessors(cur)
        not_visited = [
            n for n in pre_nodes if visited[cur].get(n, False) is False]
        if len(not_visited) > 0:
            for node in not_visited:
                k = (node, cur, 0)
                cost = g.edges[k][weight]
                visited[cur][node] = True
                if sum_cost+cost <= threshold:
                    stack.appendleft(node)
                    sum_cost += cost
                    costs[node].append((sum_cost, tuple(stack)))
                    break
        else:
            pop = stack.popleft()
            if len(stack) > 0:
                cur = stack[0]
                k = (pop, cur, 0)
                cost = g.edges[k][weight]
                sum_cost -= cost

    d = dict(costs)
    if is_sort is True:
        for k, v in d.items():
            d[k] = sorted(v, key=lambda x: x[0])
    return d


class RegionManager(object):

    def __init__(self):
        pass


class VehicleManager(object):

    def __init__(self, roads: nx.MultiDiGraph, vehicles={}, name='', cache_size=3000):
        if isinstance(roads, (nx.DiGraph, nx.MultiDiGraph)) is False:
            raise ValueError(
                "roads must be type of nx.DiGraph or nx.MultiDiGraph")
        self.roads = roads
        self.vehicles = vehicles
        ngdf, egdf = ox.graph_to_gdfs(roads)
        self.nodes_gdf: gpd.GeoDataFrame = ngdf
        self.edges_gdf: gpd.GeoDataFrame = egdf
        self.travel_cache = {}
        self.distance_cache = {}
        self.cache_size = cache_size

    def reinitialize_vehicles(self):
        vhs0 = self.vehicles
        vhs = {k: Vehicle(*list(v.values()))
               for k, v in self.vehicles_initial_info.items()}
        self.init_vehicles(vhs)
        return vhs0

    def init_vehicles(self, vehicles: dict):
        self.vehicles = vehicles
        self.vehicles_initial_info = {
            k: v.initial_info for k, v in vehicles.items()}
        self.vehicles_code = {k: i for i, k in enumerate(vehicles.keys())}
        self.vehicles_rtree = RIndex()

        for id_, vh in self.vehicles.items():
            v: Vehicle = vh
            self.add_available_vehicle(v)

    def clear(self):
        self.vehicles.clear()
        self.vehicles_code.clear()
        self.vehicles_rtree = None

    def clear_vehicles_data(self):
        for i, v in self.vehicles.items():
            v.clear()

    def uniformed_initialize_vehicles(self, num=500, capacity=4, seed=42):
        roads: nx.MultiDiGraph = self.roads
        nodes = list(roads.nodes)
        if seed is not None:
            np.random.seed(seed)
        init_nodes = [nodes[n]
                      for n in np.random.randint(1, len(roads.nodes), size=num)]
        init_coords = [(roads.nodes[n]['x'], roads.nodes[n]['y'])
                       for n in init_nodes]
        init_vehicles = {i: Vehicle(
            i, capacity, coord, cur_node=init_nodes[i]) for i, coord in enumerate(init_coords)}
        self.init_vehicles(init_vehicles)
        return init_vehicles

    def add_available_vehicle(self, vehicle: Vehicle):
        #        if vehicle.id not in self.vehicles:
        #            self.vehicles[vehicle.id]=vehicle
        #        if vehicle.id not in self.vehicles_code:
        #            n=len(self.vehicles_code)
        #            self.vehicles_code[vehicle.id]=n+1
        code = self.vehicles_code[vehicle.id]
        self.vehicles_rtree.insert(code, vehicle.location, vehicle)

    def remove_available_vehicle(self, vehicle: Vehicle):

        code = self.vehicles_code[vehicle.id]
        self.vehicles_rtree.delete(code, vehicle.location)

#    def add_unavailable_vehicle(self,vehicle:Vehicle):
# if vehicle.id not in self.vehicles:
# self.vehicles[vehicle.id]=vehicle
# if vehicle.id not in self.vehicles_code:
# n=len(self.vehicles_code)
# self.vehicles_code[vehicle.id]=n+1
#        code=self.vehicles_code[vehicle.id]
#        self.unvehicles_rtree.insert(code,vehicle.location,vehicle)
#
#    def remove_unavailable_vehicle(self,vehicle:Vehicle):
#
#        code=self.vehicles_code[vehicle.id]
#        self.unvehicles_rtree.delete(code,vehicle.location)

    def on_update_vehicle_location(self, vid, timestamp: pd.Timestamp, node: int = None, location=None):
        pass

    def pick_up_passenger(self, vh: Vehicle, passenger, timestamp: pd.Timestamp):
        pass

    def drop_off_passenger(self, vh: Vehicle, passenger, timestamp: pd.Timestamp):
        pass

    def get_nearest_node(self, location):
        a = list(self.nodes_gdf.sindex.nearest(location))
        node = self.nodes_gdf.iloc[a[0]]
        d = gps_distance(list(node.geometry.coords)[0], location)
        return node.name, d

    def get_vehicles_by_grid(self, location, coordinates, type_="avialable", return_dist=True):
        if type_ == "available":
            idx = self.vehicles_rtree
        else:
            idx = self.unavilable_vehicles
        vh = self.query_vehicles(coordinates, idx)
        if return_dist is False:
            return vh
        else:
            vh1 = [(v, gps_distance(v.location, location)) for v in vh]
            return vh1

    def get_vehicles_by_radius(self, location, radius: float = 1000, type_="available"):
        if type_ == "available":
            idx = self.vehicles_rtree
        else:
            idx = self.unavilable_vehicles

        lon_diff = meters_to_longitude(radius, location[1])
        lat_diff = meters_to_latitude(radius)
        left = location[0]-lon_diff
        right = location[0]+lon_diff
        bottom = location[1]-lat_diff
        top = location[1]+lat_diff
        vh = self.query_vehicles((left, bottom, right, top), idx)
        r = []
        for v in vh:
            d = gps_distance(v.location, location)
            if d <= radius:
                r.append((v, d))
#        r=[v for v in vh if gps_distance(v.location,location)<=radius]
        return r

    def get_travel_time_of_vehicle(self, location, timestamp: pd.Timestamp, vehicle):
        return self.shortest_travel_path(vehicle.location, location, timestamp)

    def query_vehicles(self, coordinates, idx: RIndex):
        v = idx.intersection(coordinates, True)
        vh = [i.object for i in v]
        return vh

    def get_nearest_reachable_vehicle(self, location, radius: float = 5000, type_="available"):
        if type_ == "available":
            idx = self.vehicles_rtree
        else:
            idx = self.unavilable_vehicles
        vs = self.get_vehicles_by_radius(location, radius, type_)

    def search_converage_by_location(self, location, connect_time: float, timestamp: pd.Timestamp):
        node, dists = self.get_nearest_node(location)
        return self.search_converage(self, node, connect_time, timestamp)

    def search_converage(self, source_node: int, connect_time: float, timestamp: pd.Timestamp):
        tid = self.get_time_id(timestamp)
        weight = "travel_time_{0}".format(tid)
        cover_nodes = get_reachable_nodes(
            self.roads, source_node, connect_time, weight)
        return cover_nodes

    def search_reachable(self, target: int, connect_time: float, timestamp: pd.Timestamp):
        itd = self.get_time_id(timestamp)
        weight = "travel_time_{0}".format(tid)
        reachable = get_reverse_reachable_nodes(
            self.roads, target, connect_time, weight)
        return reachable

    def search_vehicle_coverage(self, vehicle_id, connect_time: float, timestamp: pd.Timestamp):
        cur_node = self.vehicles[vehicle_id].cur_node
        return self.search_converage(cur_node, connect_time, timestamp)

    def search_reachable_vehicles(self, target_node: int, connect_time: float, timestamp: pd.Timestamp, available=True):
        tid = self.get_time_id(timestamp)
        weight = "travel_time_{0}".format(tid)
        cover_nodes = get_reverse_reachable_nodes(
            self.roads, target_node, connect_time, weight)
        vh_nodes = {k: v.cur_node for k, v in self.vehicles.items()}

        reach_vhs = {}
        reach_vhs_path = {}
        for vid, vnode in vh_nodes.items():
            if vnode in cover_nodes:
                if available:
                    if self.vehicles[vid].available is False:
                        continue
                reach_vhs[vnode] = vid
                reach_vhs_path[vid] = cover_nodes[vnode]
        vhs_paths = sorted(reach_vhs_path.items(), key=lambda x: (
            x[1][0][0], self.vehicles[x[0]].carry_num))
        return vhs_paths

    def calculate_path_distance(self, path: list):
        length = 0

        for i in range(len(path)-1):
            k = (path[i], path[i+1], 0)
            w = "length"
            c = self.roads.edges[k][w]
            length += c
        return length

    def calculate_path_travel_time(self, path: list, timestamp: pd.Timestamp):
        cost = 0
        t = timestamp

        for i in range(len(path)-1):
            k = (path[i], path[i+1], 0)
            w = "travel_time_{0}".format(self.get_time_id(t))
            c = self.roads.edges[k][w]
            delta = pd.Timedelta(c, unit='s')
            t = t+delta
            cost += c

        return cost, t

    def update_vehicle_location(self, vehicle: Vehicle, cur_location, pre_location):
        code = self.vehicles_code[vehicle.id]
        self.vehicles_rtree.delete(code, pre_location)
        self.vehicles_rtree.insert(code, cur_location, vehicle)

    def shortest_travel_path(self, source_node, target_node, weight: str):
        try:
            pass

        except:
            return None

    def shortest_travel_path_cost(self, source, target, timestamp: pd.Timestamp):

        try:
            tid = self.get_time_id(timestamp)
            k = (source, target, tid)
            if k in self.travel_cache:
                return self.travel_cache[k]
            else:
                w = "travel_time_{0}".format(tid)
                cost, path = nx.single_source_dijkstra(
                    self.roads, source, target, weight=w)
                self.travel_cache[k] = (path, cost)
                if len(self.travel_cache) > self.cache_size:
                    fk = list(self.travel_cache.keys())[0]
                    self.travel_cache.pop(fk)
                return path, cost

        except Exception as e:
            return None, None

    def shortest_travel_cost(self, source_node, target_node, weight: str):
        try:
            if source_node in self.travel_times:
                if target_node in self.travel_times[source_node]:
                    return self.travel_times[source_node][target_node]
                else:
                    return None
            else:
                c = nx.dijkstra_path_length(
                    self.roads, source_node, target_node, weight)
                return c
        except:
            return None

    def get_vehicles_location(self, vehicle_ids):
        locations = {k: v.location for k, v in vehicle_ids.items()}
        return locations

    def get_vehicles_location_node(self, vehicle_ids):
        nodes = {k: v.cur_node for k, v in vehicle_ids.items()}
        return nodes

    def get_euclidean_nearest_vehicles(self, location, num, type_="available"):
        if type_ == "available":
            idx = self.vehicles_rtree
        else:
            idx = self.unavilable_vehicles
        v = idx.nearest(location, num, True)
        vh = [ri.object for ri in v]
        vh_dist = [gps_distance(v.location, location) for v in vh]
        s = pd.Series(vh_dist, index=vh)
        s.sort_values(inplace=True)
        return s

    def get_nearest_ndoes(self, location):
        a = list(self.nodes_gdf.sindex.nearest(location))
        node = self.nodes_gdf.iloc[a[0]]
        d = gps_distance(list(node.geometry.coords)[0], location)
        return node.name, d

    def get_nearest_nodes(self, location, num_of_results=1):
        a = list(self.nodes_gdf.sindex.nearest(location, num_of_results))
        nodes = self.nodes_gdf.iloc[a]
        ds = {n: gps_distance(list(node.geometry.coords)[
                              0], location) for n, node in nodes.iterrows()}
        s = pd.Series(ds)
        s.sort_values(inplace=True)
        return s

    def shortest_travel_path_by_location(self, start, end, timestamp: pd.Timestamp):

        s_node, s_dist = self.get_nearest_node(start)
        e_node, e_dist = self.get_nearest_node(end)
        # r = self.shortest_travel_path_by_node(
        #     s_node, e_node, timestamp, s_dist, e_dist)
        r = self.shortest_travel_path_cost(s_node, e_node, timestamp)
        return r

    def shortest_travel_path_by_node(self, start_node, end_node, timestamp: pd.Timestamp, start_dist: float = 0, end_dist: float = 0):
        tid = self.get_time_id(timestamp)
        weight = "travel_time_{0}".format(tid)
        try:
            cost, path = nx.single_source_dijkstra(
                self.roads, start_node, end_node, weight=weight)
            return path, cost
        except:
            return None, None

    def shortest_distance_path(self, start, end, timestamp: pd.Timestamp):
        s_node, s_dist = self.get_nearest_node(start)
        e_node, e_dist = self.get_nearest_node(end)
        r = self.shortest_travel_path_node(
            s_node, e_node, timestamp, s_dist, e_dist)
        return r

    def shortest_distance_path_by_node(self, start_node, end_node, timestamp: pd.Timestamp, start_dist: float = 0, end_dist: float = 0):
        tid = self.get_time_id(timestamp)
        weight = "length".format(tid)
        length, path = nx.single_source_dijkstra(
            self.roads, start_node, end_node, weight=weight)
        return path, length

    @staticmethod
    def get_time_id(timestamp: pd.Timestamp, unit=60):
        t0 = pd.Timestamp(timestamp.date())
        time_id = int((timestamp-t0).total_seconds()/60/unit)
        return time_id


class RegionManager(object):

    def __init__(self):
        pass
