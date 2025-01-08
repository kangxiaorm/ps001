# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:30:00 2020

@author: Administrator
"""
import copy
import datetime
import sys

import pandas as pd
from collections import deque

import torch
import os
import json
import time

from ridesharing.engine import SimulatorEngine
import networkx as nx
import math
import random
from ridesharing.datastructure import Vehicle
from ridesharing.manager import VehicleManager
import logging
import itertools
from collections import defaultdict
import typing
import pickle
import csv

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from .nets.critic_network import CriticNetwork
from ridesharing.options import get_options
from .train import get_inner_model
from .reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from .nets.attention_model import AttentionModel
#from nets.attention_model_minsum import AttentionModel
from .nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from .utils.log_utils import log_values, log_batch_values
from .utils import torch_load_cpu, move_to

from .problems.ridesharing.problem_ridesharing import RideSharing
from .nets.attention_model import set_decode_type
from statistics import mean

# logging.getLogger().setLevel(logging.DEBUG) # 直接使用logging输出日志
# logging.basicConfig(filename="train.log", filemode="w", format="%(asctime)s %(name)s : %(levelname)s : %(message)s",
#                     datefmt="%Y-%M-%d %H:%M:%S", level=logging.ERROR)

class SharingStrategy(object):

    def __init__(self):
        self.__description = '''This is a base ride sharing strategy '''
        self.parameters = {}
        self.parameters_discrption = []
        self.logger = logging.getLogger("{0}.{1}".format(self.__class__.__module__, self.__class__.__name__))
        self.logger.setLevel(level=logging.INFO)   # DEBUG < INFO < WARNING < ERROR < CRITICAL
        handler = logging.FileHandler("train.log")  # FileHandler 文件输出
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(filename)s[%(funcName)s line:%(lineno)d] - %(levelname)s : %(message)s'))
        self.logger.addHandler(handler)

    def description(self):
        return self.__description

    def clear(self):
        pass

    def on_requests(self, req: pd.Series, timestamp: pd.Timestamp):
        if callable(self.dispatch):
            self.dispatch(req, timestamp)

    def set_parameter(self, name: str, value):
        self.parameters[name] = value

    def get_parameter(self, name: str):
        self.parameters.get(name)

    def get_time_constraint(self, vehicle: Vehicle, waiting_time: float, delta_time: float, unit='s'):
        con = {}
        for rid, p in vehicle.waiting_passengers.items():
            con[(rid, p["source"], 0)] = p["timestamp"] + pd.Timedelta(waiting_time, unit=unit)
            con[(rid, p["target"], 1)] = p["timestamp"] + pd.Timedelta(waiting_time + p["shortest_cost"] + delta_time,
                                                                       unit=unit)

        for rid, p in vehicle.passengers.items():
            con[(rid, p["target"], 1)] = p["pick_up_time"] + pd.Timedelta(p["shortest_cost"] + delta_time, unit=unit)
        return con

    @staticmethod
    def digraph_weight_of_path(g: nx.DiGraph, path, weight: str):

        v = 0
        for i in range(0, len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            k = (node, next_node)
            w = g.edges[k][weight]
            v += w
        return v

    def on_pick_up_passenger(self, vehicle: Vehicle, passenger: dict, timestamp: pd.Timestamp):
        self.logger.debug(
            "on_pick_up_passenger-> vehicle:{0} passenger:{1} timestamp:{2}".format(vehicle.id, passenger["id"],
                                                                                    timestamp))

    def on_drop_off_passenger(self, vehicle: Vehicle, passenger: dict, timestamp: pd.Timestamp):
        self.logger.debug(
            "on_drop_off_passenger-> vehicle:{0} passenger:{1} timestamp:{2}".format(vehicle.id, passenger["id"],
                                                                                     timestamp))

    def on_vehicle_location_changed(self, vehicle: Vehicle, timestamp: pd.Timestamp, cur_location):
        self.logger.debug(
            "on_vehicle_location_changed-> vehicle:{0} timestamp:{1} cur_location:{2}".format(vehicle.id, timestamp,
                                                                                              cur_location))

    def on_vehicle_arrive_road_node(self, vehicle: Vehicle, timestamp: pd.Timestamp, cur_node):
        self.logger.debug(
            "on_vehicle_arrive_road_node-> vehicle:{0} timestamp:{1} cur_node:{2}".format(vehicle.id, timestamp,
                                                                                          cur_node))

    def on_vehicle_empty(self, vehicle, timestamp):
        self.logger.debug("on_vehicle_empty-> vehicle:{0} timestamp:{1}".format(vehicle.id, timestamp))

    def on_vehicle_fully(self, vehicle, timestamp):
        self.logger.debug("on_vehicle_fully-> vehicle:{0} timestamp:{1}".format(vehicle.id, timestamp))

class SingleDispatch(SharingStrategy):

    def __init__(self, max_waiting_time=300, additional_time=300):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):

        self.assign_vehicles(req, timestamp, engine)

    def on_pick_up_passenger(self, vehicle, passenger, timestamp):
        pass

    def assign_vehicles(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())
        last_delta = self.parameters["max_waiting_time"] - duration
        rid = request["id"]
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)

        for value in vhs:
            vid, paths = value
            cost_path = paths[0]
            cost, path = cost_path
            #            cost=cost
            vh: Vehicle = engine.vehicle_manager.vehicles[vid]
            if vh.carry_num == 0 and vh.waiting_num == 0:
                od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(request["source"],
                                                                                    request["target"], timestamp)
                if od_path is not None:
                    #                    r=engine.vehicle_manager.shortest_travel_path_cost(vh.cur_node,o,timestamp)
                    schedule = [(rid, o), (rid, request["target"])]

                    engine.assign_vehicle_to_passenger(request, vh, timestamp)
                    engine.update_vehicle_schedule(vh, timestamp, schedule)
                    success = True
                    break
        return success

class NearestSharing(SharingStrategy):

    def __init__(self, max_waiting_time=600, additional_time=600):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)

        self.waiting_requests = []  # 请求队列一个graph
        self.waiting_requests_batch = []  # 请求队列的一个batch

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):

        #        source,target=req["source"],req["target"]
        #
        #        tid=self.simulator_engine.get_time_id(req["timestamp"])
        #        w="travel_time_"+tid
        #        cost,path=nx.single_source_dijkstra(self.simulator_engine.roads,source,target,weight=w)
        #        p=Passenger(req["id"],o_x=req["o_x"],o_y=req["o_y"],d_x=req["d_x"],d_y=req["d_y"],
        #                    target=req["target"],source=req["source"],timestamp=req["timestamp"],
        #                    cost=cost,path=path)
        self.dispatch(req, timestamp, engine)

    def dispatch(self, req: dict, timestamp: pd.Timestamp, engine: SimulatorEngine):
        self.assign_vehicles(req, timestamp, engine)

    def assign_vehicles(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())

        last_delta = self.parameters["max_waiting_time"] - duration
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        d = request["target"]

        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)
        od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(o, d, timestamp)
        if od_path is None:
            return False

        for value in vhs:
            vid, paths = value
            cost_path = paths[0]
            cost, path = cost_path

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]

            if vh.avail_capacity > 0:
                is_assign = self.assigin_with_passengers(vh, request, timestamp, cost, path, last_delta, engine)
                if is_assign:
                    success = True
                    break
            break
            # continue
        return success

    def get_edge_path_cost(self, g, source, target, timestamp, engine: SimulatorEngine):
        e = (source, target)
        if e not in g.edges:
            path, cost = engine.vehicle_manager.shortest_travel_path_cost(source, target, timestamp)
            g.add_edge(e[0], e[1], path=list(path), cost=cost)

        else:
            path = g.edges[e]["path"]
            cost = g.edges[e]["cost"]
        return path, cost

    def on_pick_up_passenger(self, vehicle: Vehicle, passenger, timestamp: pd.Timestamp):
        m = self.parameters.get("additional_time", 300)

    #        vehicle.targets_graph.nodes[passenger["target"]]["lastest_time"]=timestamp+pd.Timedelta(passenger["shortest_cost"]+m,unit='s')
    #
    @staticmethod
    def digraph_weight_of_path(g: nx.DiGraph, path, weight: str):

        v = 0
        for i in range(0, len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            k = (node, next_node)
            w = g.edges[k][weight]
            v += w
        return v

    def assigin_with_passengers(self, vehicle: Vehicle, request, timestamp: pd.Timestamp, cost, path, last_delta,
                                engine: SimulatorEngine):
        o = request["source"]
        d = request["target"]
        od_cost = request["shortest_cost"]
        #        od_path=request["shortest_path"]
        rid = request["id"]

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)

        o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
        d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
        time_constraint[(rid, o, 0)] = o_ts
        time_constraint[(rid, d, 1)] = d_ts

        passengers = vehicle.passengers.copy()
        passengers.update(vehicle.waiting_passengers)
        passengers[rid] = request

        schedule = list(vehicle.schedule)
        pick_up_times = {pid: p["pick_up_time"] for pid, p in vehicle.passengers.items()}
        #        targets.insert(0,start)
        tsize = len(schedule)
        schedules = []
        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                constraint = time_constraint.copy()
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k

                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > constraint[nex_k]:
                        valid = False
                        break
                    r = passengers[next_id]
                    if ntype == 0:
                        pick_up_times[next_id] = ts
                        constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi, unit='s')
                    elif ntype == 1:

                        scost = r["shortest_cost"]
                        delta_cost = (ts - pick_up_times[next_id]).total_seconds() - scost
                        costs[next_id] = (scost, delta_cost)

                    cur_node = next_node
                if valid is True:
                    schedules.append((acost, sc))

        if len(schedules) > 0:
            #            ws=[(self.digraph_weight_of_path(g,sc,"cost"),sc) for tcost,sc in schedules]
            ws = sorted(schedules, key=lambda x: x[0])
            s = ws[0][1]
            engine.assign_vehicle_to_passenger(request, vehicle, timestamp)
            engine.update_vehicle_schedule(vehicle, timestamp, s)
            return True
        else:
            return False

class APART(SharingStrategy):

    def __init__(self, max_waiting_time=500, additional_time=500):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)

    #        self.waiting_requests=dequ
    #        self.unsuccess_requests=[]

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):

        self.dispatch(req, timestamp, engine)

    def dispatch(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())

        last_delta = self.parameters["max_waiting_time"] - duration
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        d = request["target"]

        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)
        od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(o, d, timestamp)
        if od_path is None:
            return False

        scs = []
        for value in vhs:
            vid, paths = value
            cost_path = paths[0]
            cost, path = cost_path

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]

            if vh.avail_capacity > 0:
                sc, profit = self.assigin_with_passengers(vh, request, timestamp, engine)
                if sc is not None and profit > 0:
                    scs.append((sc, profit, vh))

        if len(scs) > 0:
            scs1 = sorted(scs, key=lambda x: x[1])
            sc, profit, vh = scs1[0]
            engine.assign_vehicle_to_passenger(request, vh, timestamp)
            engine.update_vehicle_schedule(vh, timestamp, sc)

        return success

    def get_edge_path_cost(self, g, source, target, timestamp, engine: SimulatorEngine):
        e = (source, target)
        if e not in g.edges:
            path, cost = engine.vehicle_manager.shortest_travel_path_cost(source, target, timestamp)
            g.add_edge(e[0], e[1], path=list(path), cost=cost)

        else:
            path = g.edges[e]["path"]
            cost = g.edges[e]["cost"]
        return path, cost

    def rider_fare(self, shortest_cost, delta_cost):
        F = 2 * abs((shortest_cost - delta_cost))
        return F

    def assigin_with_passengers(self, vehicle: Vehicle, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        o = request["source"]
        d = request["target"]
        od_cost = request["shortest_cost"]
        #        od_path=request["shortest_path"]
        rid = request["id"]

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)

        o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
        d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
        time_constraint[(rid, o, 0)] = o_ts
        time_constraint[(rid, d, 1)] = d_ts

        passengers = vehicle.passengers.copy()
        passengers.update(vehicle.waiting_passengers)
        passengers[rid] = request

        schedule = list(vehicle.schedule)

        #        targets.insert(0,start)
        tsize = len(schedule)
        schedule1 = None
        profit = -1

        pick_up_times = {pid: p["pick_up_time"] for pid, p in vehicle.passengers.items()}

        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                constraint = time_constraint.copy()
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k

                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > constraint[nex_k]:
                        valid = False
                        break
                    r = passengers[next_id]
                    if ntype == 0:
                        pick_up_times[next_id] = ts
                        constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi, unit='s')
                    elif ntype == 1:

                        scost = r["shortest_cost"]
                        delta_cost = (ts - pick_up_times[next_id]).total_seconds() - scost
                        costs[next_id] = (scost, delta_cost)

                    cur_node = next_node

                if valid is True:
                    fare = 0
                    for pid, v in costs.items():
                        sp, d = v
                        f = self.rider_fare(sp, d)
                        fare += f

                    pf = fare - acost * 0.9
                    if pf > profit:
                        profit = pf
                        schedule1 = sc

        return schedule1, profit

class BatchModel(SharingStrategy):

    def __init__(self, max_waiting_time=300, additional_time=300):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)

    #        self.waiting_requests=dequ
    #        self.unsuccess_requests=[]

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):

        self.dispatch(req, timestamp, engine)

    def dispatch(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())

        last_delta = self.parameters["max_waiting_time"] - duration
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        d = request["target"]

        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)
        od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(o, d, timestamp)
        if od_path is None:
            return False

        scs = []
        for value in vhs:
            vid, paths = value
            cost_path = paths[0]
            cost, path = cost_path

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]

            if vh.avail_capacity > 0:
                sc, profit = self.assigin_with_passengers(vh, request, timestamp, engine)
                if sc is not None and profit > 0:
                    scs.append((sc, profit, vh))

        if len(scs) > 0:
            scs1 = sorted(scs, key=lambda x: x[1])
            sc, profit, vh = scs1[0]
            engine.assign_vehicle_to_passenger(request, vh, timestamp)
            engine.update_vehicle_schedule(vh, timestamp, sc)

        return success

    def get_edge_path_cost(self, g, source, target, timestamp, engine: SimulatorEngine):
        e = (source, target)
        if e not in g.edges:
            path, cost = engine.vehicle_manager.shortest_travel_path_cost(source, target, timestamp)
            g.add_edge(e[0], e[1], path=list(path), cost=cost)

        else:
            path = g.edges[e]["path"]
            cost = g.edges[e]["cost"]
        return path, cost

    def rider_fare(self, shortest_cost, delta_cost):
        F = 2 * abs((shortest_cost - delta_cost))
        return F

    def assigin_with_passengers(self, vehicle: Vehicle, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        o = request["source"]
        d = request["target"]
        od_cost = request["shortest_cost"]
        #        od_path=request["shortest_path"]
        rid = request["id"]

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)

        o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
        d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
        time_constraint[(rid, o, 0)] = o_ts
        time_constraint[(rid, d, 1)] = d_ts

        passengers = vehicle.passengers.copy()
        passengers.update(vehicle.waiting_passengers)
        passengers[rid] = request

        schedule = list(vehicle.schedule)

        #        targets.insert(0,start)
        tsize = len(schedule)
        schedule1 = None
        profit = -1

        pick_up_times = {pid: p["pick_up_time"] for pid, p in vehicle.passengers.items()}

        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k

                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > time_constraint[nex_k]:
                        valid = False
                        break
                    r = passengers[next_id]
                    if ntype == 0:
                        pick_up_times[next_id] = ts
                        time_constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi,
                                                                                       unit='s')
                    elif ntype == 1:

                        scost = r["shortest_cost"]
                        delta_cost = (ts - pick_up_times[next_id]).total_seconds() - scost
                        costs[next_id] = (scost, delta_cost)

                    cur_node = next_node

                if valid is True:
                    fare = 0
                    for pid, v in costs.items():
                        sp, d = v
                        f = self.rider_fare(sp, d)
                        fare += f

                    pf = fare - acost * 0.9
                    if pf > profit:
                        profit = pf
                        schedule1 = sc

        return schedule1, profit

class Record:
    def __init__(self, taxi='', passengers=[], value=0, schedule=[]):
        self.taxi = taxi
        self.passengers = passengers
        self.neighbors = []
        self.value = value
        self.name = (self.taxi, str([p["id"] for p in self.passengers]))
        self.schedule = schedule

class Graph:
    def __init__(self, all_records):
        self.all_records = all_records
        self.result = []
        self.remaining_records = []

    def setup(self):
        for i in range(0, len(self.all_records)):
            current = self.all_records[i]
            for j in range(i + 1, len(self.all_records)):
                other = self.all_records[j]
                if current.taxi == other.taxi or len(set([p["id"] for p in current.passengers]).intersection(
                        set([p["id"] for p in other.passengers]))) > 0:
                    current.neighbors.append(other)
                    other.neighbors.append(current)

    def localSearch(self):
        self.outerSearch()
        self.innerSearch()

    def outerSearch(self):
        while True:
            total_value = self.getSumValue(self.result)
            iterations = []
            iterations.extend(self.remaining_records)
            for record in iterations:
                inside_neighbors = [neighbor for neighbor in record.neighbors if neighbor in self.result]
                if len(inside_neighbors) > 0 and self.getSumValue(inside_neighbors) < record.value:
                    self.result = [record for record in self.result if record not in inside_neighbors]
                    self.result.append(record)
                    self.remaining_records.remove(record)
                    self.remaining_records.extend(inside_neighbors)
                elif len(inside_neighbors) == 0:
                    self.result.append(record)
                    self.remaining_records.remove(record)
            if total_value >= self.getSumValue(self.result):
                break

    def getSumValue(self, records):
        return sum(record.value for record in records)

    def getResultValue(self):
        return self.getSumValue(self.result)

    def innerSearch(self):
        while True:
            total_value = self.getSumValue(self.result)
            iterations = []
            iterations.extend(self.result)
            for record in iterations:
                only_neighbors = [neighbor for neighbor in record.neighbors if len(neighbor.neighbors) == 1]
                if len(only_neighbors) > 0 and self.getSumValue(only_neighbors) > record.value:
                    self.result.remove(record)
                    self.result.extend(only_neighbors)
                    self.remaining_records.append(record)
                    self.remaining_records = [record for record in self.remaining_records if
                                              record not in only_neighbors]
            if total_value >= self.getSumValue(self.result):
                break

    def initResult(self):
        for record in self.all_records:
            if all(record not in r.neighbors for r in self.result):
                self.result.append(record)
            else:
                self.remaining_records.append(record)

    def perturbResult(self):
        for i in range(math.ceil(len(self.result) * 5 / 100)):
            del_index = random.randint(0, len(self.result) - 1)
            pert_record = self.result[del_index]
            self.result.remove(pert_record)
            self.remaining_records.append(pert_record)

class PDTL(SharingStrategy):

    def __init__(self, regions: dict, demands: pd.DataFrame, batch=pd.Timedelta(60*5, unit='s'), time_unit=60,
                 max_waiting_time=600, additional_time=600):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)
        self.waiting_requests = deque()
        self.waiting_requests_batch = []  # 请求队列的一个batch
        self.unsuccess_requests = []
        self.batch_timestamp = None
        self.batch = batch if isinstance(batch, pd.Timedelta) else pd.Timedelta(60, unit='s')
        self.regions = regions
        self.region_nodes = None
        self.nodes_region = None
        self.demands = demands
        self.time_unit = time_unit
        self.time_interval_num = int(24 * 60 / time_unit)
        self.region_vehicle_num = pd.DataFrame(data=0, index=demands.index, columns=[rid for rid in regions.keys()])
        self.region_vh_init = False

    def clear(self):
        self.region_vehicle_num = pd.DataFrame(data=0, index=self.region_vehicle_num.index,
                                               columns=self.region_vehicle_num.columns)
        self.waiting_requests.clear()
        self.batch_timestamp = None
        self.region_vh_init = False

    def init_region_nodes(self, vehicle_manager: VehicleManager, read_file=True):
        self.region_nodes = {}
        self.nodes_region = {}
        filedir = 'D:/cjx_code/fleet_rs_v_2.0/data/sf_init_region_nodes/'
        # filedir = 'D:/cjx_code/fleet_rs_v_2.0/data/wh_init_region_nodes/'
        if not read_file:
            for rid, region in self.regions.items():  # 10205
                n = vehicle_manager.nodes_gdf.within(region)
                nodes = vehicle_manager.nodes_gdf[n]
                self.region_nodes[rid] = nodes
                for nid in nodes.index:
                    self.nodes_region[nid] = rid
            with open(filedir + 'region_nodes.pkl', 'wb') as f:
                pickle.dump(self.region_nodes, f, pickle.HIGHEST_PROTOCOL)
            with open(filedir + 'nodes_region.pkl', 'wb') as f:
                pickle.dump(self.nodes_region, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(filedir + 'region_nodes.pkl', 'rb') as f:
                self.region_nodes = pickle.load(f)
            with open(filedir + 'nodes_region.pkl', 'rb') as f:
                self.nodes_region = pickle.load(f)

        print("Initialize region nodes completed!")

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):
        if self.region_nodes is None:
            self.init_region_nodes(engine.vehicle_manager)

        if self.batch_timestamp is None:
            self.batch_timestamp = timestamp + self.batch

        if timestamp >= self.batch_timestamp:  #分配当前batch的所有请求 然后进入下一个batch
            self.batch_timestamp += self.batch
            reqs = tuple(self.waiting_requests)
            self.waiting_requests.clear()
            self.dispatch(reqs, timestamp, engine)
        else:
            self.waiting_requests.append(req)  #将请求加入等待队列


    def query_reachable_vehicles(self, requests: list, timestamp, engine: SimulatorEngine):
        vh_requests = {}
        for req in requests:
            rid = req["id"]
            w = self.parameters["max_waiting_time"]
            d: pd.Timedelta = timestamp - req["timestamp"]
            w = w - d.total_seconds()
            vhs: list = engine.vehicle_manager.search_reachable_vehicles(req["source"], w, timestamp)
            for vid, paths in vhs:
                if vid not in vh_requests:
                    vh_requests[vid] = []
                vh_requests[vid].append(req)
        return vh_requests

    def schedule_insert(self, start, request: dict, timestamp, schedule: list, time_constraint: dict,
                        vehicle_passengers: dict, engine: SimulatorEngine):
        schedules = []
        tsize = len(schedule)
        rid = request["id"]
        o = request["source"]
        d = request["target"]

        addi = self.parameters["additional_time"]
        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                constraint = time_constraint.copy()
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k
                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        logging.debug(f"n_cost is None valid is False")
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > constraint[nex_k]:
                        valid = False
                        logging.debug(f"schedule_insert-> not satisify time constraint")
                        break
                    r = vehicle_passengers[next_id]
                    if ntype == 0:
                        #                        pick_up_times[next_id]=ts
                        constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi, unit='s')
                    elif ntype == 1:
                        pass
                    #                        scost=r["shortest_cost"]
                    #                        delta_cost=(ts-pick_up_times[next_id]).total_seconds()-scost
                    #                        costs[next_id]=(scost,delta_cost)
                    #
                    cur_node = next_node

                if valid is True:
                    schedules.append(sc)
        return schedules

    def mapping_to_region(self, node_path):
        node_region = [self.nodes_region[n] for n in node_path]
        return node_region

    def calculate_region_vehicle_num(self, timestamp: pd.Timestamp, engine: SimulatorEngine):

        rvn = self.region_vehicle_num.copy()
        if any(rvn.index.contains(timestamp)):

            for vid, vh in engine.vehicle_manager.vehicles.items():
                v: Vehicle = vh
                cur = v.cur_node
                path = v.path
                ts = timestamp
                for i in range(1, len(path)):
                    n1 = path[i - 1]
                    n2 = path[i]
                    pre = self.nodes_region[n1]
                    cur = self.nodes_region[n2]

                    #                    e=engine.vehicle_manager.roads.edges[(n1,n2,0)]
                    #                    tid=engine.get_time_id(ts,unit=self.time_unit)
                    #                    cost=e[f"travel_time_{tid}"]
                    if pre != cur:
                        if any(rvn.index.contains(ts)) and cur in rvn.columns:
                            ri = rvn.index.get_loc(ts)
                            ci = rvn.columns.get_loc(cur)
                            rvn.iat[ri, ci] += 1
        return rvn

    def predicted_demands_num(self, region: typing.Hashable, timestamp: pd.Timestamp):
        if any(self.demands.index.contains(timestamp)) and region in self.demands.columns:
            r = self.demands.index.get_loc(timestamp)
            c = self.demands.columns.get_loc(region)
            num = self.demands.iat[r, c]
        else:
            num = 0
        return num

    def calculate_schedule_probability(self, vehicle_id, start_node, carry_num, schedule, requests,
                                       regions_vh_num: dict, timestamp: pd.Timestamp, engine):

        #        node_path=[n[1] for n in schedule]
        #        node_path.insert(0,start_node)


        proba = {}

        for request in requests:
            rid = request["id"]
            o = request["source"]
            d = request["target"]
            r_start = (rid, o, 0)
            r_end = (rid, d, 1)
            rp = defaultdict(list)
            is_calculate = False
            ts = timestamp
            cur = start_node
            cur_carry_num = carry_num
            #            cur_region=self.nodes_region[cur]
            for k in schedule:
                nid, node, ntype = k
                region = self.nodes_region[node]
                n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur, node, ts)
                ts += pd.Timedelta(n_cost, unit='s')

                if ntype == 0:
                    cur_carry_num += 1
                else:
                    cur_carry_num -= 1
                if k == r_start:
                    is_calculate = True
                if k == r_end:
                    is_calculate = False

                if is_calculate:
                    self.logger.debug("calculate_schedule_probability-> cur_carry_num:{0}".format(cur_carry_num))
                    p = 0
                    if cur_carry_num <= 0:
                        print("Error cur_carry_num:{0}".format(cur_carry_num))
                    elif cur_carry_num == 1:
                        r_num = self.predicted_demands_num(region, ts)
                        if any(regions_vh_num.index.contains(ts)) and region in regions_vh_num.columns:
                            r = regions_vh_num.index.get_loc(ts)
                            c = regions_vh_num.columns.get_loc(region)
                            vh_num = regions_vh_num.iat[r, c]
                            p = r_num / vh_num if vh_num > 0 else 0

                    elif cur_carry_num > 1:
                        p = 1
                    rp[region].append(p)

            # end for
            req_proba = {k: sum(v) / len(v) for k, v in rp.items() if len(v) > 0}
            l = len(req_proba)
            req_proba = sum(req_proba.values()) / l if l > 0 else 0
            proba[rid] = req_proba
        return proba

    def find_max_probability_of_combination(self, vehicle: Vehicle, requests: list, timestamp: pd.Timestamp,
                                            regions_vh_num: pd.DataFrame, engine):

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)
        passengers = vehicle.passengers.copy()
        #        pick_up_times={pid:p["pick_up_time"] for pid,p in vehicle.passengers.items()}
        passengers.update(vehicle.waiting_passengers)


        schedule = list(vehicle.schedule)
        pending = list(requests).copy()
        schedules = [schedule]
        logging.debug(f"vh:{vehicle.id} carry:{vehicle.carry_num} reqs:{len(requests)}")
        while len(pending) > 0:
            request = pending.pop()
            o = request["source"]
            d = request["target"]
            od_cost = request["shortest_cost"]
            #               od_path=request["shortest_path"]
            rid = request["id"]
            passengers[rid] = request
            o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
            d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
            time_constraint[(rid, o, 0)] = o_ts
            time_constraint[(rid, d, 1)] = d_ts
            scs = []
            success = True
            while len(schedules) > 0:
                sc = schedules.pop()  
                sc_list = self.schedule_insert(start, request, timestamp, sc, time_constraint, passengers, engine)
                if len(sc_list) == 0:
                    success = False
                    break
                else:
                    scs.extend(sc_list)
            if success is False:
                break
            schedules.extend(scs)

        logging.debug(f"vh:{vehicle.id} req:{len(requests)} schedules:{len(schedules)} ")
        # calculate probability of each schedule
        probability = -1
        s = None
        for sc in schedules:
            proba = self.calculate_schedule_probability(vehicle.id, start, vehicle.carry_num, sc, requests,
                                                        regions_vh_num, timestamp, engine)
            p = sum(proba.values())
            if p > probability:
                probability = p
                s = sc
        return probability, s

    def init_region_vehicle_num(self, timestamp: pd.Timestamp, engine: SimulatorEngine):
        for vid, vh in engine.vehicle_manager.vehicles.items():
            cur_reg = self.nodes_region[vh.cur_node]
            if cur_reg in self.region_vehicle_num.columns and any(self.region_vehicle_num.index.contains(timestamp)):
                r = self.region_vehicle_num.index.get_loc(timestamp)
                c = self.region_vehicle_num.columns.get_loc(cur_reg)
                self.region_vehicle_num.iat[r, c] += 1

    def schedule_combinations(self, vehicle: Vehicle, reachable_requests: list, timestamp, engine: SimulatorEngine):

        if self.region_vh_init is False:
            self.init_region_vehicle_num(timestamp, engine)
            self.region_vh_init = True
        regions_vh_num = self.calculate_region_vehicle_num(timestamp, engine)
        avail = vehicle.avail_capacity
        combinations = []
        schedules = []

        for insert_num in range(1, avail + 1):
            combine = itertools.combinations(reachable_requests, insert_num)
            combinations.extend(combine)
        logging.debug(f"vh:{vehicle.id} combinations:{len(combinations)}")
        #        logging.debug(f"combinations:{combinations}")
        for combine in combinations:

            p, sc = self.find_max_probability_of_combination(vehicle, combine, timestamp, regions_vh_num, engine)
            if sc is not None and p != -1:
                schedules.append((vehicle.id, combine, p, sc))
        logging.debug(f"vh:{vehicle.id} schedules:{len(schedules)}")
        return schedules

    def on_vehicle_arrive_road_node(self, vehicle: Vehicle, timestamp, cur_node):
        super().on_vehicle_arrive_road_node(vehicle, timestamp, cur_node)
        cur_region = self.nodes_region[cur_node]
        pre_region = self.nodes_region[vehicle.cur_node]
        if cur_region != pre_region and any(self.region_vehicle_num.index.contains(timestamp)):
            r = self.region_vehicle_num.index.get_loc(timestamp)
            c = self.region_vehicle_num.columns.get_loc(cur_region)
            self.region_vehicle_num.iat[r, c] += 1

    def dispatch(self, requests: list, timestamp: pd.Timestamp, engine: SimulatorEngine):

        vh_requests = self.query_reachable_vehicles(requests, timestamp, engine) #为每一个request找到车辆 一个request可能有多个车 keys为车的编号
        
        vh_requests_combination = []
        #        logging.debug(f"requests:{requests}")
        #        logging.debug(f"vh_requests:{vh_requests_combination}")
        logging.debug(f"PDTL dispatching->request_num:{len(requests)} find_vhs_num:{len(vh_requests.keys())}")
        for vid, reqs in vh_requests.items():
            vh = engine.vehicle_manager.vehicles[vid]
            vh_combine = self.schedule_combinations(vh, reqs, timestamp, engine)
            vh_requests_combination.extend(vh_combine)

        logging.debug(f"combinations num:{len(vh_requests_combination)}")
        # to_do
        # a function to return a sub set of vh_requests_combination for comfirm which requests a vehicle to responding
        records = []
        for v in vh_requests_combination:
            vid, pgers, prob, sc = v
            r = Record(vid, pgers, prob, sc)
            records.append(r)
        logging.debug(f"records num:{len(records)}")

        results: typing.Sequence[Record] = self.iterate_local_search(records)
        logging.debug(f"records results_1114_1120 num:{len(results)}")
        for record in results:
            try:
                vh: Vehicle = engine.vehicle_manager.vehicles[record.taxi]
                sc = record.schedule
                logging.debug(f"assign->vehicle:{vh.id} passenger:{len(record.passengers)}")
                for psg in record.passengers:
                    engine.assign_vehicle_to_passenger(psg, vh, timestamp)
                engine.update_vehicle_schedule(vh, timestamp, sc)
            except Exception as e:
                print(e,"dispatch1")

    def iterate_local_search(self, records: typing.Sequence[Record], iter_num: int = 5) -> typing.Sequence[Record]:
        graph = Graph(records)
        graph.setup()

        for i in range(0, iter_num):
            if i == 0:
                graph.initResult()
                logging.debug("init:" + str(graph.getResultValue()))
                for item in graph.result:
                    logging.debug(item.name)
            else:
                graph.perturbResult()
            graph.localSearch()

        #        logging.debug("end:" + str(self.graph.getResultValue()))
        #        for item in self.graph.result:
        #            logging.debug(item.name)

        return graph.result

class AM(SharingStrategy):

    def __init__(self, regions: dict, demands: pd.DataFrame, name, batch=pd.Timedelta(60*5, unit='s'), whole_batch=1, time_unit=60,
                 max_waiting_time=600, additional_time=600):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)    # 300
        self.set_parameter("additional_time", additional_time)      # 300
        self.waiting_requests = [] # 请求队列一个graph
        self.waiting_requests_batch=[] # 请求队列的一个batch
        self.candidate_vehicles_batch=[] # 一个batch的候选车辆
        self.waiting_requests_id_batch = []
        self.assigned_request = []
        self.unsuccess_requests = []
        self.batch_timestamp = None
        self.whole_batch_timestamp=None
        self.batch = batch if isinstance(batch, pd.Timedelta) else pd.Timedelta(60, unit='s')   # 5 mins
        self.whole_batch = self.batch * whole_batch
        self.regions = regions
        self.region_nodes = None
        self.nodes_region = None
        self.demands = demands
        self.time_unit = time_unit  # 60
        self.time_interval_num = int(24 * 60 / time_unit)   # 24
        self.region_vehicle_num = pd.DataFrame(data=0, index=demands.index, columns=[rid for rid in regions.keys()])
        self.region_vh_init = False
        self.dataset_name = name

        self.init_run_param()

    def init_run_param(self):
        self.opts = get_options()

        self.opts.run_name = "{}_{}".format(self.dataset_name, time.strftime("%Y%m%dT%H%M%S"))
        self.opts.save_dir = os.path.join(
            self.opts.output_dir,  # 'ridesharing/outputs'
            self.opts.run_name   # sf_ridesharing
        )

        # 训练 or 测试
        # self.opts.load_path = 'ridesharing/pretrained/Wuhan/WaitingTime300_AdditionTime600_Vehicles500/epoch-95.pt'
        self.opts.load_path = 'ridesharing/pretrained/SanFrancisco/max_waiting_time_600/epoch-407.pt'
        # self.opts.load_path = 'ridedispatching_model/wh/max_waiting_time_500/train/sf_ridesharing_20230226T115341/epoch-959.pt'
        self.opts.eval_only = True

        # Optionally configure tensorboard
        tb_logger = None
        if not self.opts.no_tensorboard:
            self.tb_logger = TbLogger(
                os.path.join(self.opts.log_dir, self.opts.run_name))
                # opts.log_dir = 'ridesharing/logs'

        # save model to outputs dir
        os.makedirs(self.opts.save_dir) # opts.save_dir

        # Save arguments so exact configuration can always be found
        with open(os.path.join(self.opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=True)

        # Set the random seed
        torch.manual_seed(self.opts.seed)

        # Set the device
        self.opts.device = torch.device("cuda:0" if self.opts.use_cuda else "cpu")

        # Figure out what's the problem
        # problem = load_problem(opts.problem)
        self.problem = RideSharing()

        # Load data from load_path
        # if u have run the model before, u can continue from resume path
        load_data = {}
        assert self.opts.load_path is None or self.opts.resume is None, "Only one of load path and resume can be given"
        load_path = self.opts.load_path if self.opts.load_path is not None else self.opts.resume
        if load_path is not None:
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)

        # Initialize model
        model_class = {
            'attention': AttentionModel,
            'pointer': PointerNetwork
        }.get(self.opts.model, None)
        assert model_class is not None, "Unknown model: {}".format(model_class)
        self.model = model_class(
            self.opts.embedding_dim,
            self.opts.hidden_dim,
            self.opts.obj,
            self.problem,
            n_encode_layers=self.opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=self.opts.normalization,
            tanh_clipping=self.opts.tanh_clipping,
            checkpoint_encoder=self.opts.checkpoint_encoder,
            shrink_size=self.opts.shrink_size
        ).to(self.opts.device)

        # multi-gpu
        if self.opts.use_cuda and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Overwrite model parameters by parameters to load
        self.model_ = get_inner_model(self.model)
        self.model_.load_state_dict({**self.model_.state_dict(), **load_data.get('model', {})})

        # Initialize baseline
        if self.opts.baseline == 'exponential':
            self.baseline = ExponentialBaseline(self.opts.exp_beta)
        elif self.opts.baseline == 'critic' or self.opts.baseline == 'critic_lstm':
            assert self.problem.NAME == 'tsp', "Critic only supported for TSP"
            self.baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(
                        2,
                        self.opts.embedding_dim,
                        self.opts.hidden_dim,
                        self.opts.n_encode_layers,
                        self.opts.tanh_clipping
                    )
                    if self.opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        self.opts.embedding_dim,
                        self.opts.hidden_dim,
                        self.opts.n_encode_layers,
                        self.opts.normalization
                    )
                ).to(self.opts.device)
            )
        elif self.opts.baseline == 'rollout':
            self.baseline = RolloutBaseline(self.model, self.problem, self.opts)
        else:
            assert self.opts.baseline is None, "Unknown baseline: {}".format(self.opts.baseline)
            self.baseline = NoBaseline()

        if self.opts.bl_warmup_epochs > 0:
            self.baseline = WarmupBaseline(self.baseline, self.opts.bl_warmup_epochs, warmup_exp_beta=self.opts.exp_beta)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            self.baseline.load_state_dict(load_data['baseline'])

        # Initialize optimizer
        self.optimizer = optim.Adam(
            [{'params': self.model.parameters(), 'lr': self.opts.lr_model}]
            + (
                [{'params': self.baseline.get_learnable_parameters(), 'lr': self.opts.lr_critic}]
                if len(self.baseline.get_learnable_parameters()) > 0
                else []
            )
        )

        # Load optimizer state from trained model
        if 'optimizer' in load_data:
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.opts.lr_decay ** epoch)

        if load_path is not None:
            self.epoch_id = self.opts.epoch_start
        else:
            self.epoch_id = 0

        self.batch_id = 0

    def clear(self):
        self.region_vehicle_num = pd.DataFrame(data=0, index=self.region_vehicle_num.index,
                                               columns=self.region_vehicle_num.columns)
        self.waiting_requests.clear()
        self.batch_timestamp = None
        self.region_vh_init = False

    def init_region_nodes(self, vehicle_manager: VehicleManager, read_file=True):
        self.region_nodes = {}
        self.nodes_region = {}
        filedir = 'D:/cjx_code/fleet_rs_v_2.0/data/' + self.dataset_name[:2] + '_init_region_nodes/'
        if not read_file:
            for rid, region in self.regions.items():  # 10205
                n = vehicle_manager.nodes_gdf.within(region)
                nodes = vehicle_manager.nodes_gdf[n]
                self.region_nodes[rid] = nodes
                for nid in nodes.index:
                    self.nodes_region[nid] = rid
            with open(filedir + 'region_nodes.pkl', 'wb') as f:
                pickle.dump(self.region_nodes, f, pickle.HIGHEST_PROTOCOL)
            with open(filedir + 'nodes_region.pkl', 'wb') as f:
                pickle.dump(self.nodes_region, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(filedir + 'region_nodes.pkl', 'rb') as f:
                self.region_nodes = pickle.load(f)
            with open(filedir + 'nodes_region.pkl', 'rb') as f:
                self.nodes_region = pickle.load(f)

        # print("Initialize region nodes completed!")

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):
        if self.region_nodes is None:
            self.init_region_nodes(engine.vehicle_manager)

        if self.batch_timestamp is None:
            self.batch_timestamp = timestamp + self.batch # self.batch 5*60 = self.whole_batch

        if self.whole_batch_timestamp is None:
            self.whole_batch_timestamp=timestamp+self.whole_batch

        self.waiting_requests.append(req)  # 将请求加入等待队列
        if timestamp >= self.batch_timestamp:  # 分配当前batch的所有请求 然后进入下一个batch
            self.batch_timestamp += self.batch
            reqs = tuple(self.waiting_requests)
            self.waiting_requests_batch.append(reqs)
            self.waiting_requests.clear()
            if timestamp>=self.whole_batch_timestamp:
                self.whole_batch_timestamp+=self.whole_batch # 2008-5-20 00:00:00

                input = self.requests_to_am_one_batch(timestamp, self.waiting_requests_batch, engine)   # 模型需要的数据格式

                if not self.opts.eval_only:
                    routes = self.train(input, engine)
                else:
                    routes = self.validate(input, engine)

                self.dispatch(self.waiting_requests_batch, timestamp, engine, routes)
                # print("{} is finished".format(timestamp))
                self.waiting_requests_batch.clear()
                self.candidate_vehicles_batch.clear()
                self.waiting_requests_id_batch.clear()
            else:
                pass
        else:
            pass

    def requests_to_am_one_batch(self, timestamp, reqs:list, engine):
        data_am = {'loc': [], 'veh': [], 'to_visit': [], 'demand': [], 'capacity': [], 'time': []}
        graph = reqs[0]

        new_graph_size = len(graph) # 新的请求数量
        loc = []
        vehicles_loc = []
        demand = []
        capacity = []
        time = []

        for req in self.waiting_requests_batch[0]:
            self.waiting_requests_id_batch.append(req['id'])

        vh_requests = self.query_reachable_vehicles(graph, graph[-1]["timestamp"], engine)
        vehicles = list(set(vh_requests.keys()))
        self.candidate_vehicles_batch.extend(vehicles)  # 存储所选车辆的编号

        for v_id in vehicles:
            vh: Vehicle = engine.vehicle_manager.vehicles[v_id]
            if len(vh.passengers) > 0 or len(vh.waiting_passengers) > 0:
                for req in vh.passengers.values():
                    graph = graph + (req, )
                    self.waiting_requests_id_batch.append(req['id'])
                for wait_req in vh.waiting_passengers.values():
                    graph = graph + (wait_req, )
                    self.waiting_requests_id_batch.append(wait_req['id'])

        total_graph_size = len(graph)
        self.waiting_requests_batch[0] = graph

        to_visit_ = torch.ones(len(vehicles), total_graph_size * 2, dtype=torch.int64)

        for veh, requests in vh_requests.items():
            veh_index = vehicles.index(veh)
            for req in requests:
                req_index = graph.index(req)
                to_visit_[veh_index, req_index] = 0

        for vh in vehicles:
            capacity.append(engine.vehicle_manager.vehicles[vh].avail_capacity)
            vehicles_loc.append(list(engine.vehicle_manager.vehicles[vh].location))

        start_timestamp = graph[0]['timestamp']
        for j in range(0, len(graph)):
            loc.append([graph[j]['o_x'], graph[j]['o_y']])
            demand.append(1)
            o_time: pd.Timedelta = timestamp - graph[j]['timestamp']
            time.append(self.parameters["max_waiting_time"] - o_time.total_seconds())

            # o_time: pd.Timedelta = graph[j]['timestamp'] - start_timestamp
            # time.append(self.parameters["max_waiting_time"] + o_time.total_seconds())

        for j in range(0, len(graph)):
            loc.append([graph[j]['d_x'], graph[j]['d_y']])
            demand.append(-demand[j])
            d_time: pd.Timedelta = timestamp - graph[j]['timestamp']
            time.append(self.parameters["additional_time"] + graph[j]['shortest_cost'] - d_time.total_seconds())

            # d_time: pd.Timedelta = graph[j]['timestamp'] - start_timestamp
            # time.append(self.parameters["additional_time"] + graph[j]['shortest_cost'] + d_time.total_seconds())

        data_am['loc'].append(torch.Tensor(loc))
        data_am['veh'].append(torch.Tensor(vehicles_loc))
        data_am['demand'].append(torch.Tensor(demand))
        data_am['capacity'].append(torch.Tensor(capacity))
        data_am['time'].append(torch.Tensor(time))
        data_am['to_visit'].append(to_visit_)

        for key, value in data_am.items():
            data_am[key] = torch.stack(data_am[key], 0)

        return data_am

    def query_reachable_vehicles(self, requests: list, timestamp, engine: SimulatorEngine):
        vh_requests = {}
        for req in requests:
            rid = req["id"]
            w = self.parameters["max_waiting_time"]
            d: pd.Timedelta = timestamp - req["timestamp"]
            w = w - d.total_seconds()
            vhs: list = engine.vehicle_manager.search_reachable_vehicles(req["source"], w, timestamp)
            for vid, paths in vhs:
                if vid not in vh_requests:
                    vh_requests[vid] = []
                vh_requests[vid].append(req)
        return vh_requests

    def init_region_vehicle_num(self, timestamp: pd.Timestamp, engine: SimulatorEngine):
        for vid, vh in engine.vehicle_manager.vehicles.items():
            cur_reg = self.nodes_region[vh.cur_node]
            if cur_reg in self.region_vehicle_num.columns and any(self.region_vehicle_num.index.contains(timestamp)):
                r = self.region_vehicle_num.index.get_loc(timestamp)
                c = self.region_vehicle_num.columns.get_loc(cur_reg)
                self.region_vehicle_num.iat[r, c] += 1

    def on_vehicle_arrive_road_node(self, vehicle: Vehicle, timestamp, cur_node):
        super().on_vehicle_arrive_road_node(vehicle, timestamp, cur_node)
        cur_region = self.nodes_region[cur_node]
        pre_region = self.nodes_region[vehicle.cur_node]
        if cur_region != pre_region and any(self.region_vehicle_num.index.contains(timestamp)):
            r = self.region_vehicle_num.index.get_loc(timestamp)
            c = self.region_vehicle_num.columns.get_loc(cur_region)
            self.region_vehicle_num.iat[r, c] += 1

    def clip_grad_norms(self, param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
        :param optimizer:
        :param max_norm:
        :param gradient_norms_log:
        :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped

    def train(self, input, engine: SimulatorEngine):
        epoch_id, batch_id = self.epoch_id, self.batch_id
        if batch_id==0:
            # print("Start train epoch {}, lr={} for run {}".format(epoch_id, self.optimizer.param_groups[0]['lr'], self.opts.run_name))
            self.step = epoch_id * (self.opts.epoch_size // self.opts.batch_size)
            self.epoch_start_time = time.time()
            self.lr_scheduler.step(epoch_id)
            if not self.opts.no_tensorboard:  # need tensorboard
                self.tb_logger.log_value('learnrate_pg0', self.optimizer.param_groups[0]['lr'], self.step)

            # Put model in train mode!
            self.model.train()
            set_decode_type(self.model, "sampling") # 设置为采样

            torch.cuda.empty_cache()

        # start train batch
        x, bl_val = self.baseline.unwrap_batch(input)  # data, baseline(cost of data)
        x = move_to(x, self.opts.device)
        bl_val = move_to(bl_val, self.opts.device) if bl_val is not None else None

        # Evaluate proposed model, get costs and log probabilities
        cost, log_likelihood, log_veh, routes = self.model(x, engine)

        '''
        print("********************************测试reward的收敛性********************************")
        episodes_start_time = datetime.datetime.now()
        EPISODES = 100000
        for episode in range(0, EPISODES):
            if episode == 20:
                print("debug")
            cost, log_likelihood, log_veh, routes = self.model(x, engine)
            # Evaluate baseline, get baseline loss if any (only for critic)
            bl_val, bl_loss = self.baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

            # Calculate loss
            reinforce_loss = ((cost - bl_val) * (log_likelihood + log_veh)).mean()
            loss = reinforce_loss + bl_loss

            # Perform backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = self.clip_grad_norms(self.optimizer.param_groups, self.opts.max_grad_norm)

            self.optimizer.step()
            log_batch_values(cost, episode, EPISODES, log_likelihood + log_veh, reinforce_loss, self.tb_logger, self.opts)
        episodes_end_time = datetime.datetime.now()
        print("********************************测试reward的收敛性********************************")
        print("测试收敛性花费总时间:", episodes_end_time - episodes_start_time)
        '''
        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = self.baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss
        # reinforce_loss = ((cost - bl_val) * (log_likelihood + log_veh)).mean()
        reinforce_loss = ((cost - bl_val) * (log_likelihood + log_veh)).mean()
        loss = reinforce_loss + bl_loss

        # Perform backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = self.clip_grad_norms(self.optimizer.param_groups, self.opts.max_grad_norm)

        self.optimizer.step()

        # Logging
        if self.step % int(self.opts.log_step) == 0:    # log_step
            log_values(cost, grad_norms, epoch_id, batch_id, self.step,
                       log_likelihood + log_veh, reinforce_loss, bl_loss, self.tb_logger, self.opts)

        # 每个epoch结束
        if (self.batch_id == self.opts.epoch_size // self.opts.batch_size - 1) or (((self.epoch_id + 1) % 24 == 0) and self.batch_id == 10):
            epoch_duration = time.time() - self.epoch_start_time
            print("Finished epoch {}, took {} s".format(self.epoch_id, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

            # save results every checkpoint_epoches, saving memory 先只保存最终模型
            if (self.opts.checkpoint_epochs != 0 and (self.epoch_id + 1) % self.opts.checkpoint_epochs == 0) or self.epoch_id == self.opts.n_epochs - 1:
                print('Saving model and state...')
                torch.save(
                    {
                        'model': get_inner_model(self.model).state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        # rng_state is the state of random generator
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': self.baseline.state_dict()
                    },
                    # save state of runned model in outputs
                    os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch_id))
                )

            # avg_reward = validate(self.model, val_dataset, opts)

            # if not self.opts.no_tensorboard:
            #     self.tb_logger.log_value('val_avg_reward', avg_reward, self.step)

            self.baseline.epoch_callback(self.model, self.epoch_id)

            if ((self.epoch_id + 1) % 24 == 0) and (self.batch_id == 10):
                self.batch_id = self.batch_id + 1

            self.epoch_id += 1

        self.batch_id = (self.batch_id + 1) % (self.opts.epoch_size // self.opts.batch_size)
        self.step += 1

        return routes

    def validate(self, dataset, engine):
        cost, routes = self.rollout(dataset, engine)
        avg_cost = cost.mean()
        print('Validation avg_cost: {}'.format(avg_cost))

        return routes

    def rollout(self, dataset, engine):
        # Put in greedy evaluation mode!
        self.model.set_decode_type("greedy")
        self.model.eval()

        with torch.no_grad():
            cost, _, _, routes = self.model(move_to(dataset, self.opts.device), engine)
        return cost.data.cpu(), routes

    def dispatch(self, requests: list, timestamp: pd.Timestamp, engine: SimulatorEngine, routes: list):
        vehicles = self.candidate_vehicles_batch

        request_size = len(requests[0])
        num_veh = len(vehicles)
        used_vehicles = 0
        for i in range(num_veh):
            if len(routes[0][i]) > 1:
                used_vehicles += 1
                try:
                    vh: Vehicle = engine.vehicle_manager.vehicles[vehicles[i]]
                    schedule = []
                    for psg_id in routes[0][i]:
                        if psg_id >= num_veh and psg_id < (num_veh + request_size):
                            psg = requests[0][psg_id - num_veh]
                            if psg['id'] not in vh.waiting_passengers:
                                engine.assign_vehicle_to_passenger(psg, vh, timestamp)
                                psg['status'] = "assigned"
                            schedule.append((psg['id'], psg['source'], 0))
                        elif psg_id >= (num_veh + request_size) and psg_id < (num_veh + request_size * 2):
                            psg = requests[0][psg_id - (num_veh + request_size)]
                            schedule.append((psg['id'], psg['target'], 1))
                        else:
                            pass
                    engine.update_vehicle_schedule(vh, timestamp, schedule)
                except Exception as e:
                    self.logger.exception(e)

class MTShare(SharingStrategy):

    def __init__(self, max_waiting_time=600, additional_time=600):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)
        # 是am需要的，没有用
        self.waiting_requests = []  # 请求队列一个graph
        self.waiting_requests_batch = []  # 请求队列的一个batch

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):
        self.dispatch(req, timestamp, engine)

    def dispatch(self, req: dict, timestamp: pd.Timestamp, engine: SimulatorEngine):
        self.assign_vehicles(req, timestamp, engine)

    def assign_vehicles(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())

        last_delta = self.parameters["max_waiting_time"] - duration
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        d = request["target"]

        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)
        od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(o, d, timestamp)
        if od_path is None:
            return False

        req_vec = [request['o_x'], request['o_y'], request['d_x'], request['d_y']]
        max_cos = -2
        max_vh = -1

        # 需要进一步筛选路径比较相似的车辆
        for value in vhs:
            vid, _ = value

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]
            if not vh.schedule: # 车辆计划路线为空
                continue
            else:
                if vh.passengers or vh.waiting_passengers:
                    load_req_lon = []
                    load_req_lat = []
                    for served_psg in vh.passengers.values():
                        load_req_lon.append(served_psg['d_x'])
                        load_req_lat.append(served_psg['d_y'])
                    for wait_psg in vh.waiting_passengers.values():
                        load_req_lon.append(wait_psg['d_x'])
                        load_req_lat.append(wait_psg['d_y'])
                    vh_vec = [vh.location[0], vh.location[1], mean(load_req_lon), mean(load_req_lat)]
                    cos_val = self.cosine_similarity(vh_vec, req_vec)
                    # print("余弦相似度是：{}".format(cos_val))
                    if cos_val > max_cos:
                        max_cos = cos_val
                        max_vh = vh.id

        if max_vh != -1:    # 直接派给余弦相似度最大的车辆
            vh: Vehicle = engine.vehicle_manager.vehicles[max_vh]
            if vh.avail_capacity > 0:
                schedule = self.get_vehicle_available_schedules(vh,request,timestamp,engine)
                if schedule is not None:
                    is_assign = self.assign_vehicle_to_passenger(vh,request,timestamp,schedule[1],engine)
                    if is_assign:
                        success = True
                        return success

        #具有最小路径增量成本的车辆
        min_cost_vh = -1
        min_increment_cost = float('inf')
        min_schedule = []
        for value in vhs:
            vid, _ = value

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]
            if vh.avail_capacity > 0:
                increment, schedule = self.get_vehicle_insert_increment(vh, request, timestamp, engine)
                if increment is not None and increment < min_increment_cost:
                    min_cost_vh = vid
                    min_schedule = schedule

        if min_cost_vh != -1:
            vh: Vehicle = engine.vehicle_manager.vehicles[min_cost_vh]
            is_assign = self.assign_vehicle_to_passenger(vh, request, timestamp, min_schedule, engine)
            if is_assign:
                success = True

        return success

    def get_edge_path_cost(self, g, source, target, timestamp, engine: SimulatorEngine):
        e = (source, target)
        if e not in g.edges:
            path, cost = engine.vehicle_manager.shortest_travel_path_cost(source, target, timestamp)
            g.add_edge(e[0], e[1], path=list(path), cost=cost)

        else:
            path = g.edges[e]["path"]
            cost = g.edges[e]["cost"]
        return path, cost

    def on_pick_up_passenger(self, vehicle: Vehicle, passenger, timestamp: pd.Timestamp):
        m = self.parameters.get("additional_time", 300)

    @staticmethod
    def digraph_weight_of_path(g: nx.DiGraph, path, weight: str):

        v = 0
        for i in range(0, len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            k = (node, next_node)
            w = g.edges[k][weight]
            v += w
        return v

    def get_vehicle_available_schedules(self, vehicle: Vehicle, request, timestamp: pd.Timestamp,
                                engine: SimulatorEngine):
        o = request["source"]
        d = request["target"]
        od_cost = request["shortest_cost"]
        #        od_path=request["shortest_path"]
        rid = request["id"]

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)

        o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
        d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
        time_constraint[(rid, o, 0)] = o_ts
        time_constraint[(rid, d, 1)] = d_ts

        passengers = vehicle.passengers.copy()
        passengers.update(vehicle.waiting_passengers)
        passengers[rid] = request

        schedule = list(vehicle.schedule)
        pick_up_times = {pid: p["pick_up_time"] for pid, p in vehicle.passengers.items()}
        #        targets.insert(0,start)
        tsize = len(schedule)
        schedules = []
        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                constraint = time_constraint.copy()
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k

                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > constraint[nex_k]:
                        valid = False
                        break
                    r = passengers[next_id]
                    if ntype == 0:
                        pick_up_times[next_id] = ts
                        constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi, unit='s')
                    elif ntype == 1:

                        scost = r["shortest_cost"]
                        delta_cost = (ts - pick_up_times[next_id]).total_seconds() - scost
                        costs[next_id] = (scost, delta_cost)

                    cur_node = next_node
                if valid is True:
                    schedules.append((acost, sc))

        if len(schedules) > 0:
            ws = sorted(schedules, key=lambda x: x[0])
            return ws[0]
        else:
            return None

    def get_vehicle_insert_increment(self, vehicle: Vehicle, request, timestamp: pd.Timestamp,
                                engine: SimulatorEngine):
        cur_node = vehicle.cur_node
        sc = list(vehicle.schedule)
        acost = 0
        ts = timestamp
        increment = None
        schedule = []

        for i in range(0, len(sc)):
            next_k = sc[i]
            next_id, next_node, ntype = next_k

            n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
            acost += n_cost
            cur_node = next_node
            ts += pd.Timedelta(n_cost, unit='s')

        update_sc = self.get_vehicle_available_schedules(vehicle, request, timestamp, engine)
        if update_sc is not None:
            increment = update_sc[0] - acost
            schedule = update_sc[1]

        return increment, schedule

    def assign_vehicle_to_passenger(self,vehicle: Vehicle, request, timestamp: pd.Timestamp, schedule,
                                    engine: SimulatorEngine):
        engine.assign_vehicle_to_passenger(request, vehicle, timestamp)
        engine.update_vehicle_schedule(vehicle, timestamp, schedule)
        return True

    def cosine_similarity(self, vec1, vec2):
        x = [vec1[2] - vec1[0], vec1[3] - vec1[1]]
        y = [vec2[2] - vec2[0], vec2[3] - vec2[1]]
        result1 = 0.0
        result2 = 0.0
        result3 = 0.0
        for i in range(len(x)):
            result1 += x[i] * y[i]  # sum(X*Y)
            result2 += x[i] ** 2  # sum(X*X)
            result3 += y[i] ** 2  # sum(Y*Y)

        return result1 / ((result2 * result3) ** 0.5)

class DeepPool(SharingStrategy):

    def __init__(self, max_waiting_time=600, additional_time=600):
        super().__init__()
        self.set_parameter("max_waiting_time", max_waiting_time)
        self.set_parameter("additional_time", additional_time)
        # 是am需要的，没有用
        self.waiting_requests = []  # 请求队列一个graph
        self.waiting_requests_batch = []  # 请求队列的一个batch

    def on_requests(self, req: dict, timestamp: pd.Timestamp, engine):
        self.dispatch(req, timestamp, engine)

    def dispatch(self, req: dict, timestamp: pd.Timestamp, engine: SimulatorEngine):
        self.assign_vehicles(req, timestamp, engine)

    def assign_vehicles(self, request, timestamp: pd.Timestamp, engine: SimulatorEngine):
        ts = request["timestamp"]
        delta = timestamp - ts
        duration = float(delta.total_seconds())

        last_delta = self.parameters["max_waiting_time"] - duration
        success = False
        if last_delta < 0:
            return False

        o = request["source"]
        d = request["target"]

        vhs: list = engine.vehicle_manager.search_reachable_vehicles(o, last_delta, timestamp, available=True)
        od_path, od_cost = engine.vehicle_manager.shortest_travel_path_cost(o, d, timestamp)
        if od_path is None:
            return False

        #选择到达时间最小的车
        min_time_cost = sys.maxsize
        matching_veh=None
        for value in vhs:
            vid, paths = value
            cost_path = paths[0]
            cost, path = cost_path

            vh: Vehicle = engine.vehicle_manager.vehicles[vid]
            if vh.avail_capacity > 0 :
                if cost<min_time_cost:
                     matching_veh=vh

        # 具有最小路径增量成本的车辆
        min_increment_cost = float('inf')
        min_schedule = []

        if matching_veh is not None:
            increment, schedule = self.get_vehicle_insert_increment(matching_veh, request, timestamp, engine)
            if increment is not None and increment < min_increment_cost:
                min_schedule = schedule

            is_assign = self.assign_vehicle_to_passenger(matching_veh, request, timestamp, min_schedule, engine)
            if is_assign:
                success = True
        else:
            print(matching_veh)

        return success

    def get_edge_path_cost(self, g, source, target, timestamp, engine: SimulatorEngine):
        e = (source, target)
        if e not in g.edges:
            path, cost = engine.vehicle_manager.shortest_travel_path_cost(source, target, timestamp)
            g.add_edge(e[0], e[1], path=list(path), cost=cost)

        else:
            path = g.edges[e]["path"]
            cost = g.edges[e]["cost"]
        return path, cost

    def on_pick_up_passenger(self, vehicle: Vehicle, passenger, timestamp: pd.Timestamp):
        m = self.parameters.get("additional_time", 300)

    @staticmethod
    def digraph_weight_of_path(g: nx.DiGraph, path, weight: str):

        v = 0
        for i in range(0, len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            k = (node, next_node)
            w = g.edges[k][weight]
            v += w
        return v

    def get_vehicle_available_schedules(self, vehicle: Vehicle, request, timestamp: pd.Timestamp,
                                engine: SimulatorEngine):
        o = request["source"]
        d = request["target"]
        od_cost = request["shortest_cost"]
        #        od_path=request["shortest_path"]
        rid = request["id"]

        m = self.parameters["max_waiting_time"]
        addi = self.parameters["additional_time"]
        start = vehicle.cur_node
        time_constraint = self.get_time_constraint(vehicle, m, addi)

        o_ts = request["timestamp"] + pd.Timedelta(m, unit='s')
        d_ts = o_ts + pd.Timedelta(od_cost + addi, unit='s')
        time_constraint[(rid, o, 0)] = o_ts
        time_constraint[(rid, d, 1)] = d_ts

        passengers = vehicle.passengers.copy()
        passengers.update(vehicle.waiting_passengers)
        passengers[rid] = request

        schedule = list(vehicle.schedule)
        pick_up_times = {pid: p["pick_up_time"] for pid, p in vehicle.passengers.items()}
        #        targets.insert(0,start)
        tsize = len(schedule)
        schedules = []
        for i in range(0, tsize + 1):

            for ii in range(i + 1, tsize + 2):
                sc = schedule.copy()
                ts = timestamp
                valid = True
                sc.insert(i, (rid, o, 0))
                sc.insert(ii, (rid, d, 1))
                cur_node = start

                costs = {}
                acost = 0
                constraint = time_constraint.copy()
                for iii in range(0, len(sc)):
                    nex_k = sc[iii]
                    next_id, next_node, ntype = nex_k

                    n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
                    if n_cost is None:
                        valid = False
                        break
                    acost += n_cost
                    ts += pd.Timedelta(n_cost, unit='s')
                    if ts > constraint[nex_k]:
                        valid = False
                        break
                    r = passengers[next_id]
                    if ntype == 0:
                        pick_up_times[next_id] = ts
                        constraint[(next_id, r["target"], 1)] = ts + pd.Timedelta(r["shortest_cost"] + addi, unit='s')
                    elif ntype == 1:

                        scost = r["shortest_cost"]
                        delta_cost = (ts - pick_up_times[next_id]).total_seconds() - scost
                        costs[next_id] = (scost, delta_cost)

                    cur_node = next_node
                if valid is True:
                    schedules.append((acost, sc))

        if len(schedules) > 0:
            ws = sorted(schedules, key=lambda x: x[0])
            return ws[0]
        else:
            return None

    def get_vehicle_insert_increment(self, vehicle: Vehicle, request, timestamp: pd.Timestamp,
                                engine: SimulatorEngine):
        try:
            cur_node = vehicle.cur_node
        except Exception as e:
            print(e)
        sc = list(vehicle.schedule)
        acost = 0
        ts = timestamp
        increment = None
        schedule = []

        for i in range(0, len(sc)):
            next_k = sc[i]
            next_id, next_node, ntype = next_k

            n_path, n_cost = engine.vehicle_manager.shortest_travel_path_cost(cur_node, next_node, ts)
            acost += n_cost
            cur_node = next_node
            ts += pd.Timedelta(n_cost, unit='s')

        update_sc = self.get_vehicle_available_schedules(vehicle, request, timestamp, engine)
        if update_sc is not None:
            increment = update_sc[0] - acost
            schedule = update_sc[1]

        return increment, schedule

    def assign_vehicle_to_passenger(self,vehicle: Vehicle, request, timestamp: pd.Timestamp, schedule,
                                    engine: SimulatorEngine):
        engine.assign_vehicle_to_passenger(request, vehicle, timestamp)
        engine.update_vehicle_schedule(vehicle, timestamp, schedule)
        return True

    def cosine_similarity(self, vec1, vec2):
        x = [vec1[2] - vec1[0], vec1[3] - vec1[1]]
        y = [vec2[2] - vec2[0], vec2[3] - vec2[1]]
        result1 = 0.0
        result2 = 0.0
        result3 = 0.0
        for i in range(len(x)):
            result1 += x[i] * y[i]  # sum(X*Y)
            result2 += x[i] ** 2  # sum(X*X)
            result3 += y[i] ** 2  # sum(Y*Y)

        return result1 / ((result2 * result3) ** 0.5)

if __name__ == "__main__":
    pass
#    vm=VehicleManager(g)
#    vm.uniformed_initialize_vehicles()
#    ns=NearestSharing()
#    engine=SimulatorEngine(vm,ns)
#    engine.set_requests(req)
#    engine.run()
