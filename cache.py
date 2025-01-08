# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:38:22 2020

@author: Administrator
"""

import sqlite3
import networkx as nx
import pandas as pd 
import joblib
import os 
import pickle

def create_shortest_path_cost_table(tname,cursor):
    sql="""create table if not exists {0}(
    source int ,
    target int ,
    cost real,
    primary key(source,target));""".format(tname)
    cursor.execute(sql)

def create_all_time_table(con):
    cursor=con.cursor()
    for h in range(24):
        create_shortest_path_cost_table(h,cursor)
        print("processing time id:{0}".format(h))
        
def cache_nodes_shortest_path_cost(g:nx.MultiDiGraph,weight:str,nodes:list=None,cache_dir=".//",report=0.1):
    if nodes is None:
        nodes=g.nodes
    len_node=len(nodes)
    unit=1/report
    p=0
    dir_=os.path.abspath(cache_dir)
    if os.path.exists(dir_) is False:
        os.mkdir(dir_)
    path=os.path.join(dir_,"{0}.pkl".format(weight))
    d={}
    print("pending process nodes num:{0}".format(len_node))
    for i,n in enumerate(nodes):
        source=n
        p1=int(i/len_node*100*unit)
        if p1!=p:
            p=p1
            p0=i/len_node*100
            print("complete: {0} % source node:{1}".format(p0,source))
        
        c=nx.single_source_dijkstra_path_length(g,source,weight=weight)
        d[n]=c
    
    try:
#        joblib.dump(d,path,3)
        with open(path,'wb') as f:
            pickle.dump(d,f)
    except Exception as e:
        print(e,"cache")
    return d
        
        
def cache_all_pair_shortest_path_cost(g:nx.MultiGraph,weight:str,con,nodes:list=None,report=0.1):
    if nodes is None:
        nodes=g.nodes
        
    cursor=con.cursor()
    create_shortest_path_cost_table(weight,cursor)
    len_node=len(nodes)
    unit=1/report
    p=0
    
    print("pending process nodes num:{0}".format(len_node))
    for i,n in enumerate(nodes):
        source=n
        p1=int(i/len_node*100*unit)
        if p1!=p:
            p=p1
            p0=i/len_node*100
            print("complete: {0} % source node:{1}".format(p0,source))
        
        c=nx.single_source_dijkstra_path_length(g,source,weight=weight)
        c1=[(source,k,v) for k,v in c.items()]
        c2=pd.DataFrame(c1,columns=["source","target","cost"])
        c2.set_index(["source","target"],inplace=True)
        c2.to_sql(weight,con,if_exists="append")
        
    
def cache_shortest_path_costs(g:nx.MultiDiGraph,time_ids,nodes:list=None,cache_dir=".//",report=0.1):
    
    if nodes is None:
        nodes=g.nodes
    for h in time_ids:
        w="travel_time_{0}".format(h)
        print("processing time id: {0}".format(h))
        try:
            a=cache_nodes_shortest_path_cost(g,w,nodes,cache_dir,report)
        except Exception as e:
            print(e,"cache_shortest_path_costs")
        