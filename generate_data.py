import os
import numpy as np
from ridesharing.utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_hcvrp_data(dataset_size, hcvrp_size, veh_num):
    data = []
    for seed in range(24601, 24611):    # 为什么是10个seed？
        rnd = np.random.RandomState(seed)
        detour = 5
        loc = rnd.uniform(0, 1, size=(dataset_size, hcvrp_size, 2))
        veh_loc = rnd.uniform(0, 1, size=(dataset_size, veh_num, 2))    # 初始化车的位置
        # depot = loc[:, -1]
        cust = loc[:, :]
        veh = veh_loc[:, :]
        # d = rnd.randint(1, 10, [dataset_size, hcvrp_size+1])
        # d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here
        veh_cap = get_capacity(dataset_size, veh_num)    # 初始化车的容量
        d = get_demand(dataset_size, hcvrp_size)
        t = get_time(dataset_size, hcvrp_size, detour)
        # if veh_num == 3:
        #     cap = [20., 25., 30.]
        #     thedata = list(zip(#depot.tolist(),   Depot location
        #                        cust.tolist(),
        #                        d,
        #                        np.full((dataset_size, 3), cap).tolist(),
        #                         t
        #                         ))
        #
        #     data.append(thedata)
        #
        # elif veh_num == 5:
        #     cap = [20., 25., 30., 35., 40.]
        #     thedata = list(zip(# Depot location depot.tolist(),
        #                        cust.tolist(),
        #                        d,
        #                        np.full((dataset_size, 5), cap).tolist(),
        #                         t
        #                        ))
        #     data.append(thedata)
        thedata = list(zip(
            veh.tolist(),
            cust.tolist(),
            d,
            np.full((dataset_size, veh_num), veh_cap).tolist(),
            t
        ))
        data.append(thedata)


    # data = np.array(data).reshape(1280, 4)
    return data

def get_capacity(dataset_size, veh_num):
    cap = np.random.randint(1, 5, size=(dataset_size, veh_num))     # 最大容量为6
    return cap.tolist()

def get_demand(dataset_size, hcvrp_size):

    a = np.random.randint(1, 5, size=(dataset_size, hcvrp_size))    # 默认最多一次4个人
    for row in a:
        for i in range(int(len(row) / 2)):
            row[i + int(hcvrp_size / 2)] = row[i] * -1

    return a.tolist()

def get_time(dataset_size, hcvrp_size, DETOUR_TIME=None):
    time = torch.cat((torch.FloatTensor(dataset_size, hcvrp_size//2).uniform_(5, 15), torch.FloatTensor(dataset_size, hcvrp_size//2).uniform_(10, 30)+DETOUR_TIME), 1)
    return time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=128, help="1/10 Size of the dataset")   # default 128
    parser.add_argument("--veh_num", type=int, default=10, help="number of the vehicles; 3 or 5")    # default 3
    parser.add_argument('--graph_size', type=int, default=50,
                        help="Sizes of problem instances: {40, 60, 80, 100, 120} for 3 vehicles, "
                             "{80, 100, 120, 140, 160} for 5 vehicles")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'ridesharing'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating HCVRP data
    np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size)
    filename = os.path.join(datadir, '{}_v{}_{}_seed{}.pkl'.format(problem, opts.veh_num, opts.graph_size, seed))

    dataset = generate_hcvrp_data(opts.dataset_size, opts.graph_size, opts.veh_num)
    print(dataset[0])
    save_dataset(dataset, filename)



