import os
import datatable
import itertools
import torch

import numpy as np
import pandas as pd

from datetime import datetime
from loguru import logger
from typing import Union
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, HeteroData
from torch_geometric.typing import OptTensor
from torch_geometric.transforms import BaseTransform


def to_adj_nodes_with_times(data):
    # Codes modified from https://github.com/IBM/Multi-GNN/blob/main/data_util.py
    num_nodes = data.num_nodes
    timestamps = torch.zeros(
        (data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1, 1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(
        data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out


def to_adj_edges_with_times(data):
    # Codes adopted from https://github.com/IBM/Multi-GNN/blob/main/data_util.py
    num_nodes = data.num_nodes
    timestamps = torch.zeros(
        (data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1, 1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u, v, t) in enumerate(edges):
        u, v, t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out


def ports(edge_index, adj_list):
    # Codes adopted from https://github.com/IBM/Multi-GNN/blob/main/data_util.py
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1:
            continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:, [0]], return_index=True, axis=0)
        nbs_unique = a[np.sort(idx)][:, 0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u, v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports


def time_deltas(data, adj_edges_list):
    # Codes adopted from https://github.com/IBM/Multi-GNN/blob/main/data_util.py
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1:
            continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1, -1] - a[i, -1] for i in range(a.shape[0]-1)]
        tds = np.hstack(
            (a[:, 0].reshape(-1, 1), np.array(a_tds).reshape(-1, 1)))
        for i, td in tds:
            time_deltas[i] = td
    return time_deltas


# Codes adopted from https://github.com/IBM/Multi-GNN/blob/main/data_util.py
class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''

    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None,
        readout: str = 'edge',
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps
        elif edge_attr is not None:
            self.timestamps = edge_attr[:, 0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(),
                           adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat(
            [self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self


# Codes adopted from https://github.com/IBM/Multi-GNN/blob/main/train_util.py
class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """

    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(
                data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else:
            data['node'].x = torch.cat([x, ids], dim=1)

        return data


def z_norm(data):
    std = data.std(0)
    std = torch.where(std == 0, torch.tensor(
        1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0)) / std


class AMLworld(InMemoryDataset):

    def __init__(self, root, opt="HI-Small", split="train", load_time_stamp=True, load_ports=True, load_time_delta=False, transform=None, pre_transform=None, pre_fileter=None, force_download=False, verbose=True, ibm_split=False, *args, **kwargs):
        """
        Args:
            opt (str, optional): _description_. Defaults to "HI-Small".
        """

        self.force_download = force_download
        self.verbose = verbose
        self.processed_in_this_call = False
        self.ibm_split = ibm_split

        all_options = ["HI-Small", "HI-Medium", "HI-Large",
                       "LI-Small", "LI-Medium", "LI-Large"]

        if opt not in all_options:
            raise Exception(
                f"Unsupported option. Must be choosen from {all_options}.")

        self.opt = opt

        super().__init__(root, transform, pre_transform, pre_fileter, *args, **kwargs)

        # self.load(self.processed_file_paths_by_split[split])
        self.data = torch.load(self.processed_file_paths_by_split[split])


        # Select features based on attribute options
        feature_cols = []
        exclude_feature_name = []
        # TODO: Add 24-h time; One hot encoding for Payment Method
        if not load_time_stamp:
            exclude_feature_name.append("Timestamp")
        if not load_ports:
            exclude_feature_name.extend(["In-Port", "Out-Port"])
        if not load_time_delta:
            exclude_feature_name.extend(["In-TimeDelta", "Out-TimeDelta"])
        exclude_feature_name = set(exclude_feature_name)
        cnt_newID = 0
        feature_newID = {}
        for feature, id in self._data.edge_features_colID.items():
            if feature not in exclude_feature_name:
                feature_cols.append(id)
                feature_newID[feature] = cnt_newID
                cnt_newID += 1
        self._data.edge_attr = self._data.edge_attr[:, feature_cols]
        self._data.edge_features_colID = feature_newID

        if not self.processed_in_this_call and self.verbose:
            logger.info(self._data.information)
            edge_features = list(feature_newID.keys())
            logger.info(f'Edge features being used: {edge_features}')
            
        self.num_nodes = self._data.num_nodes
        self.num_edges = self._data.num_edges

    @property
    def raw_file_names(self):
        all_files = ["HI-Large_Patterns.txt",
                     "HI-Large_Trans.csv",
                     "HI-Medium_Patterns.txt",
                     "HI-Medium_Trans.csv",
                     "HI-Small_Patterns.txt",
                     "HI-Small_Trans.csv",
                     "LI-Large_Patterns.txt",
                     "LI-Large_Trans.csv",
                     "LI-Medium_Patterns.txt",
                     "LI-Medium_Trans.csv",
                     "LI-Small_Patterns.txt",
                     "LI-Small_Trans.csv"]

        opt_files = []
        for file in all_files:
            if file.startswith(self.opt):
                opt_files.append(file)

        return opt_files

    @property
    def processed_file_names_by_split(self):
        file_dict = {
                "train": f"{self.opt}-train.pt",
                "val": f"{self.opt}-val.pt",
                "test": f"{self.opt}-test.pt"
            }
        if self.ibm_split:
            for split, file in file_dict.items():
                file_name = os.path.splitext(file)[0]
                file_dict[split] = f"{file_name}-ibm_split.pt"
        return file_dict

    @property
    def processed_file_names(self):
        return list(self.processed_file_names_by_split.values())
    
    
    @property 
    def processed_file_paths_by_split(self):
        file_paths_dict = {}
        for split, file_name in self.processed_file_names_by_split.items():
            file_paths_dict[split] = os.path.join(self.processed_dir, file_name)
        return file_paths_dict

    def download(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files("ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
                                       self.raw_dir, unzip=True, force=self.force_download, quiet=not self.verbose)
        except:
            raise Exception(
                "Downloading this dataset requires a configured Kaggle API for Python. Please refer to https://github.com/Kaggle/kaggle-api for proper configuration!")

    def process(self):
        self.processed_in_this_call = True
        infor_str_list = []


        ################################################################
        # Step 1. Convert raw files from Kaggle to a formatted datatable. Adopt codes from "https://github.com/IBM/Multi-GNN/blob/main/format_kaggle_files.py"

        def get_dict_val(name, collection):
            if name in collection:
                val = collection[name]
            else:
                val = len(collection)
                collection[name] = val
            return val

        raw_trans_file = os.path.join(self.raw_dir, self.opt+"_Trans.csv")
        raw = datatable.fread(raw_trans_file, columns=datatable.str32)

        formatted_trans_file = os.path.join(
            self.processed_dir, self.opt+"-formatted_transactions.csv")

        header = "EdgeID,from_id,to_id,Timestamp,Amount Sent,Sent Currency,Amount Received,Received Currency,Payment Format,Is Laundering\n"

        firstTs = -1
        currency = dict()
        paymentFormat = dict()
        account = dict()

        with open(formatted_trans_file, "w") as writer:
            writer.write(header)
            for i in tqdm(range(raw.nrows), desc="Formatting files from Kaggle raw data:", disable=not self.verbose):
                datetime_object = datetime.strptime(
                    raw[i, "Timestamp"], '%Y/%m/%d %H:%M')
                ts = datetime_object.timestamp()
                day = datetime_object.day
                month = datetime_object.month
                year = datetime_object.year

                if firstTs == -1:
                    startTime = datetime(year, month, day)
                    firstTs = startTime.timestamp() - 10

                ts = ts - firstTs

                cur1 = get_dict_val(raw[i, "Receiving Currency"], currency)
                cur2 = get_dict_val(raw[i, "Payment Currency"], currency)

                fmt = get_dict_val(raw[i, "Payment Format"], paymentFormat)

                fromAccIdStr = raw[i, "From Bank"] + raw[i, 2]
                fromId = get_dict_val(fromAccIdStr, account)

                toAccIdStr = raw[i, "To Bank"] + raw[i, 4]
                toId = get_dict_val(toAccIdStr, account)

                amountReceivedOrig = float(raw[i, "Amount Received"])
                amountPaidOrig = float(raw[i, "Amount Paid"])

                isl = int(raw[i, "Is Laundering"])

                line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i, fromId, toId, ts, amountPaidOrig,
                     cur2, amountReceivedOrig, cur1, fmt, isl)

                writer.write(line)

        formatted = datatable.fread(formatted_trans_file)
        formatted = formatted[:, :, datatable.sort(3)]
        formatted.to_csv(formatted_trans_file)

        ################################################################
        # Step 2. Load datatable as a PyG data object. Adopt codes from "https://github.com/IBM/Multi-GNN/blob/main/data_loading.py"
        df_edges = pd.read_csv(formatted_trans_file)
        if self.verbose:
            logger.info(
                f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges['Timestamp'] = df_edges['Timestamp'] - \
            df_edges['Timestamp'].min()

        # Node Features are not available in this dataset. Set to all 1 as an auxiliary feature.
        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({'NodeID': np.arange(
            max_n_id), 'Feature': np.ones(max_n_id)})

        # Edge: transaction timestamp
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())

        # Edge: transaction lable
        y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

        infor_str_list.append(
            f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        infor_str_list.append(
            f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
        infor_str_list.append(
            f"Number of transactions = {df_edges.shape[0]}")
        if self.verbose:
            logger.info(
                f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
            logger.info(
                f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
            logger.info(f"Number of transactions = {df_edges.shape[0]}")

        # Define Node and Edge
        edge_features = ['Timestamp', 'Amount Received',
                         'Received Currency', 'Payment Format']
        node_features = ['Feature']

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(
            df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
        edge_attr = torch.tensor(
            df_edges.loc[:, edge_features].to_numpy()).float()

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        infor_str_list.append(
            f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')
        if self.verbose:
            logger.info(
                f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

        # Data splitting
        # irs = illicit ratios, inds = indices, trans = transactions
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(
                y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i, j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(),
                                d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v/split_totals_sum for v in split_totals]
                split_error = [
                    abs(v-t)/t for v, t in zip(split_props, split_per)]
                score = max(split_error)  # - (split_totals_sum/total) + 1
                split_scores[(i, j)] = score
            else:
                continue

        i, j = min(split_scores, key=split_scores.get)
        # Split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)),
                 list(range(j, len(daily_totals)))]
        if self.verbose:
            logger.info(f'Calculate split: {split}')

        # Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately
                split_inds[i].append(daily_inds[day])

        tr_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        te_inds = torch.cat(split_inds[2])

        infor_str_list.append(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                              f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0]}")
        infor_str_list.append(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                              f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1]}")
        infor_str_list.append(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                              f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2]}")
        if self.verbose:
            logger.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                        f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0]}")
            logger.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1]}")
            logger.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2]}")

        # Creating the final data objects
        tr_x, val_x, te_x = x, x, x
        e_tr = tr_inds.numpy()
        if self.ibm_split:
            e_val = np.concatenate([tr_inds, val_inds])
            e_te = np.arange(edge_attr.shape[0])
            try:
                logger.warning("Following IBM's repo, all edges are considered in test set. ")
            except:
                print("Warning: Following IBM's repo, all edges are considered in test set.")
        else:
            e_val = val_inds.numpy()
            e_te = te_inds.numpy()

        tr_edge_index = edge_index[:, e_tr]
        tr_edge_attr = edge_attr[e_tr]
        tr_y = y[e_tr]
        tr_edge_times = timestamps[e_tr]
        tr_data = GraphData(
            x=tr_x,
            y=tr_y,
            edge_index=tr_edge_index,
            edge_attr=tr_edge_attr,
            timestamps=tr_edge_times
        )

        val_edge_index = edge_index[:, e_val]
        val_edge_attr = edge_attr[e_val]
        val_y = y[e_val]
        val_edge_times = timestamps[e_val]
        val_data = GraphData(
            x=val_x,
            y=val_y,
            edge_index=val_edge_index,
            edge_attr=val_edge_attr,
            timestamps=val_edge_times
        )

        # TODO: Check why te_inds is not used!
        te_edge_index = edge_index[:, e_te]
        te_edge_attr = edge_attr[e_te]
        te_y = y[e_te]
        te_edge_times = timestamps[e_te]
        te_data = GraphData(
            x=te_x,
            y=te_y,
            edge_index=te_edge_index,
            edge_attr=te_edge_attr,
            timestamps=te_edge_times
        )

        # Adding ports and time-deltas
        if self.verbose:
            logger.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        edge_features.extend(["In-Port", "Out-Port"])
        if self.verbose:
            logger.info(f"Done: adding ports")

        if self.verbose:
            logger.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        edge_features.extend(["In-TimeDelta", "Out-TimeDelta"])
        if self.verbose:
            logger.info(f"Done: adding time-deltas")

        edge_features_colID = {}
        for id, feature in enumerate(edge_features):
            edge_features_colID[feature] = id

        infor_str_list.append(
            f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')
        if self.verbose:
            logger.info(
                f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        # Normalize node attribute
        tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)

        # Normalize the edge attribute
        norm_col = []
        for feature, id in edge_features_colID.items():
            if feature not in ["Payment Format"]:
                norm_col.append(id)

        tr_data.edge_attr[:, norm_col] = z_norm(tr_data.edge_attr[:, norm_col])
        val_data.edge_attr[:, norm_col] = z_norm(
            val_data.edge_attr[:, norm_col])
        te_data.edge_attr[:, norm_col] = z_norm(te_data.edge_attr[:, norm_col])

        information = "\n".join(infor_str_list)
        for data_split in [tr_data, val_data, te_data]:
            data_split.information = information
            data_split.edge_features_colID = edge_features_colID

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        file_dict = self.processed_file_paths_by_split
        torch.save(tr_data, file_dict["train"])
        torch.save(val_data, file_dict["val"])
        torch.save(te_data, file_dict["test"])

        # TODO: Process pattern tag into data object
        raw_pattern_file = os.path.join(self.raw_dir, self.opt+"_Patterns.txt")


def main():
    dataset = AMLworld("./dataset/AMLworld")


if __name__ == "__main__":
    main()
