import os
import numpy as np
import torch
import torch.random
import pandas as pd
import pickle as pkl
import random
import model
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Bio import SeqIO
from config import args
from utils import PosEdge, AverageMeter, check_folder

check_folder("edge_index/")

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
inputs = args.parse_args()

if args.use_cpu:
    print("Running with CPU")
    device = torch.device('cpu')
elif torch.cuda.is_available():
    torch.cuda.set_device(inputs.gpus)
    device = torch.device('cuda')
     print(f"Running with GPU: {args.gpus}")
else:
    print("Running with cpu (no GPU available)")
    device = torch.device('cpu')

adj = pkl.load(open("GCN_data/graph.list", "rb"))
features = pkl.load(open("GCN_data/feature.list",'rb'))
id2node = pkl.load(open("GCN_data/id2node.dict", "rb"))
node2id = pkl.load(open("GCN_data/node2id.dict", "rb"))
idx_test = pkl.load(open("GCN_data/test_id.dict", "rb"))
node2label = pkl.load(open("GCN_data/node2label.dict", "rb"))
crispr_pred = pkl.load(open("out/crispr_pred.dict", "rb"))
RBP_pred = pkl.load(open("out/RBP_pred.dict", "rb"))
rRNA_pred = pkl.load(open("out/rRNA_pred.dict", "rb"))
prokaryote_df = pd.read_csv('dataset/prokaryote.csv')

num_test = 0
for _ in SeqIO.parse('test.fasta', 'fasta'):
    num_test += 1

trainable_host = []
for file in os.listdir('prokaryote/'):
    trainable_host.append(file.rsplit('.', 1)[0])

host2id = {}
label2hostid = {}
trainable_host_idx = []
trainable_label = []
for idx, node in id2node.items():
    if node in trainable_host:
        host2id[node] = idx
        trainable_host_idx.append(idx)
        for label in node2label[node]:
            trainable_label.append(label)
        for label in node2label[node]:
            if label not in label2hostid:
                label2hostid[label] = []
                label2hostid[label].append(idx)
            elif label in label2hostid:
                label2hostid[label].append(idx)

trainable_label = list(set(trainable_label))

phages2spe = {}
df_phages = pd.read_csv('dataset/phages.csv')

for i in range(len(df_phages)):
    phages = df_phages.loc[i]['Accession']
    label = df_phages.loc[i]['Species']
    #label = df_phages.loc[i]['Genus']
    #label = df_phages.loc[i]['Family']
    #label = df_phages.loc[i]['Order']
    if phages not in phages2spe:
        phages2spe[phages] = []
        phages2spe[phages].append(label)
    elif phages in phages2spe:
        phages2spe[phages].append(label)

for key in phages2spe:
    phages2spe[key] = list(set(phages2spe[key]))

newphage2name = {}
name2newphage = {}
df_name = pd.read_csv('name_list.csv')

for i in range(len(df_name)):
    name = df_name.loc[i]['contig_name'].split('.')[0]
    newphage = df_name.loc[i]['idx']

    newphage2name[newphage] = name
    name2newphage[name] = newphage

edge_index = from_scipy_sparse_matrix(adj)[0]
adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t().to(device)
x = torch.tensor(features, dtype=torch.float, device=device)

preprocess_dir = 'edge_index/'
print('Preprocessing the edge_index data')
edge_index_pos_list = [[], []]

for label in trainable_label:
    host_idxs = label2hostid[label]
    for host_idx in host_idxs:
        for idx in range(len(node2id)):
            if idx not in trainable_host_idx and label in node2label[id2node[idx]]:
                edge_index_pos_list[0].append(idx)
                edge_index_pos_list[1].append(host_idx)
edge_index_pos = torch.tensor(edge_index_pos_list, dtype=torch.long)

trainable_host_idx_set = set(trainable_host_idx)
phages2pro_neg = {}

for phages, pro in zip(edge_index_pos_list[0], edge_index_pos_list[1]):
    phages2pro_neg[phages] = list(trainable_host_idx_set - set([pro]))

torch.save(edge_index_pos_list, preprocess_dir + 'edge_index_pos_list')
torch.save(edge_index_pos, preprocess_dir + 'edge_index_pos')
torch.save(trainable_host_idx_set, preprocess_dir + 'trainable_host_idx_set')
torch.save(phages2pro_neg, preprocess_dir + 'phages2pro_neg')

print('Finished preprocess..')

def neg_sample(edge_index_pos_batch):
    edge_index_neg = [[], []]
    for phages in edge_index_pos_batch[0].cpu().tolist():
        fake_pro = random.choice(phages2pro_neg[phages])
        edge_index_neg[0].append(phages)
        edge_index_neg[1].append(fake_pro)

    return torch.tensor(edge_index_neg, dtype=torch.long, device=device)

def train_accuracy():
    with torch.no_grad():
        total = 0
        correct = 0
        for i in range(len(encode)):
            if 'newphage' in id2node[i] or id2node[i] not in idx_test:
                continue
            phages_feature = encode[i]

            prokaryote_feature = encode[trainable_host_idx]
            preds = decoder(phages_feature, prokaryote_feature)
            pred_label = node2label[id2node[trainable_host_idx[preds.argmax().cpu().item()]]]
            for label in pred_label:
                if label in node2label[id2node[i]]:
                    correct += 1
            total += 1
    return correct / total

# def train_topk_accuracy(k):
#     with torch.no_grad():
#         total = 0
#         correct = 0
#         for i in range(len(encode)):
#             if 'newphage' in id2node[i] or id2node[i] not in idx_test:
#                 continue
#             pred_label = []
#             phages_feature = encode[i]
#             prokaryote_feature = encode[trainable_host_idx]
#             preds = decoder(phages_feature, prokaryote_feature)
#             _, sorted_indices = preds.sort(0, descending=False)
#             for index in sorted_indices:
#                 label = node2label[id2node[trainable_host_idx[index.item()]]]
#                 pred = preds[index.item()].item()
#                 pred_label.append((label, pred))
#             real_label = node2label[id2node[i]]
#             if real_label in pred_label[:k]:
#                 correct += 1
#             total += 1
#     return correct/total



net = model.GATV2_Encoder(len(node2id), inputs, device).to(device)
decoder = model.DotDecoder().to(device)

params = list(net.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=inputs.lr, weight_decay=inputs.weight_decay)
#criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

if inputs.model == 'pretrain':
    net_dict = torch.load(f"saved_model/encoder.pkl", map_location='cpu')
    current_num_nodes = len(node2id)
    pretrained_num_nodes = net_dict["emb.weight"].shape[0]
    
    if current_num_nodes < pretrained_num_nodes:
        net_dict["emb.weight"] = net_dict["emb.weight"][:current_num_nodes, :]
    elif current_num_nodes > pretrained_num_nodes:
        padding = torch.randn(current_num_nodes - pretrained_num_nodes, 256)
        net_dict["emb.weight"] = torch.cat([net_dict["emb.weight"], padding], dim=0)
    
    net.load_state_dict(net_dict, strict=False)

def collate_fn(data):
    return torch.hstack(data)

batch_size = inputs.batch_size

edge_index_pos_datset = PosEdge(edge_index_pos)
edge_index_pos_dataloader = DataLoader(edge_index_pos_datset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#################################################################
##########################  Training  ###########################
#################################################################

early_stop = 5
cur_test_acc = 0.0
EPS = 1e-15

if inputs.model == 'retrain':
    _ = net.train()
    _ = decoder.train()
    for epoch in range(1, 1 + inputs.epochs):
        batch_cal = AverageMeter()
        for batch, edge_index_pos_batch in enumerate(edge_index_pos_dataloader):
            encode = net(x, adj_t)
            edge_index_neg_batch = neg_sample(edge_index_pos_batch)
            loss = 0
            phages_feat = encode[edge_index_pos_batch[0]]
            pro_feat_pos = encode[edge_index_pos_batch[1]]
            pro_feat_neg = encode[edge_index_neg_batch[1]]
            pred_pos = decoder(phages_feat, pro_feat_pos)
            pred_neg = decoder(phages_feat, pro_feat_neg)
            loss = torch.sum(F.softplus(pred_neg - pred_pos))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_cal.update(loss.cpu().item())

        if (epoch % 10 == 0):
            train_acc = train_accuracy()
            print("[Epoch {0:>4d}] Train acc: {1:.2f}, Loss: {2:.3f}".format(epoch, train_acc, loss.cpu().item()))

            if train_acc > 0.8:
                break

    torch.save(net.state_dict(), f'saved_model/encoder.pkl')
    torch.save(decoder.state_dict(), f'saved_model/decoder.pkl')

#################################################################
#########################  Prediction  ##########################
#################################################################

total_confident = 0
top_eq_conf = 0

node2pred = {}
with torch.no_grad():
    encode = net(adj_t)
    for i in range(len(encode)):
        confident_label = 'unknown'
        if id2node[i] not in idx_test or idx_test[id2node[i]] == 0:
            continue
        if id2node[i] in idx_test and idx_test[id2node[i]] == 1:
            confident_label = node2label[id2node[i]]
        phages_feature = encode[i]
        pred_label_score = []
        for label in set(trainable_label):
            if label in confident_label:
                pred_label_score.append((label, float('inf')))
                continue
            prokaryote_feature = encode[label2hostid[label]]
            preds = decoder(phages_feature, prokaryote_feature)
            for pred in preds:
                pred_label_score.append((label, pred.detach().cpu().numpy()))
        node2pred[id2node[i]] = sorted(pred_label_score, key=lambda tup: tup[1], reverse=True)

       


k = inputs.topk
data = {'contig': []}

for i in range(k):
    data['top_{0}'.format(i+1)] = []

for node, pred in node2pred.items():
    data['contig'].append(newphage2name[node])
    for i in range(k):
        data['top_{0}'.format(i+1)].append(pred[i][0])

df_pred = pd.DataFrame(data=data)
df_pred.to_csv('final_prediction.csv', index=False)

print('Prediction Finished...')




