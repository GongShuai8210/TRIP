# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import clip
import torchvision.datasets as datasets
import torch.nn as nn
from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import os
from collections import defaultdict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import json
import sys
import os
import time
from utils.config import img_param_init, set_random_seed,get_classname
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from nets.coop2_NR_data import *
from collections import OrderedDict
from torch.nn import functional as F
from utils.tools import print_color_text
from utils.lr_scheduler import ConstantWarmupScheduler
from torch.cuda.amp import autocast, GradScaler
import scipy as sp




try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.t())

def kmeans_torch(X, num_clusters, num_iters=10):
    """
    使用 PyTorch 在 GPU 上实现 KMeans 聚类

    参数：
    - X: 数据张量，形状为 (num_samples, embedding_dim)
    - num_clusters: 聚类数量（专家数量）
    - num_iters: 迭代次数

    返回：
    - cluster_assignments: 每个样本的聚类分配，形状为 (num_samples,)
    - centroids: 聚类中心，形状为 (num_clusters, embedding_dim)
    """
    # 随机从数据中初始化聚类中心
    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices]

    for _ in range(num_iters):
        # 计算样本与聚类中心之间的距离
        distances = torch.cdist(X, centroids, p=2)  # 形状：(num_samples, num_clusters)
        # 分配每个样本到最近的聚类中心
        cluster_assignments = torch.argmin(distances, dim=1)  # 形状：(num_samples,)
        # 更新聚类中心
        for k in range(num_clusters):
            assigned_samples = X[cluster_assignments == k]
            if assigned_samples.size(0) > 0:
                centroids[k] = assigned_samples.mean(dim=0)
            else:
                # 如果某个聚类中心没有分配到样本，重新随机初始化
                centroids[k] = X[torch.randint(0, X.size(0), (1,))]
    return cluster_assignments, centroids

def initialize_farthest_point(X, centroids):
    """
    选择距离当前所有聚类中心最远的样本作为新的聚类中心。

    参数：
    - X: 输入数据，形状为 (num_samples, embedding_dim)
    - centroids: 当前的聚类中心，形状为 (num_clusters, embedding_dim)

    返回：
    - new_centroid: 新的聚类中心，形状为 (embedding_dim,)
    """
    if centroids.numel() == 0:
        # 如果当前没有聚类中心，随机选择一个样本
        return X[torch.randint(0, X.size(0), (1,))]

    # 计算每个样本到所有聚类中心的最小距离
    distances = torch.cdist(X, centroids)  # (num_samples, num_clusters)
    min_distances, _ = distances.min(dim=1)  # (num_samples,)

    # 选择距离最远的样本
    farthest_index = torch.argmax(min_distances)
    return X[farthest_index]

def initialize_kmeans_pp(X, num_clusters):
    # 初始化一个张量以存储聚类中心
    centroids = torch.empty((num_clusters, X.size(1)), device=X.device)
    # 随机选择第一个聚类中心
    first_idx = torch.randint(0, X.size(0), (1,)).item()
    centroids[0] = X[first_idx]
    for i in range(1, num_clusters):
        # 计算每个样本到当前已选择的聚类中心的最小距离
        distances = torch.cdist(X, centroids[:i].half())  # 只取前 i 个已选择的聚类中心
        min_distances, _ = distances.min(dim=1)
        # 根据距离平方计算选择新聚类中心的概率分布
        probabilities = min_distances ** 2
        probabilities /= probabilities.sum()

        if probabilities.sum() == 0:  # 如果概率和为0，则重新随机选择一个样本
            next_centroid_idx = torch.randint(0, X.size(0), (1,)).item()
        else:
            next_centroid_idx = torch.multinomial(probabilities, 1).item()

            # 根据概率分布选择下一个聚类中心
        # 根据概率分布选择下一个聚类中心
        centroids[i] = centroids[i] = X[next_centroid_idx]
    return centroids.half()

def capacity_constrained_assignment(distances, lambdas, capacities):
    """
    在容量限制下，根据调整后的成本（距离加上拉格朗日乘子）为样本分配聚类。
    如果首选的聚类已满，未能分配的样本将被标记为 -1（舍弃）。

    参数：
    - distances: 张量，形状为 (num_samples, num_clusters)，表示样本到聚类中心的距离。
    - lambdas: 张量，形状为 (num_clusters,)，每个聚类的拉格朗日乘子。
    - capacities: 张量，形状为 (num_clusters,)，每个聚类的剩余容量。

    返回：
    - assignments: 张量，形状为 (num_samples,)，每个样本的聚类分配。
    """
    num_samples, num_clusters = distances.shape
    device = distances.device

    # 计算调整后的成本：距离加上拉格朗日乘子
    adjusted_cost = distances + lambdas.unsqueeze(0)  # (num_samples, num_clusters)
    min_costs, preferred_clusters = torch.min(adjusted_cost, dim=1)
    sorted_costs, sorted_indices = torch.sort(min_costs)
    sorted_preferred_clusters = preferred_clusters[sorted_indices]
    # 初始化 assignments，所有值为 -1（未分配）
    assignments = torch.full((num_samples,), -1, dtype=torch.long, device=device)

    # 复制 capacities，以便在分配过程中更新
    remaining_capacities = capacities.clone()

    for idx in range(num_samples):
        i = sorted_indices[idx]  # 原始样本索引
        k = sorted_preferred_clusters[idx]  # 样本 i 的首选聚类
        if remaining_capacities[k] > 0:
            assignments[i] = k
            remaining_capacities[k] -= 1
        else:
            # 首选聚类已满，无法分配，样本保持为 -1
            pass

    return assignments

def balanced_kmeans_lagrangian(X, capacity,num_clusters, max_iters=100, eta=1.0, tol=1e-4, reinit_strategy='farthest'):

    device = X.device
    num_samples, embedding_dim = X.shape

    # 计算每个聚类的目标容量，向下取整
    base_capacity = num_samples // num_clusters * capacity
    capacities = torch.full((num_clusters,), base_capacity, dtype=torch.long, device=device)
    # 将剩余的容量分配给前面的聚类
    for i in range(num_samples % num_clusters):
        capacities[i] += 1

    # 随机初始化聚类中心
    indices = torch.randperm(num_samples)[:num_clusters]
    centroids = X[indices].clone()

    # 初始化拉格朗日乘子
    lambdas = torch.zeros(num_clusters, device=device)

    for iteration in range(max_iters):
        # 计算样本与聚类中心的距离
        distances = torch.cdist(X, centroids)  # (num_samples, num_clusters)

        assignments = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            assigned_indices = (assignments == k).nonzero(as_tuple=True)[0]
            if assigned_indices.numel() > 0:
                new_centroids[k] = X[assigned_indices].mean(dim=0)
            else:
                # 重新初始化空聚类中心
                if reinit_strategy == 'farthest':
                    new_centroids[k] = initialize_farthest_point(X, centroids)
                elif reinit_strategy == 'kmeans++':
                    new_centroids[k] = initialize_kmeans_pp(X, 1)
                else:
                    new_centroids[k] = X[torch.randint(0, num_samples, (1,))]
                # 重置对应的拉格朗日乘子


        # 检查收敛
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).mean()
        if centroid_shift < tol:
            break

        centroids = new_centroids

    # 最终的聚类大小
    cluster_sizes = torch.bincount(assignments, minlength=num_clusters)
    return assignments, centroids, cluster_sizes, lambdas

def expert_weights(centroids,key_vectors,cluster_sizes):
    similarity_matrix = cosine_similarity(centroids, key_vectors.half())
    cost_matrix = -similarity_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    selected_experts = col_ind

    # 构建专家到聚类的映射
    cluster_indices = torch.arange(len(selected_experts), device=key_vectors.device)
    expert_indices = torch.tensor(selected_experts, device=cluster_sizes.device)

    # 排序专家编号，获取对应的聚类编号
    sorted_expert_indices, sort_idx = torch.sort(expert_indices)
    sorted_cluster_indices = cluster_indices[sort_idx]

    # 获取每个专家对应的聚类中的 token 数量
    token_counts = cluster_sizes[sorted_cluster_indices]

    # 总的 token 数量
    total_tokens = cluster_sizes.sum()

    # 计算比例
    proportions = token_counts.float() / total_tokens

    return proportions
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):

    def __init__(self, dataset, root,domain_index,model, args):
        dataset = os.path.join(root, dataset)
        data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels
        self.preprocess = model.preprocess
        if args.cluster:
            self.key_vectors = args.key_vectors
            with torch.no_grad():
                weight_train_list = []
                weight_infer_list = []
                for idx,value in enumerate(tqdm(data.imgs)):

                    image = self.preprocess(Image.open(value[0])).cuda()
                    image = image.unsqueeze(0)

                    image_features, image_tokens = model.model.visual(image.type(model.model.dtype))

                    capacity_factor_train = args.capacity_factor_train
                    capacity_factor_infer = args.capacity_factor_infer

                    tokens = image_tokens.squeeze(0)  # 已经在 GPU 上

                    assignments_train, centroids_train, cluster_sizes_train, lambdas_train = balanced_kmeans_lagrangian(
                        tokens,
                        capacity=capacity_factor_train,
                        num_clusters=args.num_experts,
                        max_iters=100,
                        eta=0.0001,
                        tol=1e-4,
                        reinit_strategy='farthest'
                    )
                    assignments_infer, centroids_infer, cluster_sizes_infer, lambdas_infer = balanced_kmeans_lagrangian(
                        tokens,
                        capacity=capacity_factor_infer,
                        num_clusters=args.num_experts,
                        max_iters=100,
                        eta=0.0001,
                        tol=1e-4,
                        reinit_strategy='farthest'
                    )

                    weight_train = expert_weights(centroids_train, self.key_vectors.cuda(), cluster_sizes_train)
                    weight_infer = expert_weights(centroids_infer, self.key_vectors.cuda(), cluster_sizes_infer)
                    weight_train = weight_train.half().cuda()
                    weight_infer = weight_infer.half().cuda()
                    weight_train_list.append(weight_train)
                    weight_infer_list.append(weight_infer)

                torch.save(weight_train_list, os.path.join(args.weight_dir, args.dataset, 'weight_train_' + str(
                    args.domains[domain_index]) + '_' + 'expert_num_' + str(
                    args.num_experts) + '_' + str(capacity_factor_train) + '_'+args.init_method +'.pt'))
                torch.save(weight_infer_list, os.path.join(args.weight_dir, args.dataset, 'weight_infer_' + str(
                    args.domains[domain_index]) + '_' + 'expert_num_' + str(
                    args.num_experts) + '_' + str(capacity_factor_infer) + '_' + str(capacity_factor_train) + '_'+args.init_method + '.pt'))


    def __getitem__(self, index):
        image, label, weight_expert_train, weight_expert_infer = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))

        return image, label, weight_expert_train, weight_expert_infer

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'office-home': 'img_union','pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'pamap': 'pamap', 'covid': 'covid','domain_net': 'img_union'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getfeadataloader_few_shot(args,model):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []

    for i, item in enumerate(args.domains):
        data = ImageTextData(
                 item, args.root_dir + args.dataset + '/', i,model,args)

    return trd, vad, ted

def img_union(args, model):
    trd, vad, ted = getfeadataloader_few_shot(args, model)
    return trd, vad, ted

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pacs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='E:/data/Domainbed/')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')

    parser.add_argument('--optimizers', type=str, default='SGD', help='SGD or Adam')

    parser.add_argument('--backbone_name', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--WARMUP_EPOCH', type=int, default=5)
    parser.add_argument('--WARMUP_CONS_LR', type=float, default=1e-05)

    #   Default settings for MaPLe,Coop and CoCoop
    parser.add_argument('--N_CTX', type=int, default=2)  # prompt length
    parser.add_argument('--CTX_INIT', type=str, default='')
    parser.add_argument('--INPUT_SIZE', type=tuple, default=(224, 224))
    parser.add_argument('--PROMPT_DEPTH', type=int, default=0)
    parser.add_argument('--PROMPT_DEPTH_VISION', type=int, default=0)
    parser.add_argument('--N_CTX_VISION', type=int, default=0)
    # CoOp
    parser.add_argument('--CTP', type=str, default='end', help='end or middle')
    parser.add_argument('--CSC', type=str, default=False, help='True or False')
    parser.add_argument('--num_shots', type=int, default=0, help='True or False')
    parser.add_argument('--num_experts', type=int, default=4, help='True or False')
    parser.add_argument('--dim_k', type=int, default=16, help='True or False')
    parser.add_argument('--tau', type=float, default=0.1, help='True or False')
    parser.add_argument('--route_len', type=int, default=4, help='length  of route prompts')
    parser.add_argument('--route_emb', type=int, default=512, help='length  of route prompts')
    parser.add_argument('--hidden_emb', type=int, default=768, help='length  of route prompts')
    parser.add_argument('--gamma', type=float, default=0.05, help='length  of route prompts')
    parser.add_argument('--beta', type=float, default=0.2, help='length  of route prompts')
    parser.add_argument('--lamba', type=float, default=0.1, help='length  of route prompts')
    parser.add_argument('--capacity_factor_train', type=float, default=1.2, help='length  of route prompts')
    parser.add_argument('--capacity_factor_infer', type=float, default=2.0, help='length  of route prompts')
    parser.add_argument('--cluster', type=bool, default=True, help='Whether to cluster image tokens')
    parser.add_argument('--weight_dir', type=str, default="E:\FedDG\\fedclip\weights\\",
                        help='Whether to cluster image tokens')
    parser.add_argument('--alpha', type=float, default=0.1, help='length  of route prompts')
    parser.add_argument('--init_method', type=str, default="orthogonal")
    # parser.add_argument('--init_method', type=str, default="rand_U")

    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    args = img_param_init(args)

    os.makedirs('../data/', exist_ok=True)

    class_names = get_classname(args)
    args.classnames = class_names

    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    args.design_details = design_details

    # build the server model
    clip_model = load_clip_to_cpu(args)
    server_model = CustomCLIP(args, class_names, clip_model)
    server_model.to(device=device)

    # for init in ['rand_U','rand_N','rand_01']:
    #     args.init_method = init
    #     print("init_method: ",args.init_method)
    #     args.key_vectors = server_model.orthogonal_vectors
    #
    #     fedclip = ClipModelat(
    #         args.backbone_name, imgadpy=False, freezepy=True)
    #     get_data(args.dataset)(args, fedclip)

    args.key_vectors = server_model.orthogonal_vectors
    fedclip = ClipModelat(
        args.backbone_name, imgadpy=False, freezepy=True)
    get_data(args.dataset)(args, fedclip)
