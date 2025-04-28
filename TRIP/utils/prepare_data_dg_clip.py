# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torchvision.datasets as datasets
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
    centroids = []
    # 随机选择第一个聚类中心
    first_idx = torch.randint(0, X.size(0), (1,)).item()
    centroids.append(X[first_idx])
    for _ in range(1, num_clusters):
        distances = torch.cdist(X, torch.stack(centroids))
        min_distances, _ = distances.min(dim=1)
        probabilities = min_distances ** 2
        probabilities /= probabilities.sum()
        next_centroid = X[torch.multinomial(probabilities, 1)].squeeze(0)
        centroids.append(next_centroid)
    return torch.stack(centroids)


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

    # 找到每个样本的首选聚类（调整后成本最低的聚类）
    preferred_clusters = torch.argmin(adjusted_cost, dim=1)  # (num_samples,)

    # 初始化 assignments，所有值为 -1（未分配）
    assignments = torch.full((num_samples,), -1, dtype=torch.long, device=device)

    # 复制 capacities，以便在分配过程中更新
    remaining_capacities = capacities.clone()

    for i in range(num_samples):
        k = preferred_clusters[i]
        if remaining_capacities[k] > 0:
            assignments[i] = k
            remaining_capacities[k] -= 1
        else:
            # 首选聚类已满，舍弃该样本（assignments[i] 保持为 -1）
            pass

    return assignments


def balanced_kmeans_lagrangian(X, capacity, num_clusters, max_iters=100, eta=1.0, tol=1e-4, reinit_strategy='farthest'):
    """
    使用拉格朗日乘子实现带容量约束的 K-Means 聚类算法。
    如果首选聚类容量已满，未能分配的样本将被舍弃。

    参数：
    - X: 输入数据，形状为 (num_samples, embedding_dim)
    - num_clusters: 聚类数量
    - max_iters: 最大迭代次数
    - eta: 学习率，用于更新拉格朗日乘子
    - tol: 收敛阈值
    - reinit_strategy: 空聚类重新初始化策略

    返回：
    - assignments: 每个样本的聚类分配，未分配的样本为 -1
    - centroids: 聚类中心
    - cluster_sizes: 每个聚类的样本数量
    - lambdas: 拉格朗日乘子
    """
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

        # 使用容量限制的分配算法，考虑拉格朗日乘子
        assignments = capacity_constrained_assignment(distances, lambdas, capacities)

        # 计算当前聚类大小
        assigned_mask = assignments >= 0  # 被成功分配的样本掩码
        cluster_sizes = torch.bincount(assignments[assigned_mask], minlength=num_clusters).float()

        # 更新拉格朗日乘子
        target_sizes = capacities.float()
        lambdas += eta * (cluster_sizes - target_sizes)

        # 重新计算聚类中心
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            assigned_indices = ((assignments == k) & assigned_mask).nonzero(as_tuple=True)[0]
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
                lambdas[k] = 0.0

        # 检查收敛
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).mean()
        if centroid_shift < tol:
            break

        centroids = new_centroids

    # 最终的聚类大小
    cluster_sizes = torch.bincount(assignments[assigned_mask], minlength=num_clusters).float()
    a = cluster_sizes
    return assignments, centroids, cluster_sizes, lambdas


def constrained_kmeans(X, num_clusters, max_iters=100, lambda_balance=1.0,
                       reinit_strategy='farthest'):
    """
    带有专家分配约束的自定义 K-Means 聚类算法。
    参数：
    - X: 输入数据，形状为 (num_samples, embedding_dim)
    - num_clusters: 聚类数量（专家数量）
    - max_iters: 最大迭代次数
    - lambda_balance: 负载均衡惩罚项的超参数
    - reinit_strategy: 空聚类重新初始化策略，可选 'farthest' 或 'kmeans++'

    返回：
    - assignments: 每个样本的聚类分配，形状为 (num_samples,)
    - centroids: 聚类中心，形状为 (num_clusters, embedding_dim)
    - cluster_sizes: 每个聚类的样本数量，形状为 (num_clusters,)
    """

    device = X.device
    num_samples, embedding_dim = X.shape

    # 随机初始化聚类中心
    indices = torch.randperm(num_samples)[:num_clusters]
    centroids = X[indices].clone()
    count = 0

    for iteration in range(max_iters):
        # 计算样本与聚类中心的距离
        distances = torch.cdist(X, centroids)  # (num_samples, num_clusters)

        # 如果不是第一次迭代，使用上一轮的 assignments，否则初始化
        if iteration == 0:
            assignments = torch.argmin(distances, dim=1)
        else:
            assignments = new_assignments

        # 计算当前的专家负载
        cluster_sizes = torch.bincount(assignments, minlength=num_clusters).float()

        # 计算负载均衡惩罚项
        mean_cluster_size = cluster_sizes.mean()
        load_penalty = lambda_balance * torch.pow((cluster_sizes - mean_cluster_size) / mean_cluster_size, 2)
        load_penalty = load_penalty.unsqueeze(0)

        # 将负载惩罚项添加到距离上
        total_cost = distances + load_penalty  # (num_samples, num_clusters)

        # 更新聚类分配
        new_assignments = torch.argmin(total_cost, dim=1)  # (num_samples,)

        # 检查是否收敛
        if torch.equal(new_assignments, assignments):
            count += 1
        else:
            count = 0  # 重置计数器

        if iteration > 0 and count == 3:
            # print(f"Converged at iteration {iteration}")
            break

        # 更新聚类中心
        for k in range(num_clusters):
            assigned_indices = (new_assignments == k).nonzero(as_tuple=True)[0]
            if assigned_indices.numel() > 0:
                centroids[k] = X[assigned_indices].mean(dim=0)
            else:
                # 如果某个聚类没有分配到样本，使用指定的策略重新初始化聚类中心
                if reinit_strategy == 'farthest':
                    centroids[k] = initialize_farthest_point(X, centroids)
                elif reinit_strategy == 'kmeans++':
                    centroids[k] = initialize_kmeans_pp(X, centroids)
                else:
                    # 默认随机重新初始化
                    centroids[k] = X[torch.randint(0, num_samples, (1,))]
                # print(f"Cluster {k} is empty. Reinitialized using {reinit_strategy} strategy.")

    return new_assignments, centroids, cluster_sizes


def expert_weights(centroids, key_vectors, cluster_sizes):
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

    def __init__(self, dataset, root, domain_index, model, args):
        dataset = os.path.join(root, dataset)
        data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels
        self.preprocess = model.preprocess
        self.cluster = args.cluster

        if args.cluster:
            print("OKKKKKKKKKKKKK")
            self.key_vectors = args.key_vectors
            save_train_path = os.path.join(args.weight_dir, args.dataset, 'weight_train_' + str(
                    args.domains[domain_index]) + '_' + 'expert_num_' + str(
                    args.num_experts) + '_' + str(args.capacity_factor_train) +'_'+args.init_method + '.pt')

            save_infer_path = os.path.join(args.weight_dir, args.dataset, 'weight_infer_' + str(
                    args.domains[domain_index]) + '_' + 'expert_num_' + str(
                    args.num_experts) + '_' + str(args.capacity_factor_infer) + '_' + str(args.capacity_factor_train) +'_'+args.init_method + '.pt')

            weights_train = torch.load(save_train_path)
            weights_infer = torch.load(save_infer_path)

            self.weights_train = weights_train
            self.weights_infer = weights_infer

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.cluster:
            weight_expert_train = self.weights_train[index]
            weight_expert_infer = self.weights_infer[index]
        else:
            weight_expert_train = 0
            weight_expert_infer = 0
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
    datalist = {'office-home': 'img_union', 'pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'pamap': 'pamap', 'covid': 'covid',
                'domain_net': 'img_union'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getfeadataloader_few_shot(args, model):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []

    for i, item in enumerate(args.domains):

        if i in args.test_envs:
            data = ImageTextData(
                item, args.root_dir + args.dataset + '/', i, model, args)

            ted.append(torch.utils.data.DataLoader(
                data, batch_size=args.batch, shuffle=False))
            trd.append(0)
            vad.append(0)
        else:
            data = ImageTextData(
                item, args.root_dir + args.dataset + '/', i, model, args)

            l = len(data)
            l2 = int(l * 0.2)
            if args.num_shots != 0:
                ## few-sahot settings.
                class_groups = defaultdict(list)
                class_names = data.data.classes

                for ind, value in enumerate(data.data.imgs):
                    for class_name in class_names:
                        if class_name in value[0]:
                            class_groups[class_name].append(ind)

                np.random.seed(args.seed)

                index_train_data = []
                index_valid_data = []
                num_shots = args.num_shots

                val_l2 = int(l2 / len(class_groups))
                for key in class_groups.keys():
                    np.random.shuffle(class_groups[key])
                    index_train_data.append(class_groups[key][:num_shots])
                    index_valid_data.append(class_groups[key][num_shots:2 * num_shots])
                index_train_data = np.array([item for sublist in index_train_data for item in sublist])
                index_valid_data = np.array([item for sublist in index_valid_data for item in sublist])

                np.random.shuffle(index_train_data)
                np.random.shuffle(index_valid_data)

            else:
                l = len(data)
                index = np.arange(l)

                np.random.seed(args.seed)
                np.random.shuffle(index)

                l1, l2, l3 = int(l * 0.8), int(l * 0.2), int(l * 0)
                index_train_data = index[:l1]
                index_valid_data = index[l1:l1 + l2]

            trl.append(torch.utils.data.Subset(data, index_train_data))
            val.append(torch.utils.data.Subset(data, index_valid_data))

            trd.append(torch.utils.data.DataLoader(
                trl[-1], batch_size=args.batch, shuffle=True))
            vad.append(torch.utils.data.DataLoader(
                val[-1], batch_size=args.batch, shuffle=False))
            ted.append(0)

    return trd, vad, ted


def img_union(args, model):
    trd, vad, ted = getfeadataloader_few_shot(args, model)
    return trd, vad, ted
