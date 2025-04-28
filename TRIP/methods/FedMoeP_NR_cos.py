# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import time
from utils.config import img_param_init, set_random_seed, get_classname
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
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
scaler = GradScaler()

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

def pkl_load(file_name):

    with open(file_name, "rb") as file:
        prototype = pickle.load(file)

    return prototype

def train_CoOp(args, model, data_loader, optimizer, scheduler, device):

    model.train()
    # count = 0
    loss_value = 0
    batch_count = 0

    for batch in tqdm(data_loader):
        image, label, weight_expert_train, weight_expert_infer = batch
        image = image.to(device)
        label = label.to(device)
        weight_expert_train = weight_expert_train.to(device)

        loss = model(image, weight_expert_train, label, Training=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        batch_count += 1

    if scheduler == 0:
        pass
    else:
        scheduler.step()
    for param_group in optimizer.param_groups:
        print('scheduler learning rate: ', param_group['lr'])
    print('current loss: ', loss_value / batch_count)



def test(args, model, data_loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in data_loader:
            image, label, weight_expert_train, weight_expert_infer = batch
            image = image.to(device)
            label = label.to(device)
            weight_expert_infer = weight_expert_infer.to(device)

            y_true = []
            y_pred = []

            output = model(image, weight_expert_infer)
            pred = output.max(1)[1]
            matches = pred.eq(label).float()
            correct += int(matches.sum().item())
            total += label.shape[0]

            acc = 100.0 * correct / total

            # err = 100.0 - acc
            # macro_f1 = 100.0 * f1_score(
            #     y_true,
            #     y_pred,
            #     average="macro",
            #     labels=np.unique(y_true)
            # )

        return acc


def communication(args, server_model, models, client_weights):
    client_num = len(models)

    with torch.no_grad():
        for key in server_model.prompt_learner.state_dict().keys():

            temp = torch.zeros_like(server_model.prompt_learner.state_dict()[
                                        key], dtype=torch.float16)

            for client_idx in range(client_num):
                if client_idx not in args.test_envs:
                    temp += client_weights[client_idx] * \
                            models[client_idx].prompt_learner.state_dict()[key]

            server_model.prompt_learner.state_dict()[key].data.copy_(temp)
            for client_idx in range(client_num):
                if client_idx not in args.test_envs:
                    models[client_idx].prompt_learner.state_dict()[key].data.copy_(
                        server_model.prompt_learner.state_dict()[key])
    return server_model, models


from math import cos, pi


def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):
    if current_epoch < warmup_epoch:

        warmup_factor = (lr_max / lr_min) ** (1.0 / warmup_epoch)
        lr = lr_min * (warmup_factor ** current_epoch)  # 指数增长的学习率

        # lr = lr_max * current_epoch / warmup_epoch
        # lr = lr_min

    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pacs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--root_dir', type=str, default='E:/data/Domainbed/')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')

    parser.add_argument('--optimizers', type=str, default='SGD', help='SGD or Adam')

    parser.add_argument('--backbone_name', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--WARMUP_EPOCH', type=int, default=5)
    parser.add_argument('--WARMUP_CONS_LR', type=float, default=5e-4)

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
    parser.add_argument('--beta', type=float, default=0.2, help='length  of route prompts')
    parser.add_argument('--lamba', type=float, default=0.1, help='length  of route prompts')
    parser.add_argument('--capacity_factor_train', type=float, default=1.0, help='length  of route prompts')
    parser.add_argument('--capacity_factor_infer', type=float, default=2.0, help='length  of route prompts')
    parser.add_argument('--cluster', type=bool, default=True, help='Whether to cluster image tokens')
    parser.add_argument('--weight_dir', type=str, default="E:\FedDG\\fedclip\weight_dir\\",
                        help='Whether to cluster image tokens')
    parser.add_argument('--data_seed', type=int, default=0, help='random seed')
    parser.add_argument('--gamma', type=float, default=0.80)
    parser.add_argument('--ema', type=float, default=0.99)
    parser.add_argument('--text_embedding_path', type=str, default="E:/FedDG/text_embedding/")
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--move', type=str, default="beta")
    parser.add_argument('--init_method', type=str, default="rand_N")



    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    args = img_param_init(args)

    os.makedirs('../data/', exist_ok=True)

    # key_vectors = torch.empty(args.num_experts, 768)
    # nn.init.orthogonal_(key_vectors, gain=1.0)
    # args.key_vectors = key_vectors.cuda()

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
    args.key_vectors = server_model.orthogonal_vectors
    fedclip = ClipModelat(
        args.backbone_name, imgadpy=False, freezepy=True)
    train_loaders, val_loaders, test_loaders = get_data(
        args.dataset)(args, fedclip)

    #  build the client models
    client_num = len(test_loaders)
    data_size = [0 for _ in range(client_num)]
    for i in range(client_num):
        if i == args.test_envs[0]:
            pass
        else:
            data_size[i] = len(train_loaders[i].sampler.data_source.dataset.weights_train)
    total_data = sum(data_size)
    client_weights = [data_size[i] / total_data for i in range(client_num)]
    models = [copy.deepcopy(server_model) for idx in range(client_num)]

    beta_dist = sp.stats.beta(args.beta, args.beta)
    total_iter = args.iters
    weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))
    server_model_bma = copy.deepcopy(server_model)
    server_model_bma.to(device)

    for i in range(client_num):
        for name, param in models[i].named_parameters():
            if "VPT" not in name:
                param.requires_grad_(False)
        models[i].to(device)
        # Double check
    param_dict = OrderedDict()
    for i in range(client_num):
        param_dict[i] = []
        for name, param in models[i].named_parameters():
            if param.requires_grad:
                param_dict[i].append(name)
    # print(f"Parameters to be updated: {param_dict}")

    best_changed = False

    best_acc = [0. for j in range(client_num)]
    finalrecord = ''
    logrecord = ''
    count = 0

    weight_sum = weight_func(0)
    schedulers = [0 for i in range(client_num)]
    total_epochs = 0

    for a_iter in range(args.iters):

        start_time = time.time()
        if args.optimizers == 'Adam':
            optimizers = [optim.Adam(params=[{'params': models[idx].prompt_learner.parameters()}], lr=args.lr, betas=(
                args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
        elif args.optimizers == 'AdamW':

            optimizers = [
                optim.AdamW(params=[{'params': models[idx].prompt_learner.VPTctxList.parameters()}], lr=args.lr, betas=(
                    args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in
                range(client_num)]
        else:
            optimizers = [optim.SGD(
                [{'params': models[idx].prompt_learner.parameters()}],
                lr=args.lr,  # 0.002
                momentum=0.9,
                weight_decay=args.weight_decay,
                dampening=0,
                nesterov=False,
            ) for idx in range(client_num)]
        adjusted_lrs = [args.lr * (args.gamma ** total_epochs) for _ in range(client_num)]

        for idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lrs[idx]

        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):

                text_embedding_path = args.text_embedding_path + args.dataset + ".pkl"
                model.fixed_embeddings = pkl_load(text_embedding_path)
                model.global_experts = server_model.prompt_learner.VPTctxList
                if client_idx in args.test_envs:
                    pass
                else:
                    train_CoOp(
                        args, model, train_loaders[client_idx], optimizers[client_idx], schedulers[client_idx], device)

        with torch.no_grad():
            print_color_text("============Communicating with the server============", 'green')

            server_model, models = communication(
                args, server_model, models, client_weights)

            if args.move == "beta":

                weight = weight_func(a_iter)
                relative_weight = weight / weight_sum
                for moving_avg_param, param in zip(server_model_bma.prompt_learner.VPTctxList.parameters(),
                                                   server_model.prompt_learner.VPTctxList.parameters()):
                    moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)

                weight_sum += weight
            else:

                for moving_avg_param, param in zip(server_model_bma.prompt_learner.VPTctxList.parameters(),
                                                   server_model.prompt_learner.VPTctxList.parameters()):
                    moving_avg_param.data = args.ema * moving_avg_param.data + (1-args.ema) * param.data

            # for moving_avg_param, param in zip(server_model_bma.prompt_learner.VPTctxList.parameters(),
            #                                    server_model.prompt_learner.VPTctxList.parameters()):
            #     moving_avg_param.data = 0.5 * moving_avg_param.data + 0.5 * param.data

            total_epochs += 1

            val_acc_list = [0. for j in range(client_num)]

            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    # val_acc = test(
                    #     args, server_model, val_loaders[client_idx], device)
                    val_acc = test(
                        args, server_model_bma, val_loaders[client_idx], device)
                    val_acc_list[client_idx] = val_acc
                    print(' Site-{:d}| Val  Acc: {:.4f}'.format(
                        client_idx, val_acc), flush=True)
                    logrecord += ' Site-{:d}| Val  Acc: {:.4f}\n'.format(
                        client_idx, val_acc)

            test_acc_list = [0. for j in range(client_num)]
            test_acc_list_bma = [0. for j in range(client_num)]

            for client_idx in range(client_num):
                if np.mean(val_acc_list) > np.mean(best_acc):

                    if client_idx in args.test_envs:
                        test_acc = test(args, server_model,
                                        test_loaders[client_idx], device)
                        test_acc_bma = test(args, server_model_bma,
                                            test_loaders[client_idx], device)
                    else:
                        test_acc = 0
                        test_acc_bma = 0
                else:
                    test_acc = 0
                    test_acc_bma = 0
                print(
                    ' Test site-{:d}| Test Acc: {:.4f}'.format(client_idx, test_acc))
                logrecord += ' Test site-{:d}| Test Acc: {:.4f}'.format(
                    client_idx, test_acc)
                print(
                    ' Test site-{:d}| Test Acc: {:.4f}'.format(client_idx, test_acc_bma))
                logrecord += ' Test site-{:d}| Test Acc: {:.4f}'.format(
                    client_idx, test_acc_bma)
                test_acc_list[client_idx] = test_acc
                test_acc_list_bma[client_idx] = test_acc_bma

            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    if client_idx in args.test_envs:
                        pass
                    else:
                        best_acc[client_idx] = val_acc_list[client_idx]
                        best_communication_epoch = a_iter
                        best_changed = True
            print(f"valid mean accuracy: {np.mean(val_acc_list)}")

            if best_changed:
                finalrecord = finalrecord + str(a_iter) + ','
                for item in test_acc_list:
                    finalrecord = finalrecord + str(item) + ','
                for item in test_acc_list_bma:
                    finalrecord = finalrecord + str(item) + ','
                best_changed = False
        end_time = time.time()
        print("One epoc takes time: {:.2f}s".format(end_time - start_time))
    print('best epoch:%d\n' % (best_communication_epoch))
    print_color_text('test_envs {}'.format(args.test_envs), 'blue')

    logrecord += '\n best epoch:%d\n' % (best_communication_epoch)
    rec = finalrecord.split(',')[-9:-1]
    ts = ''
    for item in rec:
        ts += '%.4f ' % float(item)
    print('best test acc: ' + ts)
    logrecord += 'best test acc: ' + ts
    filename = args.root_dir + '/results/FedMoep_topk/' + args.dataset + '_'  'lr' + str(
        args.lr) + '_' + '_lamba_' + str(args.lamba)
    filename = filename + '_' + args.backbone_name
    os.makedirs(filename, exist_ok=True)
    with open(filename + '/output' '.txt', 'w') as f:
        f.write(finalrecord)
    with open(filename + '/log' + str(
            args.test_envs[0]) + '.txt', 'w') as f:
        f.write(logrecord)

    save_path = os.path.join(args.root_dir, args.dataset)
    save_path = os.path.join(save_path, str(args.test_envs[0]))

    if not os.path.exists(save_path):
        print("文件夹不存在，创建！")
        os.makedirs(save_path)
    else:
        print("文件夹已创建！")

    save_model =  save_path + '/sever_model_bma.pt'
    torch.save(server_model_bma,save_model)
