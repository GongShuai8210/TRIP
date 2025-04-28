
import numpy as np
import copy
from collections import  defaultdict
import torch, torch.nn as nn

def aggregation_func_avg(keys_dict,global_para,selected,fed_avg_freqs,group_ratio,args):
    unique_dict = {}
    # unique_dict['prompt_keys'] = copy.deepcopy(global_para['prompt_keys'])
    for idx,r in enumerate(selected):
        net_para = keys_dict[r]
        if idx == 0:
            for key in net_para:
                global_para[key] = copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
        else:
            for key in net_para:
                global_para[key] += copy.deepcopy(net_para[key]) * fed_avg_freqs[idx]
    return global_para

def compute_accuracy_our(global_model,data_loader_dict,args):
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        global_model.eval()
        if net_id not in data_loader_dict.keys():
            continue
        test_dl_local = data_loader_dict[net_id]['test_dl_local']
        # traindata_cls_count = data_loader_dict[net_id]['traindata_cls_count']
        test_correct, test_total, test_avg_loss = compute_accuracy_loss_our(global_model, test_dl_local, device=args.device,args = args)
        test_results[net_id]['loss'] = test_avg_loss
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    #### global performance
    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    ### local performance
    local_mean_acc = np.mean([val['correct']/val['total'] for val in test_results.values()])
    local_min_acc = np.min([val['correct']/val['total'] for val in test_results.values()])

    return  test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc

def compute_accuracy_loss_our(model, dataloader, device="cpu",prototype = None,args=None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target, weights_train_t, weights_infer_t) in enumerate(tmp):

                x, target = x.to(args.device), target.to(args.device)

                weights_infer_t = weights_train_t.to(args.device)

                x, target = x.to(device), target.to(device,dtype=torch.int64)
                target = target.long()
                weights_infer_t = weights_infer_t.to(args.device)
                logits = model(x,weights_infer_t)
                _, pred_label = torch.max(logits.data, 1)
                loss = criterion(logits, target)
                correct += (pred_label == target.data).sum().item()
                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count