import logging
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from utils.datasets import CIFAR100_truncated, CIFAR10_truncated
import random
import copy
import torch.optim as optim
import torch.nn as nn, torch.nn.functional as F
from utils.prepare_data_dg_clip import *

# from pypapi import events, papi_high as high

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(args,datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(args,datadir,cluster=False, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(args,datadir,cluster=False, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


#     return protos
def fine_to_coarse(fine_label):
    label_np = fine_label.cpu().detach().numpy()
    label_change = np.zeros_like(label_np)
    coarse_labels = \
        np.array([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ])
    for i in range(label_change.shape[0]):
        label_change[i] = coarse_labels[label_np[i]]
    return label_change


def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None, args=None):
    if dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(args,datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    if partition == "noniid-labeluni":
        if "cifar100" == dataset:
            num = args.cls_num
            K = 100
        elif "cifar10" == dataset:
            num = args.cls_num
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i],
                                                         data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i],
                                                        data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]
    else:
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]

        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):

                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))

                ## Balance
                proportions = np.array(
                    [p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])

                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]

                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in
                                   zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts,
            testdata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", args=None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                if args is not None and args.uppper_coarse:
                    output = model(x, prompt_index=fine_to_coarse(target))
                else:
                    output = model(x)
                out = output['logits']
                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)


def compute_accuracy_loss(model, dataloader, device="cpu", args=None):
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
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                if args is not None and args.uppper_coarse:
                    output = model(x, prompt_index=fine_to_coarse(target))
                    out = output['logits']
                else:
                    output = model(x)
                    out = output['logits']
                _, pred_label = torch.max(out.data, 1)
                loss = criterion(out, target)
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

    return correct, total, total_loss / batch_count


def compute_accuracy_local(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = 0
        _, test_dl_local, _, _ = get_divided_dataloader(args, dataidxs_train, dataidxs_test, noise_level)
        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        test_results[net_id]['loss'] = test_avg_loss
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 255)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_divided_dataloader(args, dataidxs_train, dataidxs_test, noise_level=0, drop_last=False, apply_noise=False,
                           traindata_cls_counts=None):
    dataset = args.dataset
    datadir = args.datadir
    train_bs = args.batch
    test_bs = 4 * args.batch

    if 'cifar100' == dataset:
        dl_obj = CIFAR100_truncated
        if apply_noise:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                GaussianNoise(0., noise_level)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                GaussianNoise(0., noise_level)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=BICUBIC),
                CenterCrop(args.img_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=BICUBIC),
                CenterCrop(args.img_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    if 'cifar10' == dataset:
        dl_obj = CIFAR10_truncated
        if apply_noise:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                GaussianNoise(0., noise_level)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                GaussianNoise(0., noise_level)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=BICUBIC),
                CenterCrop(args.img_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size), interpolation=BICUBIC),
                CenterCrop(args.img_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    traindata_cls_counts = traindata_cls_counts
    train_ds = dl_obj(args,datadir,cluster=args.cluster,dataidxs=dataidxs_train, train=True, transform=transform_train, download=False,
                      return_index=True)
    test_ds = dl_obj(args,datadir,cluster=args.cluster, dataidxs=dataidxs_test, train=False, transform=transform_test, download=False,
                     return_index=False)

    conditioned_loader = {}
    if traindata_cls_counts is not None:
        for cls_id in traindata_cls_counts.keys():
            cnd_ds = dl_obj(args,datadir,cluster=args.cluster, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False,
                            cls_condition=cls_id)
            train_dl = data.DataLoader(dataset=cnd_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
            conditioned_loader[cls_id] = train_dl

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds, conditioned_loader, traindata_cls_counts
