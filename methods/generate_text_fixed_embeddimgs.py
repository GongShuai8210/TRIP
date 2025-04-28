# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
import argparse
from nets.models import ClipModelat
from torch.utils.tensorboard import SummaryWriter
from clip import clip
import torchvision.datasets as datasets
from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import os
import pickle
from nets.imagenet_templates import  IMAGENET_TEMPLATES
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='domain_net')
    parser.add_argument('--lr1', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr2', type=float, default=0.005, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='E:/FedDG/data/')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--backbone_name', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=4)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--optimizers', type=str, default='SGD', help='Adam or SGD')

    parser.add_argument('--WARMUP_EPOCH', type=int, default=1)
    parser.add_argument('--WARMUP_CONS_LR', type=float, default=1e-05)
    #   Default settings for MaPLe,Coop and CoCoop
    parser.add_argument('--N_CTX', type=int, default=8)  # prompt length
    parser.add_argument('--CTX_INIT', type=str, default='a picture of a')
    parser.add_argument('--INPUT_SIZE', type=tuple, default=(224, 224))
    parser.add_argument('--PROMPT_DEPTH', type=int, default=12)
    parser.add_argument('--PROMPT_DEPTH_VISION', type=int, default=12)
    parser.add_argument('--N_CTX_VISION', type=int, default=8)
    parser.add_argument('--N_CTX_TEXT', type=int, default=8)
    parser.add_argument('--PROMPT_DEPTH_TEXT', type=int, default=12)
    parser.add_argument('--cos_scale', type=int, default=1)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--if_loss_domain', type=bool, default=False)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--num_tarin_client_prompts', type=int, default=5)
    parser.add_argument('--num_tarin_adapters', type=int, default=5)

    parser.add_argument('--a_t', type=float, default=0.002)

    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    args = img_param_init(args)
    os.makedirs('../data/', exist_ok=True)

    FedCLIP = ClipModelat(
        args.backbone_name, imgadpy=True, freezepy=True)

    # Copyright (c) Microsoft Corporation.
    # Licensed under the MIT License.



    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_path)

    ImageFile.LOAD_TRUNCATED_IMAGES = True


    class ImageTextData(object):

        def __init__(self, dataset, root, preprocess, prompt='a picture of a'):
            dataset = os.path.join(root, dataset)
            data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
            labels = data.classes
            self.data = data
            self.labels = labels
            if prompt:
                self.labels = [prompt + ' ' + x for x in self.labels]

            self.preprocess = preprocess
            self.text = clip.tokenize(self.labels)

        def __getitem__(self, index):
            image, label = self.data.imgs[index]
            if self.preprocess is not None:
                image = self.preprocess(Image.open(image))
            text_enc = self.text[label]
            return image, text_enc, label

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


    # def get_data(data_name):
    #     datalist = {'pacs': 'pacs'}
    #     if datalist[data_name] not in globals():
    #         raise NotImplementedError("Dataset not found: {}".format(data_name))
    #     return globals()[datalist[data_name]]

    # 8210
    def get_data(data_name):
        """Return the algorithm class with the given name."""
        datalist = {'office-home': 'img_union', 'pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                    'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'pamap': 'pamap', 'domain_net': 'img_union'}
        if datalist[data_name] not in globals():
            raise NotImplementedError("Algorithm not found: {}".format(data_name))
        return globals()[datalist[data_name]]

    def getfeadataloader(args, model):
        trl, val, tel = [], [], []
        trd, vad, ted = [], [], []
        for i, item in enumerate(args.domains):


            data = ImageTextData(
                item, args.root_dir + args.dataset + '/', model.preprocess)
            l = len(data)
            index = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(index)
            l1, l2, l3 = int(l * 0.8), int(l * 0.2), int(l * 0)
            trl.append(torch.utils.data.Subset(data, index[:l1]))
            trd.append(torch.utils.data.DataLoader(
                trl[-1], batch_size=4, shuffle=True))

        return trd

    def img_union(args, model):
        trd = getfeadataloader(args, model)
        return trd


    def load_clip_to_cpu(args, zero_shot_model=False):
        backbone_name = args.backbone_name
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root_dir)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        if not zero_shot_model:
            design_details = {"trainer": 'IVLP',
                              "vision_depth": 0,
                              "language_depth": 0, "vision_ctx": 0,
                              "language_ctx": 0,
                            }

            model = clip.build_model(state_dict or model.state_dict(), design_details)
        else:
            # Return original CLIP model for generating frozen VL features
            design_details = {"trainer": 'IVLP',
                              "vision_depth": 0,
                              "language_depth": 0, "vision_ctx": 0,
                              "language_ctx": 0,
                              }
            model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model

    # file_path = "/home/gongshuai/code/FedDG/text_embedding/" + args.dataset + "/"
    file_path = "E:/FedDG/text_embedding/"

    os.makedirs(file_path,exist_ok=True)
    print("Starting generating {} text embeddings,save in {}".format(args.dataset,file_path))
    train_loaders = get_data(
        args.dataset)(args, FedCLIP)

    # build server model
    class_names = train_loaders[0].batch_sampler.sampler.data_source.dataset.data.classes
    num_class = len(class_names)


    clip_model_temp = load_clip_to_cpu(args, True).float().cuda()
    os.makedirs(file_path, exist_ok=True)

    all_teacher_features = []
    # Using multiple text templates to ensure textual diversity during training
    with torch.no_grad():
        for single_template in tqdm(IMAGENET_TEMPLATES):
            classnames = [name.replace("_", " ") for name in class_names]
            x = [single_template.replace("{}", name) for name in classnames]
            x_tokenized = torch.cat([clip.tokenize(p) for p in x])
            text_features = clip_model_temp.encode_text(x_tokenized.cuda())
            all_teacher_features.append(text_features.unsqueeze(1))
    fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

    domain_name = args.dataset +".pkl"
    prototype_file = file_path + domain_name
    with open(prototype_file, 'wb') as ff:
        pickle.dump(fixed_embeddings, ff)
    print('Making fix embedding for dataset {}  finished!!'.format(args.dataset))


# def prototype_load(file_name):
#
#     with open(file_name, "rb") as file:
#         prototype = pickle.load(file)
#
#     return prototype
#
# file_name = file_path + '2.pkl'
#
# prototype_load(file_name)