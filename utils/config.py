# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch



def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'vlcs':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'dg4':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    elif dataset == 'terra':
        domains = ['location_38', 'location_43', 'location_46', 'location_100']
    elif dataset == 'domain_net':
        domains = ["clipart", "infograph",
                   "painting", "quickdraw", "real", "sketch"]
    else:
        domains = None
    args.domains = domains

    if args.dataset == 'office-home':
        args.num_classes = 65
    elif args.dataset == 'office':
        args.num_classes = 31
    elif args.dataset == 'pacs':
        args.num_classes = 7
    elif args.dataset == 'vlcs':
        args.num_classes = 5
    elif args.dataset == 'terra':
        args.num_classes = 10
    elif args.dataset == 'domain_net':
        args.num_classes = 345
    else:
        args.num_classes = 4
    return args


#
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_classname(args):
    if args.dataset == "pacs":
        class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    elif args.dataset == "office-home":
        class_names = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                       'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                       'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                       'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker',
                       'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
                       'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors',
                       'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table',
                       'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

    elif args.dataset == "vlcs":
        class_names = ['bird', 'car', 'chair', 'dog', 'person']

    elif args.dataset == 'domain_net':  # domain_net
        class_names = ["The_Eiffel_Tower", "The_Great_Wall_of_China", "The_Mona_Lisa", "aircraft_carrier", "airplane",
                       "alarm_clock", "ambulance", "angel", "animal_migration", "ant", "anvil", "apple", "arm",
                       "asparagus", "axe", "backpack", "banana", "bandage", "barn", "baseball", "baseball_bat",
                       "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", "bee", "belt",
                       "bench", "bicycle", "binoculars", "bird", "birthday_cake", "blackberry", "blueberry", "book",
                       "boomerang", "bottlecap", "bowtie", "bracelet", "brain", "bread", "bridge", "broccoli", "broom",
                       "bucket", "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator", "calendar",
                       "camel", "camera", "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot",
                       "castle", "cat", "ceiling_fan", "cell_phone", "cello", "chair", "chandelier", "church", "circle",
                       "clarinet", "clock", "cloud", "coffee_cup", "compass", "computer", "cookie", "cooler", "couch",
                       "cow", "crab", "crayon", "crocodile", "crown", "cruise_ship", "cup", "diamond", "dishwasher",
                       "diving_board", "dog", "dolphin", "donut", "door", "dragon", "dresser", "drill", "drums", "duck",
                       "dumbbell", "ear", "elbow", "elephant", "envelope", "eraser", "eye", "eyeglasses", "face", "fan",
                       "feather", "fence", "finger", "fire_hydrant", "fireplace", "firetruck", "fish", "flamingo",
                       "flashlight", "flip_flops", "floor_lamp", "flower", "flying_saucer", "foot", "fork", "frog",
                       "frying_pan", "garden", "garden_hose", "giraffe", "goatee", "golf_club", "grapes", "grass",
                       "guitar", "hamburger", "hammer", "hand", "harp", "hat", "headphones", "hedgehog", "helicopter",
                       "helmet", "hexagon", "hockey_puck", "hockey_stick", "horse", "hospital", "hot_air_balloon",
                       "hot_dog", "hot_tub", "hourglass", "house", "house_plant", "hurricane", "ice_cream", "jacket",
                       "jail", "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern", "laptop", "leaf",
                       "leg", "light_bulb", "lighter", "lighthouse", "lightning", "line", "lion", "lipstick", "lobster",
                       "lollipop", "mailbox", "map", "marker", "matches", "megaphone", "mermaid", "microphone",
                       "microwave", "monkey", "moon", "mosquito", "motorbike", "mountain", "mouse", "moustache",
                       "mouth", "mug", "mushroom", "nail", "necklace", "nose", "ocean", "octagon", "octopus", "onion",
                       "oven", "owl", "paint_can", "paintbrush", "palm_tree", "panda", "pants", "paper_clip",
                       "parachute", "parrot", "passport", "peanut", "pear", "peas", "pencil", "penguin", "piano",
                       "pickup_truck", "picture_frame", "pig", "pillow", "pineapple", "pizza", "pliers", "police_car",
                       "pond", "pool", "popsicle", "postcard", "potato", "power_outlet", "purse", "rabbit", "raccoon",
                       "radio", "rain", "rainbow", "rake", "remote_control", "rhinoceros", "rifle", "river",
                       "roller_coaster", "rollerskates", "sailboat", "sandwich", "saw", "saxophone", "school_bus",
                       "scissors", "scorpion", "screwdriver", "sea_turtle", "see_saw", "shark", "sheep", "shoe",
                       "shorts", "shovel", "sink", "skateboard", "skull", "skyscraper", "sleeping_bag", "smiley_face",
                       "snail", "snake", "snorkel", "snowflake", "snowman", "soccer_ball", "sock", "speedboat",
                       "spider", "spoon", "spreadsheet", "square", "squiggle", "squirrel", "stairs", "star", "steak",
                       "stereo", "stethoscope", "stitches", "stop_sign", "stove", "strawberry", "streetlight",
                       "string_bean", "submarine", "suitcase", "sun", "swan", "sweater", "swing_set", "sword",
                       "syringe", "t-shirt", "table", "teapot", "teddy-bear", "telephone", "television",
                       "tennis_racquet", "tent", "tiger", "toaster", "toe", "toilet", "tooth", "toothbrush",
                       "toothpaste", "tornado", "tractor", "traffic_light", "train", "tree", "triangle", "trombone",
                       "truck", "trumpet", "umbrella", "underwear", "van", "vase", "violin", "washing_machine",
                       "watermelon", "waterslide", "whale", "wheel", "windmill", "wine_bottle", "wine_glass",
                       "wristwatch", "yoga", "zebra", "zigzag"]
    elif args.dataset == "cifar10":
        label_map = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                     8: 'ship',
                     9: 'truck'}
        class_names = list(label_map.values())
    elif args.dataset == "cifar100":
        label_map = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle',
                     8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly',
                     15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair',
                     21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab',
                     27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish',
                     33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard',
                     40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man',
                     47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree',
                     53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck',
                     59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum',
                     65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea',
                     72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake',
                     79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table',
                     85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train',
                     91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf',
                     98: 'woman', 99: 'worm'}
        class_names = list(label_map.values())
    else:
        print("Please prepare the dataset in Readme.md.")

    return class_names
