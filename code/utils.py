import shutil
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import random

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names

def save_checkpoint(state, save_folder, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(save_folder, filename)
    bestfile = os.path.join(save_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')

def assign_learning_rate(optimizer, new_lr, tgt_group_idx=None):
    for group_idx, param_group in enumerate(optimizer.param_groups):
        if tgt_group_idx is None or tgt_group_idx==group_idx:
            param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps, tgt_group_idx=None):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr, tgt_group_idx)
        return lr
    return _lr_adjuster

def warmup_lr(optimizer, base_lr, warmup_length, tgt_group_idx=None):
    # constant lr after warmup
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr, tgt_group_idx)
        return lr
    return _lr_adjuster

def null_scheduler(init_lr):
    return lambda step:init_lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    return dict_imagenet_folder2name

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).to(labels.device)
    return y[labels]

preprocess = transforms.Compose([
    transforms.ToTensor()
])
preprocess224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
preprocess224_caltech = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])
preprocess224_interpolate = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_eval_files(dataset_name):
    # only for imaegnet and tinyimagenet
    if dataset_name == 'tinyImageNet':
        refined_data_file = "ImageNet_files/tinyimagenet_refined_labels.json"
    elif dataset_name == 'ImageNet':
        refined_data_file = "ImageNet_files/imagenet_refined_labels.json"
    refined_data = read_json(refined_data_file)
    eval_select = {ssid:refined_data[ssid]['eval_files'] for ssid in refined_data}
    return eval_select

def get_text_prompts_train(args, train_dataset, template='This is a photo of a {}'):
    class_names = train_dataset.classes
    if args.dataset == 'ImageNet':
        folder2name = load_imagenet_folder2name('ImageNet_files/imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]
    return texts_train

def get_text_prompts_val(val_dataset_list, val_dataset_name, template='This is a photo of a {}.'):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes if hasattr(each, 'classes') else each.clip_categories

            if val_dataset_name[cnt] == 'tinyImageNet':
                refined_data_file = "ImageNet_files/tinyimagenet_refined_labels.json"
            elif val_dataset_name[cnt] == 'ImageNet':
                refined_data_file = "ImageNet_files/imagenet_refined_labels.json"
            else:
                refined_data_file = None
            if refined_data_file is not None:
                refined_data = read_json(refined_data_file)
                clean_class_names = [refined_data[ssid]['clean_name'] for ssid in class_names]
                class_names = clean_class_names
                
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list

def read_json(json_file:str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def get_prompts(class_names):
    # consider using correct articles
    template = "This is a photo of a {}."
    template_v = "This is a photo of an {}."
    prompts = []
    for class_name in class_names:
        if class_name[0].lower() in ['a','e','i','o','u'] or class_name == "hourglass":
            prompts.append(template_v.format(class_name))
        else:
            prompts.append(template.format(class_name))
    return prompts

def freeze(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad=False
    return

def unfreeze(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad=True
    return

DATASETS = [
    'cifar10', 'cifar100', 'STL10', 'SUN397', 'Food101',
    'oxfordpet', 'flowers102', 'Country211', 'dtd', 'fgvc_aircraft',
    'Caltech101', 'EuroSAT', 'Caltech256', 'StanfordCars',
    'tinyImageNet', 'ImageNet'
]

def from_string(input:str, check_func=None, remove_apostrophe=True):
    # get items separated by commmas from a string
    items = []
    for item in input.split(','):
        item = item.strip(" '") if remove_apostrophe else item.strip(" ")
        if len(item)==0: continue
        if check_func is not None:
            if not check_func(item):
                raise ValueError(f'{item} not supported.')
        items.append(item)
    return items

def write_file(txt:str, file:str, mode='a'):
    with open(file, mode) as f:
        f.write(txt)

def set_tunable_params(model:torch.nn.Module, module_name:str, *submodule_names:str):
    # set specified submodules to trainbale
    target_module = getattr(model, module_name)

    if len(submodule_names) > 0: # specify certain components to train
        for sub in submodule_names:
            submod = getattr(target_module, sub)
            for p in submod.parameters(): p.requires_grad = True

    else: # set all params in target module to trainable
        for p in target_module.parameters():
            p.requires_grad = True

    return

def load_resume_file(file:str, gpu:int):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        if gpu is None:
            checkpoint = torch.load(file, weights_only=False)
        else:
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(file, map_location=loc, weights_only=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(file, checkpoint['epoch']))
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return None 

def load_checkpoints(args, model:torch.nn.Module, optimizer:torch.optim.Optimizer):
    if args.resume:
        checkpoint = load_resume_file(args.resume, args.gpu)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)

        if args.mix_alpha > 0:
            alpha = args.mix_alpha
            checkpoint_ori = torch.load('/leonardo_work/IscrC_AdvFLYP/ZSRobust/original_clip.pth.tar')
            theta_ori = checkpoint_ori['vision_encoder_state_dict']
            theta_rob = checkpoint['vision_encoder_state_dict']
            theta = {
                key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key]
                for key in theta_ori.keys()
            }
            model.module.visual.load_state_dict(theta)

        else:
            # model.module.load_state_dict(checkpoint['partial_state_dict'])
            model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

def load_checkpoints_partial(args, model:torch.nn.Module, optimizer:torch.optim.Optimizer):
    if args.resume:
        checkpoint = load_resume_file(args.resume, args.gpu)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        if args.mix_alpha > 0:
            alpha = args.mix_alpha
            checkpoint_ori = torch.load('/leonardo_work/IscrC_AdvFLYP/ZSRobust/original_clip.pth.tar')
            theta_ori = checkpoint_ori['partial_state_dict']
            theta_rob = checkpoint['partial_state_dict']
            theta = {
                key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key]
                for key in theta_ori.keys() if (key in theta_rob)
            }
            model.module.visual.load_state_dict(theta, strict=False)
        elif 'partial_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['partial_state_dict'], strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass
        else:
            model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

def sample_trainfiles(args, train_ways=None, train_ratio=None, train_shots=None):
    assert train_ratio is not None or train_shots is not None, \
    "Either specify the ratio or numbers per class of training data."
    imagenet_root = args.imagenet_root
    refined_data = read_json(args.imagenet_root + "/imagenet_refined_labels.json")
    eval_select = {ssid:refined_data[ssid]['eval_files'] for ssid in refined_data}
    if train_ways is not None:
        assert train_ways <= len(eval_select), "train_ways exceeds the number of classes."
        selected_ssids = random.sample(list(eval_select.keys()), train_ways)
        # eval_select = {ssid:eval_select[ssid] for ssid in selected_ssids}
    else:
        selected_ssids = list(eval_select.keys())
    train_select = {}
    for ssid in selected_ssids:
        remaining_files = list(set(os.listdir(imagenet_root+'/train/'+ssid))-set(eval_select[ssid]))
        if train_ratio is not None:
            n_train = int(len(remaining_files) * train_ratio)
        elif train_shots is not None:
            n_train = train_shots
        sampled_files = random.sample(remaining_files, n_train)
        train_select[ssid] = sampled_files
    return train_select