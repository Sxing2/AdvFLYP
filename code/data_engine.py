import os
import json
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from code.replace.datasets import caltech, country211, dtd, eurosat, fgvc_aircraft, food101, \
                                         flowers102, oxford_iiit_pet, pcam, stanford_cars, sun397, \
                                         ImageNet_Sketch, ImageNet_a, ImageNet_r, ObjectNet
from code.replace.datasets.folder import ImageNetFolder

from code.utils import preprocess224, preprocess224_caltech, get_eval_files, read_json
from PIL import Image

class ImageNetCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, caption

class smallLAION(Dataset):
    def __init__(self, root, transform=None, n_data=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.captions = []

        json = read_json(os.path.join(root, "laion_index.json"))
        json = [{"image": os.path.join(root, item["image"]), "caption": item["caption"]} for item in json]
        items = json[:n_data] if n_data is not None and n_data < len(json) else json
        for item in items:
            img_path, caption = item["image"], item["caption"]
            self.image_paths.append(img_path)
            self.captions.append(caption)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, caption

def load_train_dataset(args):
    if args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.root, transform=preprocess224, download=True, train=True)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess224, download=True, train=True)
    elif args.dataset == 'ImageNet':
        imagenet_root = args.imagenet_root
        refined_data = read_json("ImageNet_files/imagenet_refined_labels.json")
        eval_select = {ssid:refined_data[ssid]['eval_files'] for ssid in refined_data}
        from code.utils import sample_trainfiles
        if hasattr(args, 'train_ratio') and args.train_ratio is not None:
            train_select = sample_trainfiles(args, train_ratio=args.train_ratio, train_ways=args.train_ways)
        elif hasattr(args, 'train_shots') and args.train_shots is not None:
            train_select = sample_trainfiles(args, train_shots=args.train_shots, train_ways=args.train_ways)
        else:
            train_select = {ssid:list(set(os.listdir(imagenet_root+'/train/'+ssid))-set(eval_select[ssid])) for ssid in refined_data}
        train_dataset = ImageNetFolder(
            os.path.join(eval(f"{args.dataset.lower()}_root"), 'train'), transform=preprocess224, select_files=train_select
        )
    elif args.dataset == "ImageNet_caption":
        imagenet_root = args.imagenet_root
        caption_path = os.path.join(imagenet_root, "train_captions", "all_captions.jsonl")
        image_paths, captions = [], []
        with open(caption_path, "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                image_path, caption = item["image"], item["caption"]
                image_paths.append(image_path)
                captions.append(caption)
        if hasattr(args, 'train_shots') and args.train_shots is not None:
            image_paths = image_paths[:args.train_shots]
            captions = captions[:args.train_shots]
            # use the first N words per caption
            if args.N_words_per_caption is not None:
                new_captions = []
                for caption in captions:
                    words = caption.split()
                    new_caption = ' '.join(words[:args.N_words_per_caption]) if len(words) > args.N_words_per_caption else caption
                    new_captions.append(new_caption)
                captions = new_captions
        elif hasattr(args, 'n_data') and args.n_data is not None:
            image_paths = image_paths[:args.n_data]
            captions = captions[:args.n_data]
        train_dataset = ImageNetCaptionDataset(image_paths, captions, transform=preprocess224)

    elif args.dataset == 'tinyImageNet':
        tinyimagenet_root = args.tinyimagenet_root
        refined_data = read_json("ImageNet_files/tinyimagenet_refined_labels.json")
        eval_select = {ssid:refined_data[ssid]['eval_files'] for ssid in refined_data}
        train_select = {ssid:list(set(os.listdir(tinyimagenet_root+'/train/'+ssid+'/images'))-set(eval_select[ssid])) for ssid in refined_data}
        train_dataset = ImageNetFolder(
            os.path.join(eval(f"{args.dataset.lower()}_root"), 'train'), transform=preprocess224, select_files=train_select
        )
    elif args.dataset == 'smallLAION':
        train_dataset = smallLAION(
            root=os.path.join(args.root, 'laion_samples_small'),
            transform=preprocess224,
            n_data=args.n_data if hasattr(args, 'n_data') else None,
        )
    else:
        print(f"Train dataset {args.dataset} not implemented")
        raise NotImplementedError

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
               num_workers=args.num_workers, shuffle=True,)
    
    return train_dataset, dataloader
    
def load_val_dataset(args, val_dataset_name):
    if val_dataset_name == 'cifar10':
        val_dataset = CIFAR10(args.root, transform=preprocess224, download=True, train=False)

    elif val_dataset_name == 'cifar100':
        val_dataset = CIFAR100(args.root, transform=preprocess224, download=True, train=False)

    elif val_dataset_name == 'Caltech101':
        val_dataset = caltech.Caltech101(args.root, target_type='category', transform=preprocess224_caltech, download=True)

    elif val_dataset_name == 'PCAM':
        val_dataset = pcam.PCAM(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'STL10':
        val_dataset = STL10(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'SUN397':
        val_dataset = sun397.SUN397(args.root, transform=preprocess224, download=True)

    elif val_dataset_name == 'StanfordCars':
        val_dataset = stanford_cars.StanfordCars(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'Food101':
        val_dataset = food101.Food101(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'oxfordpet':
        val_dataset = oxford_iiit_pet.OxfordIIITPet(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'EuroSAT':
        val_dataset = eurosat.EuroSAT(args.root, transform=preprocess224, download=True)

    elif val_dataset_name == 'Caltech256':
        val_dataset = caltech.Caltech256(args.root, transform=preprocess224_caltech, download=True)

    elif val_dataset_name == 'flowers102':
        val_dataset = flowers102.Flowers102(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'Country211':
        val_dataset = country211.Country211(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'dtd':
        val_dataset = dtd.DTD(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'fgvc_aircraft':
        val_dataset = fgvc_aircraft.FGVCAircraft(args.root, split='test', transform=preprocess224, download=True)

    elif val_dataset_name == 'tinyImageNet':
        if args.evaluate:
            val_dataset = ImageNetFolder(os.path.join(args.tinyimagenet_root, 'val_'), transform=preprocess224)
        else:
            eval_select = get_eval_files(val_dataset_name)
            val_dataset = ImageNetFolder(os.path.join(args.tinyimagenet_root, 'train'), transform=preprocess224, select_files=eval_select)

    elif val_dataset_name == 'ImageNet':
        if args.evaluate:
            val_dataset = ImageNetFolder(os.path.join(args.imagenet_root, 'val'), transform=preprocess224)
        else:
            eval_select = get_eval_files(val_dataset_name)
            val_dataset = ImageNetFolder(os.path.join(args.imagenet_root, 'train'), transform=preprocess224, select_files=eval_select)
    
    # cross-domain datasets of ImageNet
    elif val_dataset_name == 'ImageNet-Sketch':
        synset2class = read_json("./data/ImageNet/imagenet_refined_labels.json")
        val_dataset = ImageNet_Sketch.ImageNetSketchFolder(
            os.path.join(args.root, 'ImageNet-Sketch'),
            transform=preprocess224,
            synset2class=synset2class
        )
    elif val_dataset_name == 'ImageNet-A':
        synset2class = read_json("./data/ImageNet/imagenet_refined_labels.json")
        val_dataset = ImageNet_a.ImageNet_A_Folder(
            os.path.join(args.root, 'imagenet-a'),
            transform=preprocess224,
            synset2class=synset2class
        )
    elif val_dataset_name == 'ImageNet-R':
        synset2class = read_json("./data/ImageNet/imagenet_refined_labels.json")
        val_dataset = ImageNet_r.ImageNet_R_Folder(
            os.path.join(args.root, 'imagenet-r'),
            transform=preprocess224,
            synset2class=synset2class
        )
    elif val_dataset_name == "ObjectNet":
        val_dataset = ObjectNet.ObjectNetFolder(
            os.path.join(args.root, "objectnet-1.0", 'images'),
            transform=preprocess224,
            template="This is a photo of a {}.",
        )

    else:
        print(f"Val dataset {val_dataset_name} not implemented")
        raise NotImplementedError
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, 
               num_workers=args.num_workers, shuffle=False,)
    
    return val_dataset, val_loader

def prepare_classification_text(args, train_dataset):
    class_names = train_dataset.classes
    if args.dataset in ['ImageNet', 'tinyImageNet']:
        refined_data = read_json(eval(f"args.{args.dataset.lower()}_root") + f"/{args.dataset.lower()}_refined_labels.json")
        class_names = [refined_data[ssid]['clean_name'] for ssid in class_names]
        args.train_class_names = class_names
    texts_train = [args.template.format(label) for label in class_names]
    return texts_train