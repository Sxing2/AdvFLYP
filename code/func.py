import torch
import numpy as np
from code.replace import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).to(device)
std = torch.tensor(CIFAR100_STD).view(3, 1, 1).to(device)

def normalize(X):
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X

class BATCH_COMPUTE(object):
    def __init__(self, image_features, text_features, logits_per_image, logits_per_text,
                logits_scaled:bool=True, feature_normalized:bool=True):
        self.image_features = image_features
        self.text_features = text_features
        self.logits_per_image = logits_per_image
        self.logits_per_text = logits_per_text
        self.logits_scaled = logits_scaled
        self.feature_normalized = feature_normalized

    
def multiGPU_CLIP_image_logits(images, model, text_tokens, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(None, None, None, model, image_tokens, text_tokens, prompt_token=prompt_token)[0]


def multiGPU_CLIP(args, model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    # text_tokens = clip.tokenize(text).to(device)
    text_features = model.module.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True) # [n_class, d_emb]
    text_features = text_features.to(device)

    image_features = model.module.encode_image(images, prompt_token)
    if len(image_features.shape) == 3:
        image_features = image_features[:,0,:] # ViT backbone
    image_features = image_features / image_features.norm(dim=1, keepdim=True) # [bs, d_emb]

    logits_per_image = image_features @ text_features.t() * model.module.logit_scale.exp()
    logits_per_text = text_features @ image_features.t() * model.module.logit_scale.exp()

    return logits_per_image, logits_per_text, image_features, text_features


def image_text_cossim(prompted_images, captions, model,  prompt_token=None):
    # compute image-text cosine similarity and InfoNCE loss for COCOCaptions
    logit_scale = model.module.logit_scale.exp() if hasattr(model.module, 'logit_scale') else 1.0

    images_features = model.module.encode_image(prompted_images, prompt_token)
    if len(images_features.shape) == 3:
        images_features = images_features[:,0,:] # ViT backbone

    # images_features = model.module.encode_image(prompted_images, prompt_token)[:,0,:] # [bs, 512]
    images_features = images_features / images_features.norm(dim=-1, keepdim=True) # [bs, 512]

    if isinstance(captions[0], str): # for dataset which only has one text per image
        captions = [[caption] for caption in captions]

    min_caption_num = min([len(caption) for caption in captions])
    captions = [caption[:min_caption_num] for caption in captions] # [bs, min_caption_num (usually 5)]
    captions = np.array(captions).flatten().tolist() # [bs * min_caption_num,]
    captions_features = model.module.encode_text(clip.tokenize(captions, truncate=True).to(device)) # [bs*5, 512]
    captions_features = captions_features.view(prompted_images.size(0), min_caption_num, captions_features.size(-1)).mean(dim=1) # [bs, 512]
    captions_features = captions_features / captions_features.norm(dim=-1, keepdim=True) # [bs, 512]

    logits_per_image = images_features @ captions_features.t() * logit_scale # [bs, bs]
    logits_per_text = captions_features @ images_features.t() * logit_scale # [bs, bs]
    labels = torch.arange(prompted_images.size(0), device=device)
    # infoce_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.0
    
    return BATCH_COMPUTE(images_features, captions_features, logits_per_image, logits_per_text,
                        logits_scaled=True, feature_normalized=True)

def multiGPU_CLIP_loss(args, model_image, model_text, model, images, \
                       text=None, text_tokens=None, prompt_token=None, dataset_name=None):
    assert text is not None or text_tokens is not None, "Either text or text_tokens must be provided."
    if text is not None:
        text_tokens = clip.tokenize(text).to(device)

    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    text_features = model.module.encode_text(text_tokens).to(device)
    text_features = text_features / text_features.norm(dim=1, keepdim=True) # [n_class, d_emb]
    image_features = model.module.encode_image(images, prompt_token).to(device)
    if len(image_features.shape) == 3:
        image_features = image_features[:,0,:] # ViT backbone
    image_features = image_features / image_features.norm(dim=1, keepdim=True) # [bs, d_emb]

    logits_per_image = image_features @ text_features.t() * model.module.logit_scale.exp()
    logits_per_text = text_features @ image_features.t() * model.module.logit_scale.exp()
    # tecoa_loss = F.cross_entropy(logits_per_image, target)

    return BATCH_COMPUTE(image_features, text_features, logits_per_image, logits_per_text,
                        logits_scaled=True, feature_normalized=True)


def kl_div(p_logits, q_logits):
    # p_logits, q_logits [bs, n_class] both have been softmax normalized
    kl_divs = (p_logits * (p_logits.log() - q_logits.log())).sum(dim=1) # [bs,]
    return kl_divs.mean()

def get_loss_general(tgt_logits, a_images, model_image_copy, text_features):
    # feed the perturbed image into the original visual encoder, regularise the predictive logits
    image_features = model_image_copy(a_images) # [bs, d_emb]
    if len(image_features.shape) == 3:
        image_features = image_features[:,0,:] # ViT backbone
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logits_per_image_ = image_features @ text_features.t() * model_image_copy.module.logit_scale.exp() # [bs, n_class]
    l_general = kl_div(tgt_logits.softmax(dim=1), logits_per_image_.softmax(dim=1))
    # l_general = criterion_(F.log_softmax(logits_per_image_, dim=1), F.softmax(tgt_logits))
    return l_general

def get_loss_clean(clean_images, tgt_logits, model, text_features, prompt_token=None):
    # feed the clean image into the visual encoder, regularise the predictive logits
    image_features = model.module.encode_image(clean_images, prompt_token) # [bs, d_emb]
    if len(image_features.shape) == 3:
        image_features = image_features[:,0,:] # ViT backbone
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits_per_image = image_features @ text_features.t() * model.module.logit_scale.exp() # [bs, n_class]
    l_clean = kl_div(tgt_logits.softmax(dim=1), logits_per_image.softmax(dim=1))
    # l_clean = criterion_(F.log_softmax(logits_per_image, dim=1), F.softmax(tgt_logits, dim=1))
    return l_clean

def attention_map(text_features, vision_model, images, prompt_token, args):
    """feature extract extraction"""
    image_features = vision_model(images,prompt_token)  
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    img_spatial_feat = image_features[:,1:,:]

    """Text guided attention map"""
    am = img_spatial_feat @ text_features.unsqueeze(-1)
    am = (am-am.min(1, keepdim=True)[0]) / (am.max(1, keepdim=True)[0] - am.min(1, keepdim=True)[0] )
    """reshape"""
    side = int(am.shape[1] ** 0.5) 
    am = am.reshape(am.shape[0], side, side, -1).permute(0, 3, 1, 2)

    """interpolate"""
    am = torch.nn.functional.interpolate(am, args.image_size, mode='bilinear')

    return am

def tga_zsr_criterion(model, output, target, adv_atten, clean_atten, clean_atten_model, args):
    """Cross entropy loss"""
    CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
    loss_TeCoA = CrossEntropyLoss(output, target)
    
    """attention map loss"""
    if args.Distance_metric == 'cos':
        loss_AM1 = torch.mean(1-torch.nn.functional.cosine_similarity(adv_atten, clean_atten, dim=1, eps=1e-8))
        loss_AM2 = torch.mean(1-torch.nn.functional.cosine_similarity(clean_atten_model, clean_atten, dim=1, eps=1e-8))
    elif args.Distance_metric == 'l2':
        loss_AM1 = torch.mean(torch.norm(adv_atten - clean_atten,dim=1, p=2))
        loss_AM2 = torch.mean(torch.norm(clean_atten_model - clean_atten,dim=1, p=2))
    elif args.Distance_metric == 'l1':
        l1_loss = torch.nn.L1Loss(reduction='mean')
        loss_AM1 = l1_loss(adv_atten, clean_atten)
        loss_AM2 = l1_loss(clean_atten_model, clean_atten)
    return loss_TeCoA ,args.Alpha*loss_AM1 ,args.Beta*loss_AM2

## textual templates for imagenet
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]