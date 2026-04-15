import os, time, wandb
import torch
import torch.nn.functional as F
from code.replace import clip
from torch.amp import autocast
from tqdm import tqdm
from code.utils import AverageMeter, ProgressMeter, accuracy, save_checkpoint
from code.func import clip_img_preprocessing, image_text_cossim, multiGPU_CLIP_loss, \
                get_loss_general, get_loss_clean, attention_map, tga_zsr_criterion, kl_div
from code.attacks import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch_laion(args, train_loader, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, scaler, epoch, trainable, best_acc1, visual_model_orig):
    
    # define training meters
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    I2T_top1 = AverageMeter('I2T_Acc@1', ':6.2f')
    T2I_top1 = AverageMeter('T2I_Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, I2T_top1, T2I_top1],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    # switch to train mode
    model.module.visual.train()
    num_batches_per_epoch = len(train_loader)
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    end = time.time()
    for i, (images, captions) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        BATCH_SIZE = images.size(0)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images = images.to(device)

        # with automatic mixed precision
        with autocast('cuda'):
            
            if args.train_attack_type == 'None': # no attack
                tmp = clip_img_preprocessing(images)
            elif args.train_attack_type == 'pgd':
                delta = attack_pgd_captions(args, prompter,  model, model_text, model_image, 
                                            add_prompter, images, captions, alpha, attack_iters, 'l_inf',
                                            epsilon=args.train_eps)
                tmp = clip_img_preprocessing(images + delta)
            else:
                raise NotImplementedError("No attacks are generated for training.")

            prompted_images = prompter(tmp)
            prompt_token = None

            batch_logits = image_text_cossim(prompted_images, captions, model, prompt_token)
            labels = torch.arange(BATCH_SIZE, device=device)
            infoce_loss = (
                F.cross_entropy(batch_logits.logits_per_image, labels) + \
                F.cross_entropy(batch_logits.logits_per_text, labels)
            ) / 2.0
            loss = infoce_loss

            # logit and feature-level regularization
            adv_images_feat_orig = visual_model_orig(prompted_images) # [bs, d_emb]
            if len(adv_images_feat_orig.shape) == 3:
                adv_images_feat_orig = adv_images_feat_orig[:,0,:]
            adv_images_feat_orig = adv_images_feat_orig / adv_images_feat_orig.norm(dim=-1, keepdim=True)

            clean_images_feat_target = model.module.encode_image(
                prompter(clip_img_preprocessing(images)), prompt_token
            ) # [bs, d_emb]
            if len(clean_images_feat_target.shape) == 3:
                clean_images_feat_target = clean_images_feat_target[:,0,:]
            clean_images_feat_target = clean_images_feat_target / clean_images_feat_target.norm(dim=-1, keepdim=True)

            if 'feat' in args.reg_level: # feature-level regularization
                feat_reg1 = torch.norm(batch_logits.image_features - adv_images_feat_orig, dim=-1).mean()
                feat_reg2 = torch.norm(batch_logits.image_features - clean_images_feat_target, dim=-1).mean()
                loss = loss + args.lambda_feat * (feat_reg1 + feat_reg2)
            if 'logit' in args.reg_level: # logit-level regularization
                text_features = batch_logits.text_features / batch_logits.text_features.norm(dim=-1, keepdim=True) \
                            if not batch_logits.feature_normalized else batch_logits.text_features
                adv_logits_per_image = adv_images_feat_orig @ text_features.t() * visual_model_orig.module.logit_scale.exp()
                clean_logits_per_image = clean_images_feat_target @ text_features.t() * model.module.logit_scale.exp()
                l_general = kl_div(batch_logits.logits_per_image.softmax(dim=1), adv_logits_per_image.softmax(dim=1))
                l_clean = kl_div(batch_logits.logits_per_image.softmax(dim=1), clean_logits_per_image.softmax(dim=1))
                loss = loss + args.lambda_logit *  (l_general + l_clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        I2T_acc1 = accuracy(batch_logits.logits_per_image, labels, topk=(1,))
        T2I_acc1 = accuracy(batch_logits.logits_per_text, labels, topk=(1,))
        I2T_top1.update(I2T_acc1[0].item(), images.size(0))
        T2I_top1.update(T2I_acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'training_loss': losses.avg,
                    'I2T_training_acc': I2T_top1.avg,
                    'T2I_training_acc': T2I_top1.avg
                })
        if i % args.save_freq == 0: # save checkpoint at regular intervals
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'partial_state_dict': {k:v for k,v in model.module.state_dict().items() if k in trainable},
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args.model_folder)

def train_epoch(args, train_loader, texts, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, criterion_attack, scaler, epoch, trainable, best_acc1,
          visual_model_orig=None):
    
    # define training meters
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.module.visual.train()
    num_batches_per_epoch = len(train_loader)
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps
    text_tokens = clip.tokenize(texts).to(device)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        BATCH_SIZE = images.size(0)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)

        # with automatic mixed precision
        with autocast('cuda'):
            if args.train_attack_type is None:
                tmp = clip_img_preprocessing(images)
            elif args.train_attack_type == 'pgd':
                delta = attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion_attack, images,
                                   target, alpha, attack_iters, 'l_inf',
                                   text_tokens=text_tokens, epsilon=args.train_eps, dataset_name=args.dataset)
                tmp = clip_img_preprocessing(images + delta)
            else:
                raise NotImplementedError("No attacks are generated for training.")

            prompted_images = prompter(tmp)
            prompt_token = None

            # compute batch logits
            batch_logits = multiGPU_CLIP_loss(args, None, None, model, prompted_images, text_tokens=text_tokens, prompt_token=prompt_token)
            
            tecoa_loss = F.cross_entropy(batch_logits.logits_per_image, target)
            if args.loss == 'tecoa':
                loss = tecoa_loss
            elif args.loss == 'pmg_aft':
                l_general = get_loss_general(
                    batch_logits.logits_per_image, prompted_images, visual_model_orig, batch_logits.text_features
                )
                l_clean = get_loss_clean(
                    prompter(clip_img_preprocessing(images)), batch_logits.logits_per_image, model, batch_logits.text_features
                )
                loss = tecoa_loss + l_general + l_clean
            elif args.loss == 'tga_zsr':
                attack_tar = attention_map(
                    batch_logits.text_features[target,:], model.module.visual, prompted_images, prompt_token, args
                ).view(BATCH_SIZE, -1)
                clean_ori = attention_map(
                    batch_logits.text_features[target,:], visual_model_orig, prompter(clip_img_preprocessing(images)), prompt_token, args
                ).view(BATCH_SIZE, -1)
                clean_tar = attention_map(
                    batch_logits.text_features[target,:], model.module.visual, prompter(clip_img_preprocessing(images)), prompt_token, args
                ).view(BATCH_SIZE, -1)
                loss_AM1 = torch.mean(torch.norm(attack_tar-clean_ori, dim=1, p=2))
                loss_AM2 = torch.mean(torch.norm(clean_tar-clean_ori, dim=1, p=2))
                loss = tecoa_loss + 0.08 * loss_AM1 + 0.05 * loss_AM2
            else:
                raise NotImplementedError("Loss function not implemented: {}".format(args.loss))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(batch_logits.logits_per_image, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                     })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'partial_state_dict': {k:v for k,v in model.module.state_dict().items() if k in trainable},
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args.model_folder)

    return losses.avg, top1.avg
