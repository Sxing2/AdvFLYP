"""
Adversarially Finetune like you Pretrain.
"""
from __future__ import print_function

import argparse
import os
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler

from code.replace import clip
from code.models.prompters import TokenPrompter, NullPrompter
from code.utils import save_checkpoint, cosine_lr, convert_models_to_fp32, set_tunable_params, from_string, freeze

def parse_option():
    parser = argparse.ArgumentParser('Adversarially Finetune like you Pretrain for CLIP', add_help=False)
    parser.add_argument('--project_name', type=str, default=None,)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument("--mix_alpha", type=float, default=-1, help="interpolation")

    # optimization
    parser.add_argument('--loss', type=str, default='FLYP', choices=['tecoa', 'pmg_aft', 'tga_zsr'], 
                        help='loss function. Ignore this argument for AdvFLYP on image-text pairs.')
    parser.add_argument('--reg_level', type=str, default='logit', nargs='*', choices=['logit', 'feat'])
    parser.add_argument('--lambda_feat', type=float, default=1.)
    parser.add_argument('--lambda_logit', type=float, default=1.)
    parser.add_argument('--AFT_modules', type=str, default=['visual'], nargs='*', help='which modules to finetune')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-7, help='learning rate')
    parser.add_argument('--no_cosine_lr', action='store_true', default=False)
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--train_attack_type', type=str, default="pgd", choices=['pgd', 'CW', 'autoattack', 'None'])
    parser.add_argument('--train_eps', type=float, default=1, help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_attack_type', type=str, default="pgd", choices=['pgd', 'CW', 'autoattack', 'None'])
    parser.add_argument('--test_eps', type=float, default=1, help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=10)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--imagenet_root', type=str, default=None)
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch', choices=['null_patch'], help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0, help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data', help='dataset root')
    parser.add_argument('--dataset', type=str, default='smallLAION', help='training dataset')
    parser.add_argument('--n_data', type=int, default=None, help='number of training data')
    parser.add_argument('--train_ways', type=int, default=None, help='sample N ways for training, only for ImageNet')
    parser.add_argument('--train_shots', type=int, default=None, help='sample N shots per class for training, only for ImageNet')
    parser.add_argument('--N_words_per_caption', type=int, default=None, help='number of words per caption, only for captioned ImageNet')
    parser.add_argument('--train_ratio', type=float, default=None, help='sample ratio of data for training, only for ImageNet.')
    parser.add_argument('--image_size', type=int, default=224, help='image size')

    # other
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models', help='path to save models')
    parser.add_argument('--filename', type=str, default=None, help='filename to save')
    parser.add_argument('--trial', type=int, default=1, help='number of trials')
    parser.add_argument('--resume', type=str, default=None, help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False, action="store_true", help='evaluate model test set')
    parser.add_argument('--test_set', default=None, type=str, nargs="*", 
            choices=[
                # cross-dataset evaluation (14)
                'cifar10', 'cifar100', 'STL10', 'SUN397', 'StanfordCars', 'Food101',
                'oxfordpet', 'flowers102', 'Country211', 'dtd', 'fgvc_aircraft',
                'Caltech101', 'EuroSAT', 'Caltech256', 
                # IN variants
                'ImageNetV2', 'ImageNet-R', 'ImageNet-A', 'ImageNet-Sketch', 'ObjectNet',
            ])
    parser.add_argument('--eval_set', default='tinyImageNet', type=str, help='eval set during finetuning')
    parser.add_argument('--gpu', type=int, default=None, help='gpu to use')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    global best_acc1, device, args, trainable

    args = parse_option()
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.
    print(args)

    if args.use_wandb:
        wandb.login(key="", relogin=True)
        wandb.init(project=args.project_name if args.project_name is not None else 'AdvFLYP', config=vars(args))

    if args.seed is not None:
        from code.utils import set_random_seed
        set_random_seed(args.seed)

    imagenet_root = os.path.join(args.root, 'ImageNet')
    tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200')
    args.imagenet_root = imagenet_root
    args.tinyimagenet_root = tinyimagenet_root

    # create model
    add_prompt_len = 0
    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    for p in model.parameters(): p.requires_grad = False
    if len(args.AFT_modules) > 0:
        print("Finetuning the following modules:", args.AFT_modules)
        for module_name in args.AFT_modules:
            set_tunable_params(model, module_name)
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    model_text, model_image = None, None

    if not args.evaluate:
        print("Tunable parameters:\n")
        for param_name in trainable: print('\t', param_name)

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)  # .to(device)
    model.eval()

    # Get original CLIP vision encoder (for regularisation)
    print("Load orginal CLIP vision encoder and keep it frozen.")
    model_orig, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    visual_model_orig = model_orig.visual
    visual_model_orig.logit_scale = model_orig.logit_scale.data
    convert_models_to_fp32(visual_model_orig)
    del model_orig
    freeze(visual_model_orig)
    visual_model_orig = torch.nn.DataParallel(visual_model_orig)
    visual_model_orig.eval()

    prompter = NullPrompter()
    add_prompter = TokenPrompter(add_prompt_len)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    # define criterion and optimizer
    optimizer = torch.optim.SGD([p for p in model.module.parameters() if p.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    args.start_epoch = 0

    # optionally resume from a checkpoint
    from code.utils import load_checkpoints_partial
    load_checkpoints_partial(args, model, optimizer)

    # create data
    template = 'This is a photo of a {}.'
    print(f'template: {template}')
    args.template = template

    if args.evaluate: # testing on downstream datasets
        freeze(model.module)
        val_dataset_name = args.test_set
        print(f"{len(val_dataset_name)} datasets to be evaluated: {val_dataset_name}")

        from code.evaluate import validate

        acc1_mean, clean_acc1_mean = validate(args, val_dataset_name, model, model_text, model_image,
                             prompter, add_prompter, criterion_attack, True)
        print(f"Avg. accuracy across {len(val_dataset_name)} datasets:")
        print(f"\t - Robust Acc. {acc1_mean}")
        print(f"\t - Clean  Acc. {clean_acc1_mean}")
        return

    else: # adversarial training
        if args.eval_set is not None: # eval set during training, defaults to TinyImageNet
            from code.utils import DATASETS
            val_dataset_name = from_string(args.eval_set, check_func=lambda x:x in DATASETS)
        else:
            val_dataset_name = [args.dataset]

        from code.data_engine import load_train_dataset
        print(f"Loading training dataset: {args.dataset}")
        train_dataset, train_loader = load_train_dataset(args)
        if args.dataset not in ['smallLAION', 'ImageNet_caption']: # prepare text prompts for category datasets
            from code.data_engine import prepare_classification_text
            texts_train = prepare_classification_text(args, train_dataset)
        del train_dataset

        # prepare scheduler and scaler ()
        scaler = GradScaler()
        total_steps = len(train_loader) * args.epochs
        if args.no_cosine_lr:
            from code.utils import warmup_lr
            scheduler = warmup_lr(optimizer, args.learning_rate, args.warmup)
        else:
            scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

        cudnn.benchmark = True

        # make dir
        refined_template = template.lower().replace(' ', '_')
        args.filename = f'{args.filename}_template_{refined_template}'

        args.model_folder = os.path.join(args.model_dir, args.filename)
        if not os.path.isdir(args.model_folder):
            os.makedirs(args.model_folder)

    epochs_since_improvement = 0

    from code.evaluate import validate
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        if args.dataset in ['smallLAION', 'ImageNet_caption']: # train on image-text
            from code.training_epoch import train_epoch_laion
            train_epoch_laion(
                args, train_loader, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler,
                scaler, epoch, trainable, best_acc1, visual_model_orig
            )
        else: # train on category datasets (ImageNet)
            from code.training_epoch import train_epoch
            train_epoch(
                args, train_loader, texts_train, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler,
                criterion, criterion_attack, scaler, epoch, trainable, best_acc1, visual_model_orig
            )

        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            acc1_mean, clean_acc1_mean = validate(args, val_dataset_name, model, model_text, model_image,
                                 prompter, add_prompter, criterion_attack)

        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'partial_state_dict': {k:v for k,v in model.module.state_dict().items() if k in trainable},
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args.model_folder, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

if __name__ == '__main__':
    main()
