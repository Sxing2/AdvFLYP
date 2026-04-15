from code.utils import *
from code.data_engine import load_val_dataset
from code.replace import clip
import time
from tqdm import tqdm
from torch.amp import autocast
from code.func import multiGPU_CLIP, clip_img_preprocessing
from code.attacks import *

# import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def validate(args, val_dataset_name, model, model_text, model_image,
             prompter, add_prompter, criterion, test_mode=False):
    
    print(f"Evaluate with Attack method: {args.test_attack_type}")    

    dataset_num = len(val_dataset_name)
    acc_all = []
    clean_acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        # skip current dataset if already in the test report
        dataset_name = val_dataset_name[cnt]

        val_dataset, val_loader = load_val_dataset(args, val_dataset_name[cnt])
        texts = get_text_prompts_val([val_dataset], [dataset_name])[0]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        text_tokens = clip.tokenize(texts).to(device)

        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)

            with autocast('cuda'):

                with torch.no_grad():
                    prompt_token = None
                    output_prompt, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model,
                            prompter(clip_img_preprocessing(images)), text_tokens=text_tokens, prompt_token=prompt_token)

                    loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))
                    top1_org.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()

                # generate adv example
                if args.test_attack_type == "CW":
                    delta_prompt = attack_CW(args, prompter, model, model_text, model_image, add_prompter, criterion,
                                             images, target, text_tokens,
                                             test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "autoattack":
                    attacked_images = attack_auto(model, images, target, text_tokens,
                        None, None, epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "pgd":
                    delta_prompt = attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion,
                                              images, target, test_stepsize, args.test_numsteps, 'l_inf',
                                              text_tokens=text_tokens, epsilon=args.test_eps, dataset_name=dataset_name)
                    attacked_images = images + delta_prompt

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    output_prompt_adv, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model,
                                                         prompter(clip_img_preprocessing(attacked_images)),
                                                         text_tokens=text_tokens, prompt_token=prompt_token)
                    loss = criterion(output_prompt_adv, target)

                # bl attack
                torch.cuda.empty_cache()

                # measure accuracy and record loss
                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_prompt.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        torch.cuda.empty_cache()

        print(dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
                      top1_prompt=top1_prompt, top1_org=top1_org))
        acc_all.append(top1_adv_prompt.avg)
        clean_acc_all.append(top1_org.avg)

    if test_mode:

        avg_robust_acc = np.mean(acc_all)
        avg_clean_acc = np.mean(clean_acc_all)

        print(" ----- TEST SUMMARY -----")
        print(f"avg. robust acc: {avg_robust_acc:.2f}")
        print(f"avg.  clean acc: {avg_clean_acc:.2f}")
        print("\n")

    return np.mean(acc_all), np.mean(clean_acc_all)
