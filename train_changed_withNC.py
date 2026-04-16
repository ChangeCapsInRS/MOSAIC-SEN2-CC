import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import subprocess
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
import datetime
import copy
from models import CNN_Encoder
from models_RSICCformerDfusion import *
from datasets import *
from utils import *
from eval_changed import evaluate_transformer



seed = 1
torch.manual_seed(seed)

metrics_list = []
losses_output = []
AVG_losses_output = []
top5_accuracy_output = []
batch_time_output = []

train_model_sonuc_map = {}
text_terminal = " "

rogue_l_output = []
cider_output = []
spice_output = []
bleu_4_output = []
rogue_l_nochange_output = []
cider_nochange_output = []
spice_nochange_output = []
bleu_4_nochange_output = []
meteor1_nochange_output = []
meteor1_change_output = []
meteor1_output = []
rogue_l_change_output = []
cider_change_output = []
spice_change_output = []
bleu_4_change_output = []

val_model_sonuc_map = {}

def compute_avg_score(metrics: dict) -> float:
    return (
        metrics["Bleu_4"]
        + metrics["METEOR"]
        + metrics["ROUGE_L"]
        + metrics["CIDEr"]
        + metrics["SPICE"]
    ) / 5.0

def print_with_json(text):
    global text_terminal
    print(text)
    text_terminal += str(text) + "\n"


def train(args, train_loader, encoder_image_ms, encoder_feat, decoder, criterion, encoder_image_ms_optimizer,
          encoder_image_ms_lr_scheduler, encoder_feat_optimizer, encoder_feat_lr_scheduler, decoder_optimizer,
          decoder_lr_scheduler, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    encoder_image_ms.train()
    encoder_feat.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs_our = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    fused_data = []

    # Batches
    best_bleu4 = 0.  # BLEU-4 score right now

    scaler = torch.cuda.amp.GradScaler()
    start = time.time()

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        #if i==5:
        #    break
        data_time.update(time.time() - start)

        img_pairs["before"] = img_pairs["before"].to(device, non_blocking=True)
        img_pairs["after"] = img_pairs["after"].to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        caplens = caplens.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            images_before = img_pairs["before"]  # shape: [B, C, H, W]
            images_after = img_pairs["after"]

            feat_before = encoder_image_ms(images_before)
            feat_after = encoder_image_ms(images_after)

            fused_feat = encoder_feat(feat_before, feat_after)

            scores, caps_sorted, decode_lengths, sort_ind = decoder(fused_feat, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)

        # Backward pass
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        if encoder_image_ms_optimizer is not None:
            encoder_image_ms_optimizer.zero_grad()

        scaler.scale(loss).backward()

        if args.grad_clip is not None:
            scaler.unscale_(decoder_optimizer)
            clip_gradient(decoder_optimizer, args.grad_clip)

            scaler.unscale_(encoder_feat_optimizer)
            clip_gradient(encoder_feat_optimizer, args.grad_clip)

            if encoder_image_ms_optimizer is not None:
                scaler.unscale_(encoder_image_ms_optimizer)
                clip_gradient(encoder_image_ms_optimizer, args.grad_clip)

        scaler.step(decoder_optimizer)
        scaler.step(encoder_feat_optimizer)

        if encoder_image_ms_optimizer is not None:
            scaler.step(encoder_image_ms_optimizer)

        scaler.update()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % args.print_freq == 0:
            print_with_json(
                f"Epoch: {epoch + 1}/{args.epochs} | Step: {i}/{len(train_loader)} | Loss: {losses.val:.4f} | "
                f"Avg Loss: {losses.avg:.4f} | Top-5 Accuracy: {top5accs.val:.4f} | Batch Time: {batch_time.val:.4f}s"
            )
            #print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | "
            #      f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

            losses_output.append(losses.val)
            AVG_losses_output.append(losses.avg)
            top5_accuracy_output.append(top5accs.val)
            batch_time_output.append(batch_time.val)


def key_transformation(old_key):
    if old_key == "layer.0.weight":
        return "layer.1.weight"

    return old_key


def main(args, meteor_output=None):
    print_with_json(args)
    global metrics_list
    print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize
    # Encoder
    encoder_image_ms = CNN_Encoder(NetType=args.encoder_image, method=args.decoder,
    weight_path="BigEarthnetModels/resnet101-s2-v0.1.1.pth")

    #encoder_image_ms.load_state_dict(torch.load("BigEarthnetModels/resnet101-all.pth"))

    #encoder_image_ms = encoder_image_ms.to(device)

    encoder_image_ms.fine_tune(args.fine_tune_encoder)

    # set the encoder_dim
    encoder_image_dim = 1024  # resnet101 ms
    # filename = os.listdir(args.checkpoint)
    # checkpoint_path = os.path.join(args.checkpoint, filename[0])
    # print_with_json(args.checkpoint + filename[0])
    # checkpoint = torch.load(checkpoint_path, map_location=str(device))
    # encoder_image2 = checkpoint['encoder_image']
    # encoder_feat2 = checkpoint['encoder_feat']
    # decoder2 = checkpoint['decoder']

    if args.encoder_feat == 'MCCFormers_diff_as_Q':
        encoder_feat = MCCFormers_diff_as_Q(feature_dim=encoder_image_dim, dropout=0.5, h=15, w=15, d_model=512,
                                            n_head=args.n_heads,
                                            n_layers=args.n_layers)

    # Decoder  # 当有concat是1024,否则为512
    if args.decoder == 'trans':
        decoder = DecoderTransformer(feature_dim=args.feature_dim_de,
                                     vocab_size=len(word_map),
                                     n_head=args.n_heads,
                                     n_layers=args.decoder_n_layers,
                                     dropout=args.dropout)

    encoder_image_ms_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder_image_ms.parameters()),
        lr=args.encoder_lr) if args.fine_tune_encoder else None

    if args.checkpoint != 'None':
        filename = os.listdir(args.checkpoint)
        checkpoint_path = os.path.join(args.checkpoint, filename[0])
        # print_with_json(args.checkpoint + filename[0])
        checkpoint = torch.load(checkpoint_path, map_location=str(device))

        torch.save(checkpoint['encoder_image'].state_dict(), 'model_weights_ms.pth')
        torch.save(checkpoint['encoder_feat'].state_dict(), 'model_weights_feat.pth')
        torch.save(checkpoint['decoder'].state_dict(), 'model_weights_decoder.pth')

        encoder_image_ms.load_state_dict(torch.load('model_weights_ms.pth'))
        encoder_feat.load_state_dict(torch.load('model_weights_feat.pth'))
        decoder.load_state_dict(torch.load('model_weights_decoder.pth'))

    # encoder_image2 = checkpoint['encoder_image']

    encoder_image_ms_lr_scheduler = StepLR(encoder_image_ms_optimizer, step_size=900,
                                            gamma=1) if args.fine_tune_encoder else None

    encoder_feat_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_feat.parameters()),
                                              lr=args.encoder_lr)
    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=900, gamma=1)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    decoder_lr_scheduler = StepLR(decoder_optimizer, step_size=900, gamma=1)

    # Move to GPU, if available

    encoder_image_ms = encoder_image_ms.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)

    print_with_json("Checkpoint_savepath:{}".format(args.savepath))
    print_with_json(
        "Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(args.encoder_image, args.encoder_feat,
                                                                                args.decoder))
    print_with_json("encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
                    "decoder_lr {}".format(args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout,
                                           args.encoder_lr, args.decoder_lr))

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN',
        transform=transforms.Compose([normalize])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for x consecutive epochs, and terminate training after x
        if epochs_since_improvement == args.stop_criteria:
            print_with_json("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            if args.fine_tune_encoder:
                if encoder_image_ms_optimizer is not None:
                    print_with_json(encoder_image_ms_optimizer)
                    adjust_learning_rate(encoder_image_ms_optimizer, 0.8)

        # One epoch's training
        print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(args,
              train_loader=train_loader,
              encoder_image_ms=encoder_image_ms,
              encoder_feat=encoder_feat,
              decoder=decoder,
              criterion=criterion,
              encoder_image_ms_optimizer=encoder_image_ms_optimizer,
              encoder_image_ms_lr_scheduler=encoder_image_ms_lr_scheduler,
              encoder_feat_optimizer=encoder_feat_optimizer,
              encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
              decoder_optimizer=decoder_optimizer,
              decoder_lr_scheduler=decoder_lr_scheduler,
              epoch=epoch)

        # One epoch's validation
        metrics, nochange_metrics, change_metrics = evaluate_transformer(args,
                     encoder_image_ms=encoder_image_ms,
                     encoder_feat=encoder_feat,
                     decoder=decoder)
        metrics_list.append(metrics)
        recent_bleu4 = metrics["Bleu_4"]
        bleu_4_output.append([metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"], metrics["Bleu_4"]])
        rogue_l_output.append(metrics["ROUGE_L"])
        meteor1_output.append(metrics["METEOR"])
        cider_output.append(metrics["CIDEr"])
        spice_output.append(metrics["SPICE"])
        bleu_4_nochange_output.append(
            [nochange_metrics["Bleu_1"], nochange_metrics["Bleu_2"], nochange_metrics["Bleu_3"],
             nochange_metrics["Bleu_4"]])
        rogue_l_nochange_output.append(nochange_metrics["ROUGE_L"])
        cider_nochange_output.append(nochange_metrics["CIDEr"])
        meteor1_nochange_output.append(nochange_metrics["METEOR"])
        spice_nochange_output.append(nochange_metrics["SPICE"])
        bleu_4_change_output.append([change_metrics["Bleu_1"], change_metrics["Bleu_2"], change_metrics["Bleu_3"],
                                     change_metrics["Bleu_4"]])
        rogue_l_change_output.append(change_metrics["ROUGE_L"])
        cider_change_output.append(change_metrics["CIDEr"])
        meteor1_change_output.append(change_metrics["METEOR"])
        spice_change_output.append(change_metrics["SPICE"])
        # Check if there was an improvement
        if args.best_metric_mode == "bleu4":
            score_to_consider = metrics["Bleu_4"]
            print_with_json(f"[Best Epoch Selection] Using BLEU-4: {score_to_consider:.4f}")
        elif args.best_metric_mode == "avg":
            score_to_consider = compute_avg_score(metrics)
            print_with_json(f"[Best Epoch Selection] Using AVG of metrics: {score_to_consider:.4f}")
        else:
            raise ValueError(f"Unknown best_metric_mode: {args.best_metric_mode}")

        # Check if this is best so far
        is_best = score_to_consider > best_bleu4
        best_bleu4 = max(score_to_consider, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print_with_json("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        checkpoint_name = args.encoder_image + '_' + args.encoder_feat + '_' + args.decoder  # _tengxun_aggregation
        save_checkpoint(args, checkpoint_name, epoch, epochs_since_improvement,
                        encoder_image_ms, encoder_feat, decoder, encoder_image_ms_optimizer,
                        encoder_feat_optimizer, decoder_optimizer,
                        metrics, is_best)
    train_model_sonuc_map["losses"] = losses_output
    train_model_sonuc_map["avg_losses"] = AVG_losses_output
    train_model_sonuc_map["top5_acc"] = top5_accuracy_output
    val_model_sonuc_map["rogue_l"] = rogue_l_output
    val_model_sonuc_map["cider"] = cider_output
    val_model_sonuc_map["spice"] = spice_output
    val_model_sonuc_map["bleu_4"] = bleu_4_output
    val_model_sonuc_map["meteor"] = meteor1_output
    val_model_sonuc_map["rogue_l_nochange"] = rogue_l_nochange_output
    val_model_sonuc_map["cider_nochange"] = cider_nochange_output
    val_model_sonuc_map["meteor_nochange"] = meteor1_nochange_output
    val_model_sonuc_map["spice_nochange"] = spice_nochange_output
    val_model_sonuc_map["bleu_4_nochange"] = bleu_4_nochange_output
    val_model_sonuc_map["rogue_l_change"] = rogue_l_change_output
    val_model_sonuc_map["cider_change"] = cider_change_output
    val_model_sonuc_map["bleu_4_change"] = bleu_4_change_output
    val_model_sonuc_map["meteor_change"] = meteor1_change_output
    val_model_sonuc_map["spice_change"] = spice_change_output

    train_model_sonuc_json = json.dumps(train_model_sonuc_map, indent=4)
    val_model_sonuc_json = json.dumps(val_model_sonuc_map, indent=4)
    # Get the current date in the format YYYY-MM-DD
    current_date = datetime.date.today().strftime("%Y%m%d")

    # Define your save path
    output_save_path = args.savepath.replace('/model_dir', '')

    # Construct the filename with the current date
    file_name = f'{output_save_path}/train_{current_date}.json'
    file_name2 = f'{output_save_path}/val_{current_date}.json'
    file_name3 = f'{output_save_path}/terminal_text_{current_date}.txt'

    # Write the JSON data to the file
    with open(file_name3, 'w') as dosya:
        dosya.write(text_terminal)
    with open(file_name, 'w') as dosya:
        dosya.write(train_model_sonuc_json)
    with open(file_name2, 'w') as dosya:
        dosya.write(val_model_sonuc_json)


current_date = datetime.date.today().strftime("%Y%m%d")

if __name__ == '__main__':
    #time.sleep(7200)
    dosya_index = 0
    folder_path = f'./model_sonucları/{current_date}_{dosya_index}'
    while os.path.exists(folder_path):
        print(f"Folder '{folder_path}' already exists.")
        dosya_index += 1
        folder_path = f'./model_sonucları/{current_date}_{dosya_index}'
    folder_path += '/model_dir'
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")

    parser = argparse.ArgumentParser(description='Image_Change_Captioning')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Data parameters
    parser.add_argument('--data_folder', default="../MOSAIC-SEN2-CC/",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="MOSAIC-SEN2-CC_5_cap_per_img_5_min_word_freq",
                        help='base name shared by data files.')

    parser.add_argument('--encoder_image', default="resnet101", help='which model does encoder use?')
    parser.add_argument('--encoder_feat', default='MCCFormers_diff_as_Q')  #
    parser.add_argument('--decoder', default='trans')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim_de', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10,
                        help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--encoder_lr', type=float, default=5e-5, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=5e-5, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')

    parser.add_argument('--best_metric_mode', default='avg', choices=['bleu4', 'avg'],
                        help='Decides best model checkpoint selection metric: "bleu4" or "avg" over BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE')

    # parser.add_argument('--checkpoint', default=".\model_sonucları/20231103_5/model_dir/", help='path to checkpoint, None if none.')
    parser.add_argument('--checkpoint', default="None",
                        help='path to checkpoint, None if none.')
    # Validation
    parser.add_argument('--Split', default="VAL", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')
    parser.add_argument('--testing_beam_size', type=int, default=5, help='beam_size.')
    parser.add_argument('--savepath', default=folder_path)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parser.parse_args()
    terminal_output_dir = os.path.dirname(folder_path)
    main(args)
    subprocess.run(
        f"python eval_changed.py --data_folder {args.data_folder} "
        f"--terminal_output {terminal_output_dir} "
        f"--path {folder_path} "
        f"--beam_size {args.beam_size}")

    # subprocess.run(
    #     f"python eval_changed.py --data_folder {args.data_folder} --encoder_image {args.encoder_image} --terminal_output {folder_path.replace('/model_dir', '')} --path {folder_path} --beam_size {args.beam_size}")

    # subprocess.run(
    #         f"python eval_changed.py --data_folder {args.data_folder} --encoder_image {args.encoder_image} --terminal_output {folder_path.replace('/model_dir', '')} --path {folder_path} --beam_size {args.testing_beam_size}")
    #
    # subprocess.run(
    #         f"python captionGen_ACKonlytest_v2.py --data_folder {args.data_folder} --encoder_image {args.encoder_image} --terminal_output {folder_path.replace('/model_dir', '')} --path {folder_path} --beam_size {args.beam_size}")
