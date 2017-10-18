import argparse
import numpy as np
import mxnet as mx
import logging

logging.basicConfig(level=logging.DEBUG)
from MaskIter import MaskIter
from symbol_CDCNN import *

def get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
        
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

if __name__ == "__main__":
    # python train.py --model-prefix models/CDCNN_LS --short-skip True --long-skip Concat --gpus 0,1,2,3
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-prefix', type=str, help='model prefix', default="CDCNN_L")
    parser.add_argument('--continue-training', type=bool, help='model prefix', default=False)
    parser.add_argument('--num-epochs', type=int, help='model prefix', default=250)
    parser.add_argument('--short-skip', type=bool, help='model prefix', default=False)
    parser.add_argument('--long-skip', type=str, help='model prefix', default="Concat")
    parser.add_argument('--kv-store', type=str, help='key-value store type', default='device')
    parser.add_argument('--gpus', type=str, help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    parser.add_argument('--batch-size', type=int, help='batch size', default=16)
    parser.add_argument('--disp-batches', type=int, default=20, help='show progress for every n batches')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0, help='log network parameters every N iters if larger than 0')
    parser.add_argument('--lr', type=float, default=0.03, help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-step-epochs', type=str, help='the epochs to reduce the lr, e.g. 45,90')
    parser.add_argument('--num-examples', type=int, default=4584, help='the ratio to reduce lr on each step')
    parser.add_argument('--load-epoch', type=int, default=0, help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()

    kv = mx.kvstore.create(args.kv_store)
    CDCNN = CDCNN(short_skip=args.short_skip, long_skip=args.long_skip)
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    lr, lr_scheduler = get_lr_scheduler(args, kv)
    optimizer_params = {
            'learning_rate': 0.003,
            'wd' : 0.0000000001,
            'lr_scheduler': lr_scheduler}
    
    data_train = MaskIter(root_dir="MelanomaISIC/", flist_name="train.lst", batch_size=args.batch_size, augment=True, shuffle=True)
    data_val = MaskIter(root_dir="MelanomaISIC/", flist_name="val.lst", batch_size=args.batch_size, augment=False)

    
    
    
    if args.continue_training == False:
        model = mx.mod.Module(
            context       = devs,
            symbol        = CDCNN)
        model.fit(data_train,
            begin_epoch        = 0,
            num_epoch          = args.num_epochs,
            eval_data          = data_val,
            kvstore            = kv,
            optimizer          = "adam",
            optimizer_params   = optimizer_params,
            eval_metric        = CustomLoss(),
            batch_end_callback = batch_end_callbacks,
            epoch_end_callback = mx.callback.do_checkpoint(args.model_prefix if kv.rank == 0 else "%s-%d" % (args.model_prefix, kv.rank)),
            allow_missing      = True,
            monitor            = monitor)
    else:
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
        model = mx.mod.Module(
            context       = devs,
            symbol        = CDCNN)
        model.fit(data_train,
            begin_epoch        = args.load_epoch,
            num_epoch          = args.num_epochs,
            eval_data          = data_val,
            kvstore            = kv,
            optimizer          = "adam",
            optimizer_params   = optimizer_params,
            eval_metric        = CustomLoss(),
            batch_end_callback = batch_end_callbacks,
            epoch_end_callback = mx.callback.do_checkpoint(args.model_prefix if kv.rank == 0 else "%s-%d" % (args.model_prefix, kv.rank)),
            allow_missing      = True,
            monitor            = monitor)



