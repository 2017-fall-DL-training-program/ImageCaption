# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils

loss_file = open("loss.txt",'w')

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    cnn_model = utils.build_cnn(opt)
    cnn_model.cuda()
    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
        # only finetune the layer2 to layer4
        cnn_optimizer = optim.Adam([\
            {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
            ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Update the training stage of cnn
            if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
                for p in cnn_model.parameters():
                    p.requires_grad = False
                cnn_model.eval()
            else:
                for p in cnn_model.parameters():
                    p.requires_grad = True
                # Fix the first few layers:
                for module in cnn_model._modules.values()[:5]:
                    for p in module.parameters():
                        p.requires_grad = False
                cnn_model.train()
            update_lr_flag = False

        torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        data['images'] = utils.prepro_images(data['images'], True)
        torch.cuda.synchronize()
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = cnn_model(images).permute(0, 2, 3, 1)   # output of ResNet (image fearures)
        fc_feats = att_feats.mean(2).mean(1)                # mean-pooled image fearures

        if not opt.use_att:
            att_feats = Variable(torch.FloatTensor(1,1,1,1).cuda())

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), opt.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * opt.seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), opt.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * opt.seq_per_img,) + fc_feats.size()[1:]))
        
        optimizer.zero_grad()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            cnn_optimizer.zero_grad()
        alpha, model_output = model(fc_feats, att_feats, labels)
        loss = crit(model_output, labels[:,1:], masks[:,1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            utils.clip_gradient(cnn_optimizer, opt.grad_clip)
            cnn_optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))
       

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader, eval_kwargs)

            # Write validation result into summary
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    print("cnn model saved to {}".format(cnn_checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
opt = opts.parse_opt()
train(opt)
loss_file.close
