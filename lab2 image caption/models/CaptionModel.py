# This file contains ShowAttendTell

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

class CaptionModel(nn.Module):
    def __init__(self, opt):
        super(CaptionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size) # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        outputs = []
        alpha_list = []
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            alpha, output, state = self.core(xt, fc_feats, att_feats, state)
            alpha_list.append(alpha)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)
        
        alphas = torch.stack(alpha_list, dim=1)
        return alphas, torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        seqLogprobs = []
        alpha_list = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            alpha, output, state = self.core(xt, fc_feats, att_feats, state)
            alpha_list.append(alpha)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))
        
        alphas = torch.stack(alpha_list, dim=1) 
        
        return alphas, torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class ShowAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
		#======  TODO  =================================================================#
        # Set up the rnn (here is LSTM) 
        # Hint: self.rnn = getattr(nn, self.rnn_type.upper())(...) 
        #           input size: input_encoding_size + att_feat_size
        #           output size: rnn_size
        #           num_layers: num_layers
        #           bias: False
        #           dropout: self.drop_prob_lm
        # http://pytorch.org/docs/master/nn.html#lstm

        
        
		#=============================================================================#
        
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.h2rnn = nn.Linear(self.input_encoding_size, self.rnn_size)
        self.att2rnn = nn.Linear(self.att_feat_size, self.rnn_size)

    def forward(self, xt, fc_feats, att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        
        #======  TODO  =================================================================#
        # Implement Attention model
        # Hint: unsqueeze(), squeeze()      increase, decrease dimention
        #       view(size)                  resize tensor
        #       expand_as(Tensor)           expands tensor to the size of the specified tensor   
        #       torch.bmm()                 batch matrix-matrix product
        # http://pytorch.org/docs/master/torch.html?
        
        # V = resized image features(att_feats)
                # (batch * att_size) * att_feat_size
        # att = W*V
                # (batch * att_size) * att_hid_size
                # batch * att_size * att_hid_size
        # att_h = W*hidden
                # batch * att_hid_size
                # batch * att_size * att_hid_size
        # e = W*tanh(att+att_h)
                # batch * att_size * att_hid_size
                # batch * att_size * att_hid_size
                # (batch * att_size) * att_hid_size
                # (batch * att_size) * 1
                # batch * att_size
        # alpha = softmax(e)
                # batch * att_size
             
        # V = resized image features(att_feats)
                # batch * att_size * att_feat_size
        # C = alpha*V
                # batch * att_feat_size
                
        
        #=============================================================================#
        
        #======  TODO  =================================================================#
        # Use rnn to generate output
        
        # output, state = self.rnn(input, state)
        #       input: Concatenates(xt, C) in size=(1 * batch_size * input_size)
        
        
        #=============================================================================#
        
        
        return alpha, output, state

class ShowAttendTellModel(CaptionModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)
