import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

SMALL = 1e-08
loss_fn = nn.CrossEntropyLoss()


def getNumberAttr(dataset):
    data = pd.read_csv(dataset + "/train.csv", nrows=1)
    countAttr = 0
    for attr in list(data.keys()):
        if 'left' in attr:
            countAttr += 1
    return countAttr


class GroupMask(nn.Module):
    def __init__(self, args, sent1_len, sent2_len):
        super(GroupMask, self).__init__()
        self.device = args.device
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.rrelu}
        self.gnum = args.gnum
        self.local_bsz = 500
        self.initial_value = 1
        self.sigmoid = nn.Sigmoid()
        self.sent1_len = sent1_len
        self.sent2_len = sent2_len
        self.Z_mat = nn.Parameter(self.initial_value * torch.ones((self.sent1_len + self.sent2_len), self.gnum))
        self.T_vec = nn.Parameter(self.initial_value * torch.ones(self.gnum, 1))

    def _reset_value(self, initial_value=None):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.Z_mat.fill_(initial_value or self.initial_value)
            self.T_vec.fill_(initial_value or self.initial_value)
        return self.Z_mat, self.T_vec

    def forward(self, model, inputs, pred, flag):
        # embedding
        x = model.roberta.embeddings(inputs['input_ids'], inputs['token_type_ids'])
        attention_masks_r = inputs['attention_mask']
        input_ids_r = inputs['input_ids']

        if flag == 'train':
            # repeat local batch
            x = x.repeat(self.local_bsz, 1, 1)
            attention_masks_r = attention_masks_r.repeat(self.local_bsz, 1)
            input_ids_r = input_ids_r.repeat(self.local_bsz, 1)
            pred = pred.expand(self.local_bsz)

            z_mat = self.activations['sigmoid'](self.Z_mat)
            z_prob = F.softmax(z_mat, dim=1)
            z_mat_batch = z_mat.unsqueeze(0).repeat(self.local_bsz, 1, 1)
            z_s = F.gumbel_softmax(z_mat_batch, dim=2, tau=0.5, hard=True)

            t_vec = self.activations['sigmoid'](self.T_vec)
            t_prob = F.softmax(t_vec, dim=0)
            t_vec_batch = t_vec.unsqueeze(0).repeat(self.local_bsz, 1, 1)
            t_s = F.gumbel_softmax(t_vec_batch, dim=1, tau=0.5, hard=True)
            x_s = torch.bmm(z_s, t_s)

            x_prime = torch.cat((
                x_s.new_ones(x_s.shape[0], 1, x_s.shape[2]),  # <s> inalterato
                x_s[:, :self.sent1_len, :],
                x_s.new_ones(x_s.shape[0], 1, x_s.shape[2]),  # </s> inalterato
                x_s.new_ones(x_s.shape[0], 1, x_s.shape[2]),  # </s> inalterato
                x_s[:, self.sent1_len:, :],
                x_s.new_ones(x_s.shape[0], x.shape[1] - x_s.shape[1] - 3, x_s.shape[2])),
                dim=1) * x
            attention_masks_r = torch.cat((x_s.new_ones(x_s.shape[0], 1),
                                           x_s[:, :self.sent1_len].squeeze(-1),
                                           x_s.new_ones(x_s.shape[0], 2),
                                           x_s[:, self.sent1_len:].squeeze(-1),
                                           x_s.new_ones(x_s.shape[0], x.shape[1] - x_s.shape[1] - 3))
                                          , dim=1) * attention_masks_r

            input_ids_r = torch.cat((x_s.new_ones(x_s.shape[0], 1), x_s[:, :self.sent1_len].squeeze(-1),
                                     x_s.new_ones(x_s.shape[0], 1), x_s[:, self.sent1_len:].squeeze(-1),
                                     x_s.new_ones(x_s.shape[0], x.shape[1] - x_s.shape[1] - 2)), dim=1) * input_ids_r

            z_margin1 = torch.sum(z_prob[:self.sent1_len, :], dim=0)
            z_margin1_p = F.softmax(z_margin1, dim=0)
            self.z_loss1 = torch.sum(z_margin1_p * torch.log(z_margin1_p + 1e-8))
            z_margin2 = torch.sum(z_prob[self.sent1_len:, :], dim=0)
            z_margin2_p = F.softmax(z_margin2, dim=0)
            self.z_loss2 = torch.sum(z_margin2_p * torch.log(z_margin2_p + 1e-8))

            x_prob = torch.mm(z_prob, t_prob)
            x_prob = F.softmax(x_prob, dim=0)
            self.t_loss = torch.sum(-t_prob * torch.log(t_prob + 1e-8))

            output = model(inputs_embeds=x_prime.long(), token_type_ids=inputs['token_type_ids'],
                           attention_mask=attention_masks_r, labels=pred)
            pred_loss, out_logits = output[:2]
            _, out_pred = out_logits.max(dim=1)

            return pred_loss, out_pred, z_prob, t_prob, x_prob


        else:
            z_prob = F.softmax(self.activations['sigmoid'](self.Z_mat), dim=1).unsqueeze(0)
            t_prob = F.softmax(self.activations['sigmoid'](self.T_vec), dim=0).unsqueeze(0)
            x_prob = torch.bmm(z_prob, t_prob)
            x_prime = torch.cat((
                x_prob.new_ones(x_prob.shape[0], 1, x_prob.shape[2]),  # <s> inalterato
                x_prob[:, :self.sent1_len, :],
                x_prob.new_ones(x_prob.shape[0], 1, x_prob.shape[2]),  # </s> inalterato
                x_prob.new_ones(x_prob.shape[0], 1, x_prob.shape[2]),  # </s> inalterato
                x_prob[:, self.sent1_len:, :],
                x_prob.new_ones(x_prob.shape[0], x.shape[1] - x_prob.shape[1] - 3, x_prob.shape[2])),
                dim=1) * x

            output = model(inputs_embeds=x_prime.long(), token_type_ids=inputs['token_type_ids'],
                           attention_mask=attention_masks_r, labels=pred)
            pred_loss, out_logits = output[:2]
            _, out_pred = out_logits.max(dim=1)

            z_max_mat = z_prob.new_zeros(z_prob.shape)
            z_max_idx = z_prob.argmax(dim=2)[0]
            for i, v in enumerate(z_max_idx):
                z_max_mat[0][i][v] = 1

            return pred_loss, out_pred, z_max_mat, t_prob, x_prob, z_max_idx


class WordMask(nn.Module):
    def __init__(self, args, sent1_len, sent2_len):
        super(WordMask, self).__init__()
        self.device = args.device
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.rrelu}
        self.local_bsz = 10
        self.initial_value = 1
        self.sigmoid = nn.Sigmoid()
        self.sent1_len = sent1_len
        self.sent2_len = sent2_len
        self.R_vec = nn.Parameter(self.initial_value * torch.ones((self.sent1_len + self.sent2_len), 2))

    def _reset_value(self, initial_value=None):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.R_vec.fill_(initial_value or self.initial_value)
        return self.R_vec

    def forward(self, model, inputs, pred, flag):
        # embedding
        x_left = model.encoder.transformer.embeddings(inputs['input_ids'])
        x_right = model.encoder.transformer.embeddings(inputs['input_ids_right'])
        left_input_ids_r = inputs['input_ids']
        left_attention_masks = inputs['attention_mask']
        right_input_ids_r = inputs['input_ids_right']
        right_attention_masks = inputs['attention_mask_right']

        if flag == 'train':
            # repeat local batch
            x_left = x_left.repeat(self.local_bsz, 1, 1)
            left_input_ids = left_input_ids_r.repeat(self.local_bsz, 1)
            left_attention_masks = left_attention_masks.repeat(self.local_bsz, 1)
            x_right = x_right.repeat(self.local_bsz, 1, 1)
            right_input_ids = right_input_ids_r.repeat(self.local_bsz, 1)
            right_attention_masks = right_attention_masks.repeat(self.local_bsz, 1)
            pred = pred.expand(self.local_bsz)

            r_vec = self.activations['sigmoid'](self.R_vec)
            r_vec_batch = r_vec.unsqueeze(0).repeat(self.local_bsz, 1, 1)
            r_s = F.gumbel_softmax(r_vec_batch, dim=2, hard=True)[:, :, 1:2]
            r_prob = F.softmax(r_vec, dim=1)[:, 1:2]

            left_x_prime = torch.cat((
                r_s.new_ones(r_s.shape[0], 1, r_s.shape[2]),  # <s> unchanged
                r_s[:, :self.sent1_len, :],
                r_s.new_ones(r_s.shape[0], x_left.shape[1] - self.sent1_len - 1, r_s.shape[2])),
                dim=1) * x_left
            right_x_prime = torch.cat((
                r_s.new_ones(r_s.shape[0], 1, r_s.shape[2]),  # <s> unchanged
                r_s[:, self.sent1_len:, :],
                r_s.new_ones(r_s.shape[0], x_right.shape[1] - self.sent2_len - 1, r_s.shape[2])),
                dim=1) * x_right
            left_attention_masks_r = torch.cat((
                r_s.new_ones(r_s.shape[0], 1),
                r_s[:, :self.sent1_len].squeeze(-1),
                r_s.new_ones(r_s.shape[0], x_left.shape[1] - self.sent1_len - 1)),
                dim=1) * left_attention_masks
            right_attention_masks_r = torch.cat((
                r_s.new_ones(r_s.shape[0], 1),
                r_s[:, self.sent1_len:].squeeze(-1),
                r_s.new_ones(r_s.shape[0], x_right.shape[1] - self.sent2_len - 1)),
                dim=1) * right_attention_masks
            left_input_ids_r = torch.cat((
                r_s.new_ones(r_s.shape[0], 1),
                r_s[:, :self.sent1_len].squeeze(-1),
                r_s.new_ones(r_s.shape[0], x_left.shape[1] - self.sent1_len - 1)),
                dim=1) * left_input_ids
            right_input_ids_r = torch.cat((
                r_s.new_ones(r_s.shape[0], 1),
                r_s[:, self.sent1_len:].squeeze(-1),
                r_s.new_ones(r_s.shape[0], x_right.shape[1] - self.sent2_len - 1)),
                dim=1) * right_input_ids

            self.r_loss = torch.sum(r_prob * torch.log(r_prob + 1e-8))

            output = model(
                input_ids=left_input_ids_r.long(), input_ids_right=right_input_ids_r.long(),
                inputs_embeds=left_x_prime.long(), inputs_embeds_right=right_x_prime.long(),
                attention_mask=left_attention_masks_r, attention_mask_right=right_attention_masks_r,
                labels=pred
            )
            pred_loss, out_logits = output[:2]
            out_pred = out_logits.flatten()
            out_pred[out_pred >= 0.5] = 1
            out_pred[out_pred < 0.5] = 0

            return pred_loss, r_prob

        else:
            r_prob = F.softmax(self.activations['sigmoid'](self.R_vec), dim=1)[:, 1:2].unsqueeze(0)
            x_prime = torch.cat((r_prob.new_ones(r_prob.shape[0], 1, r_prob.shape[2]), r_prob[:, :self.sent1_len, :],
                                 r_prob.new_ones(r_prob.shape[0], 1, r_prob.shape[2]), r_prob[:, self.sent1_len:, :],
                                 r_prob.new_ones(r_prob.shape[0], x.shape[1] - r_prob.shape[1] - 2, r_prob.shape[2])),
                                dim=1) * x

            output = model(inputs_embeds=x_prime.long(), token_type_ids=inputs['token_type_ids'],
                           attention_mask=attention_masks_r, labels=pred)
            pred_loss, out_logits = output[:2]
            _, out_pred = out_logits.max(dim=1)

            return pred_loss, r_prob


def mostImpWord(args, model, inputs, pred):
    model.eval()
    sep_idx = [d for d, i in enumerate(inputs['input_ids'][0]) if int(i) == 102]
    sent1_len = sep_idx[0] - 1
    sent2_len = sep_idx[1] - sep_idx[0] - 1

    # train word masks
    wmask = WordMask(args, sent1_len, sent2_len)
    wmask.to(torch.device(args.device))
    wmask._reset_value(wmask.initial_value)
    for param in wmask.parameters():
        param.requires_grad = True

    wmask_optimizer = torch.optim.Adam(wmask.parameters(), lr=0.1)
    w_beta = 0.1
    w_epochs = 10
    for _ in range(w_epochs):
        wmask_optimizer.zero_grad()
        pred_loss, r_prob = wmask(model, inputs, pred, 'train')
        loss = pred_loss + w_beta * wmask.r_loss
        loss.backward()
        wmask_optimizer.step()
    _, r_prob = wmask(model, inputs, pred, 'test')

    # select top important words
    kw = min(20, sent1_len + sent2_len)  # sent1_len + sent2_len # min(10, sent1_len + sent2_len)
    x_imp = r_prob.squeeze(0).squeeze(-1).detach().cpu().numpy()
    return x_imp


def interpret(args, model, inputs, pred):
    model.eval()
    sent1_len = torch.where(inputs['input_ids'][0] == 2)[0].item() - 1
    sent2_len = torch.where(inputs['input_ids_right'][0] == 2)[0].item() - 1

    # train word masks
    wmask = WordMask(args, sent1_len, sent2_len)
    wmask.to(torch.device(args.device))
    wmask._reset_value(wmask.initial_value)
    for param in wmask.parameters():
        param.requires_grad = True

    wmask_optimizer = torch.optim.Adam(wmask.parameters(), lr=0.1)
    w_beta = 0.1
    w_epochs = 10
    print(w_epochs)
    for _ in range(w_epochs):
        wmask_optimizer.zero_grad()
        pred_loss, r_prob = wmask(model, inputs, pred, 'train')
        loss = pred_loss + w_beta * wmask.r_loss
        loss.backward()
        wmask_optimizer.step()
    _, r_prob = wmask(model, inputs, pred, 'test')

    kw = min(100, sent1_len + sent2_len)  # sent1_len + sent2_len # min(10, sent1_len + sent2_len)
    x_imp = r_prob.squeeze(0).squeeze(-1).detach().cpu().numpy()
    x_ids = np.argpartition(x_imp, -len(x_imp))[-len(x_imp):]
    x_ids = x_ids[np.argsort(-x_imp[x_ids])]

    for i, v in enumerate(x_ids):
        if v < (sep_idx[0]):
            x_ids[i] += 1
        elif v > sep_idx[1]:
            x_ids[i] += 3

    skipword = ['[COL]', '[VAL]', 'COL', 'VAL', 'PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME',
                'Artist_Name', 'name',
                'Released', 'CopyRight', 'content', 'Brew_Factory_Name',
                'Time', 'type', 'Beer_Name', 'category', 'price', 'title', 'authors', 'class', 'description',
                'Song_Name', 'venue', 'brand', 'Genre', 'year', 'manufacturer', 'Style', 'addr', 'phone',
                'modelno', 'Price', 'ABV', 'city', 'Album_Name', 'specTableContent']

    sent1_r = []
    sent2_r = []
    sent1_r_ids = []
    sent2_r_ids = []
    wordFilled = 0
    for v in x_ids:
        if v > sep_idx[1]:
            wordGet = args.input_words[v - 3].replace(' ', '')
            wordGet = wordGet.replace('Ġ', '')
            if len(wordGet) > 1 and not wordGet in skipword:
                wordFilled += 1
                sent2_r.append(inputs['input_ids'][0][v])
                sent2_r_ids.append(v)
        else:
            wordGet = args.input_words[v - 1].replace(' ', '')
            wordGet = wordGet.replace('Ġ', '')
            if len(wordGet) > 1 and not wordGet in skipword:
                wordFilled += 1
                sent1_r.append(inputs['input_ids'][0][v])
                sent1_r_ids.append(v)
        if wordFilled == 20:
            break

    if sent1_r == [] or sent2_r == []:
        for v in x_ids[kw:]:
            if v >= sep_idx[0]:
                sent2_r.append(inputs['input_ids'][0][v])
                sent2_r_ids.append(v)
            else:
                sent1_r.append(inputs['input_ids'][0][v])
                sent1_r_ids.append(v)
        sent1_r = sent1_r[:min(len(sent1_r), int(kw / 2))]
        sent1_r_ids = sent1_r_ids[:min(len(sent1_r), int(kw / 2))]
        sent2_r = sent2_r[:min(len(sent2_r), int(kw / 2))]
        sent2_r_ids = sent2_r_ids[:min(len(sent2_r), int(kw / 2))]

    args.gnum = getNumberAttr(args.data_dir)  # max(2, min(len(sent1_r), len(sent2_r)))
    args.T_name = ['T_' + str(t + 1) for t in range(args.gnum)]
    sent1_r_len = len(sent1_r)
    sent2_r_len = len(sent2_r)
    sent_pad_ids = sent1_r_ids + sent2_r_ids
    sent_rest_ids = [i for i in x_ids if i not in sent_pad_ids]
    batch_r_1 = torch.LongTensor(sent1_r).to(args.device)
    batch_r_2 = torch.LongTensor(sent2_r).to(args.device)
    batch_r_com = torch.cat((torch.LongTensor([0]).to(args.device), batch_r_1, torch.LongTensor([2]).to(args.device),
                             torch.LongTensor([2]).to(args.device),
                             batch_r_2, torch.LongTensor([2]).to(args.device)), dim=0)
    inputs_r = {'input_ids': batch_r_com.unsqueeze(0),
                'attention_mask': inputs['attention_mask'].new_ones(1, batch_r_com.shape[0]),
                'token_type_ids': None,
                'labels': pred}
    # inputs_r['token_type_ids'][0][:(len(batch_r_1) + 2)] = 0

    top_words = []
    for i in sent_pad_ids:
        if i < sep_idx[0]:
            top_words.append(args.input_words[i - 1])
        elif i > sep_idx[1]:
            top_words.append(args.input_words[i - 3])
    args.top_words = top_words

    gmask = GroupMask(args, sent1_r_len, sent2_r_len)
    gmask.to(torch.device(args.device))
    gmask._reset_value(gmask.initial_value)
    for param in gmask.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(gmask.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    alpha = 1
    beta = 1
    gamma = 10
    epochs = 10
    for ep in range(epochs):
        optimizer.zero_grad()
        pred_loss, _, z_prob, t_prob, x_prob = gmask(model, inputs_r, pred, 'train')
        loss = alpha * pred_loss + beta * gmask.t_loss + gamma * (gmask.z_loss1 + gmask.z_loss2)
        loss.backward()
        optimizer.step()
        scheduler.step()

    _, output, z_prob, t_prob, x_prob, z_max_idx = gmask(model, inputs_r, pred, 'test')
    x_r_imp = x_prob.squeeze(0).squeeze(-1).detach().cpu().numpy()
    x_r_ids = np.argpartition(x_r_imp, -len(x_r_imp))[-len(x_r_imp):]
    x_r_ids = x_r_ids[np.argsort(-x_r_imp[x_r_ids])]

    x_com_ids = []
    for i in x_r_ids:
        x_com_ids.append(sent_pad_ids[i])
    x_com_ids += sent_rest_ids

    return output, z_prob, t_prob, x_com_ids, z_max_idx, sent1_len, x_imp
