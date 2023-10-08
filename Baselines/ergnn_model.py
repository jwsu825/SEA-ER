# most of the ergnn code are modifed from CGLB repo
import torch
import copy
from .ergnn_utils import *
import pickle
from .kmm import *

from Backbones.utils import evaluate, NodeLevelDataset, evaluate_batch

samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True), 'MF':MF_sampler(plus=False), 'MF_plus':MF_sampler(plus=True),'random':random_sampler(plus=False)}
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.ergnn_args['sampler']]

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = int(args.ergnn_args['budget'])
        self.d_CM = args.ergnn_args['d'] # d for CM sampler of ERGNN
        self.aux_g = None

        #set up memories for old rep
        self.rep_oldt_old = {}

    def forward(self, features):
        output = self.net(features)
        return output

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        if t != 0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                self.rep_oldt_old[oldt] = self.net.second_last_h
                

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = 0
            for k in self.buffer_node_ids:
                buffer_size += len(self.buffer_node_ids[k])
            beta = buffer_size / (buffer_size + n_nodes_current_batch)
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            
            #get new rep
            new_task_rep = self.net.second_last_h

            # sample and store ids from current task
            if t != self.current_task:
                self.current_task = t
                sampled_ids = self.sampler(ids_per_cls_train, self.budget, features.to(device='cuda:{}'.format(args.gpu)), self.net.second_last_h, self.d_CM)
                old_ids = g.ndata['_ID'].cpu()
                self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
                ag, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
                self.aux_g.append(ag.to(device='cuda:{}'.format(args.gpu)))
                if args.cls_balance:
                    n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)

            if t != 0:
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1:o2])
                    loss = beta * loss + (1 - beta) * loss_aux

                    if args.ir:
                        label_balance_constraints = np.zeros((labels.max().item()+1, len(train_ids)))
                        for i, idx in enumerate(train_ids):
                            label_balance_constraints[labels[idx], i] = 1
                        old_task_rep = self.net.second_last_h
                        kmm_weight = kernel_mean_matching(old_task_rep.detach().cpu(),torch.tensor(self.rep_oldt_old[oldt]).detach().cpu())
                        loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()

            loss.backward()
            self.opt.step()

   