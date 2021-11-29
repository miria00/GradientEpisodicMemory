# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

from jax import jit
from neural_tangents import stax
import neural_tangents as nt
from bilevel_coresets import bilevel_coreset, loss_utils

import numpy as np
import quadprog

from .common import MLP, ResNet18, ConvNet
# only for permuted MNIST case 

# Auxiliary functions useful for GEM's inner optimization.


def compute_offsets(task, nc_per_task, is_cifar):
    """
    Compute offsets for cifar to determine which
    outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
    pp: parameters
    newgrad: corrected gradient
    grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
    Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Overwrites "gradient" with the final projected update.

    input:  gradient, p-vector
    input:  memories, (t * p)-vector
    output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = args.data_file == "cifar100.pt"
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            print("n_inputs: ", n_inputs)
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            # self.net = ConvNet(n_outputs)

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
            
        # create NTK for ConvNet (aka MLP?) for use by coreset
        # _, _, kernel_fn = stax.serial(
        # stax.Conv(32, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
        # stax.Relu(),
        # stax.Conv(64, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
        # stax.Relu(),
        # stax.Flatten(),
        # stax.Dense(128, 1., 0.05),
        # stax.Relu(),
        # stax.Dense(10, 1., 0.05))
        # kernel_fn = jit(kernel_fn, static_argnums=(2,))

        # create NTK for MLP for use by coreset
        init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(100, 1., 0.05),
        stax.Relu(),
        stax.Dense(100, 1., 0.05),
        stax.Relu(),
        stax.Dense(10, 1., 0.05))
        kernel_fn = jit(kernel_fn, static_argnums=(2,))

        def generate_fnn_ntk(X, Y):
            return np.array(kernel_fn(X, Y, 'ntk'))


        self.proxy_kernel_fn = lambda x, y: generate_fnn_ntk(x.reshape(-1, 28 * 28).numpy(), y.reshape(-1, 28 * 28).numpy())
        # self.proxy_kernel_fn = lambda x, y: generate_cnn_ntk(x.view(-1, 28, 28, 1).numpy(), y.view(-1, 28, 28, 1).numpy())

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2 : self.n_outputs].data.fill_(-10e10)
        return output

    # def observe(self, x, t, y):
    #     # update memory
    #     if t != self.old_task:
    #         self.observed_tasks.append(t)
    #         self.old_task = t

    #     # update ring buffer storing examples from current task
    #     bsz = y.data.size(0)
    #     endcnt = min(self.mem_cnt + bsz, self.n_memories)
    #     effbsz = endcnt - self.mem_cnt
    #     self.memory_data[t, self.mem_cnt : endcnt].copy_(x.data[:effbsz])
    #     if bsz == 1:
    #         self.memory_labs[t, self.mem_cnt] = y.data[0]
    #     else:
    #         self.memory_labs[t, self.mem_cnt : endcnt].copy_(y.data[:effbsz])
    #     self.mem_cnt += effbsz
    #     if self.mem_cnt == self.n_memories:
    #         self.mem_cnt = 0

    # solve_coreset via https://github.com/zalanborsos/bilevel_coresets 
    def solve_coreset(self, subset_size, x, y):
        print("A")
        # bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.weighted_mse,
        #                                     inner_loss_fn=loss_utils.weighted_mse, out_dim=1, max_outer_it=1,
        #                                     inner_lr=0.00025, max_inner_it=500, logging_period=100)

        bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                            inner_loss_fn=loss_utils.cross_entropy, out_dim=10, max_outer_it=1,
                                            max_inner_it=200, logging_period=1000)
        print("B")
        print(x)
        print(y)
        coreset_inds, _ = bc.build_with_representer_proxy_batch(x, y, subset_size, kernel_fn_np=self.proxy_kernel_fn,
                                                                cache_kernel=True, start_size=1, inner_reg=1e-7)
        print("C")
        print("coreset_inds[:subset_size]: ", coreset_inds[:subset_size])
        return coreset_inds[:subset_size]

    # does observe get called multiple times per task?
    # if so, we should accumulate data for each task to get entirety, 
    # then feed entirety into coresets to get the desired subset
    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt

        # if we have enough memory for whole batch, store entire batch
        if effbsz == bsz:    
            self.memory_data[t, self.mem_cnt: endcnt].copy_(
                x.data[: effbsz])
        # otherwise we prune using coresets
        else:
            subset_inds = self.solve_coreset(effbsz, x, y)       

            # index x.data to get just subset elements, and store those into memory_data           
            sub_elements = x.data[subset_inds]
            
            self.memory_data[t, self.mem_cnt: endcnt].copy_(sub_elements)
            
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(
                    past_task, self.nc_per_task, self.is_cifar
                )
                ptloss = self.ce(
                    self.forward(self.memory_data[past_task], past_task)[
                        :, offset1:offset2
                    ],
                    self.memory_labs[past_task] - offset1,
                )
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1:offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = (
                torch.cuda.LongTensor(self.observed_tasks[:-1])
                if self.gpu
                else torch.LongTensor(self.observed_tasks[:-1])
            )
            dotp = torch.mm(
                self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx)
            )
            if (dotp < 0).sum() != 0:
                project2cone2(
                    self.grads[:, t].unsqueeze(1),
                    self.grads.index_select(1, indx),
                    self.margin,
                )
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
        self.opt.step()
