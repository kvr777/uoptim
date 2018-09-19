from torch.optim import Adam, Adamax, SGD, Adadelta, RMSprop


import math
import torch
from torch.optim import Optimizer




class UOptimizer(Optimizer):
    """Implements universal constructor for gradient descent optimization in PyTorch

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        use_exp_avg_norm (bool, optional): enables use exponential average gradients norm
                          momentum or nesterov momentum (default: False)
        use_exp_avg_sq_norm (bool, optional): enables use exponential average square gradients
                            in denominator (default: False)
        exp_avg_norm_type (string optional): select type of exponential average gradients norm
                        ['momentum', 'nesterov']. The general formula is:
                        exp_avg.mul(beta1).add(1 - beta1_dump, grad). In standard
                        optimizers two versions are available: SGD use different
                        beta1 and beta1_dump (default for beta1_dump is 0). In
                        adam-like optimizers beta1=beta1_dump. So you can specify
                        beta1_dump if needed (default 'momentum')
        exp_avg_sq_norm_type (string optional): select type of exponential average squared gradients norm
                        Three options are possible:
                            - 'classic' - the formula which is used in Adam optimizer
                            - 'infinite_l' - use the formula from Adamax algorith
                            (a variant of Adam based on infinity norm).
                            For more detail see https://arxiv.org/abs/1412.6980
                            - 'max_past_sq' -  uses the maximum of past squared gradients vt
                            rather than the exponential average to update the parameters.
                            For more details see https://openreview.net/forum?id=ryQu7f-RZ
                        (default 'classic')
        use_bias_correction (bool, optional): enables corrections of beta1 and beta2 like in
                        Adam optimizers to adapt step size multiplicator. If false, step size
                        multiplicator equals to learning rate (default: False)
        use_adadelta_lr (bool, optional): if True, nominator is multiplied by accumulated delta parameter
                        that computed  according to Adadelta algorithm.(default: False)
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta1_dump (float, optional): use if you need specify beta1 decay when computing
                running averages of gradient (see description of exp_avg_norm_type). If None,
                beta1_dump equals to beta1. (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decouple_wd (bool, optional): if True, instead of apply weight_decay (or L2 regularization)
                to gradient directly before each step (like in standard algorithms), we apply it after
                and take into consideration step size. See more details in https://arxiv.org/abs/1711.05101
                (default: False)
        eps_under_root (bool, optional): in some classical algorithms epsilon in denominator can be under square root
        (Adadelta), but in Adam-like algorithms it's added after. To preserve the flexibility it is possible
        to choose this option. (default: False)
        optimizer (string, optional): to make the life easier, user can select classical algorithm from list:
        {SGDM (SGD with momentum), SGDNM (SGD with Nesterov Momentum), Adadelta, RMSProp, Adam, Adamax, Nadam, AdamW,
        Amsgrad}. Leave None if you need vanilla SGD, or setup non-standard optimizer.

    """

    def __init__(self,
                 params,
                 use_exp_avg_norm=False,
                 use_exp_avg_sq_norm=False,
                 exp_avg_norm_type='momentum',
                 exp_avg_sq_norm_type='classic',
                 use_bias_correction=False,
                 use_adadelta_lr=False,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 beta1_dump=None,
                 eps=1e-8,
                 weight_decay=0,
                 decouple_wd=False,
                 eps_under_root=False,
                 optimizer=None):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2: {}".format(betas[1]))
        if beta1_dump is not None and beta1_dump < 0.0:
            raise ValueError("Invalid beta1_dump: {}".format(beta1_dump))

        defaults = dict(use_exp_avg_norm=use_exp_avg_norm,
                        use_exp_avg_sq_norm=use_exp_avg_sq_norm,
                        exp_avg_norm_type=exp_avg_norm_type,        # ['momentum', 'nesterov']
                        exp_avg_sq_norm_type=exp_avg_sq_norm_type,  # ['classic', 'infinite_l', 'max_past_sq']
                        use_bias_correction=use_bias_correction,
                        use_adadelta_lr=use_adadelta_lr,
                        lr=lr,
                        betas=betas,
                        beta1_dump=beta1_dump,
                        eps=eps,
                        weight_decay=weight_decay,
                        decouple_wd=decouple_wd,
                        eps_under_root=eps_under_root,
                        optimizer=optimizer)

        optimizers_dict = {
            'SGDM': {'use_exp_avg_norm': True},
            'SGDNM': {'use_exp_avg_norm': True,
                      'exp_avg_norm_type': 'nesterov'},
            'Adadelta': {'use_exp_avg_sq_norm': True,
                         'use_adadelta_lr': True,
                         'eps_under_root': True},
            'RMSProp': {'use_exp_avg_sq_norm': True},
            'Adam': {'use_exp_avg_norm': True,
                     'use_exp_avg_sq_norm': True,
                     'use_bias_correction': True},
            'Adamax': {'use_exp_avg_norm': True,
                       'use_exp_avg_sq_norm': True,
                       'use_bias_correction': True,
                       'exp_avg_sq_norm_type': 'infinite_l'},
            'Nadam': {'use_exp_avg_norm': True,
                      'use_exp_avg_sq_norm': True,
                      'use_bias_correction': True,
                      'exp_avg_sq_norm_type': 'nesterov'},
            'AdamW': {'use_exp_avg_norm': True,
                      'use_exp_avg_sq_norm': True,
                      'use_bias_correction': True,
                      'decouple_wd': True},
            'Amsgrad': {'use_exp_avg_norm': True,
                        'use_exp_avg_sq_norm': True,
                        'use_bias_correction': True,
                        'exp_avg_sq_norm_type': 'max_past_sq'},
        }

        if optimizer is not None:
            if optimizer not in optimizers_dict.keys():
                raise ValueError("Invalid optimizer: {}".format(optimizer))
            else:
                defaults.update(optimizers_dict[optimizer])

        super(UOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('use_exp_avg_norm', False)
        for group in self.param_groups:
            group.setdefault('use_exp_avg_sq_norm', False)
        for group in self.param_groups:
            group.setdefault('decouple_wd', False)
        for group in self.param_groups:
            group.setdefault('use_bias_correction', False)
        for group in self.param_groups:
            group.setdefault('use_adadelta_lr', False)
        for group in self.param_groups:
            group.setdefault('eps_under_root', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                use_exp_avg_norm = group['use_exp_avg_norm']
                use_exp_avg_sq_norm = group['use_exp_avg_sq_norm']
                exp_avg_norm_type = group['exp_avg_norm_type']
                exp_avg_sq_norm_type = group['exp_avg_sq_norm_type']
                use_bias_correction = group['use_bias_correction']
                use_adadelta_lr=group['use_adadelta_lr']
                decouple_wd = group['decouple_wd']
                eps = group['eps']
                eps_under_root = group['eps_under_root']
                state = self.state[p]

                if grad.is_sparse and (use_exp_avg_norm or use_exp_avg_sq_norm or use_adadelta_lr):
                    raise RuntimeError('The optimizer with selected parameters do not works with sparse grad.' 
                                       'Consider do not use exponential normalization or adadelta lr')

                # STATE INITIALIZATION PHASE
                if len(state) == 0:
                    state['step'] = 0

                    if use_exp_avg_norm:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                    if use_exp_avg_sq_norm:
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                        if exp_avg_sq_norm_type == 'max_past_sq':
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                        elif exp_avg_sq_norm_type == 'infinite_l':
                            state['exp_inf'] = torch.zeros_like(p.data)
                    if use_adadelta_lr:
                        state['acc_delta'] = torch.zeros_like(p.data)

                if use_exp_avg_norm:
                    exp_avg = state['exp_avg']

                if use_exp_avg_sq_norm:
                    exp_avg_sq = state['exp_avg_sq']

                    if exp_avg_sq_norm_type == 'max_past_sq':
                        max_exp_avg_sq = state['max_exp_avg_sq']

                    elif exp_avg_sq_norm_type == 'infinite_l':
                        exp_inf = state['exp_inf']

                if use_adadelta_lr:
                    acc_delta = state['acc_delta']

                beta1, beta2 = group['betas']

                state['step'] += 1

                denom , numerator = None, None

                # use weight decay for grad if needed
                if not decouple_wd and group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)


                # UPDATE REGULAZATION PARAMETERS

                # update exp_avg_norm
                if use_exp_avg_norm:
                    beta1_dump = beta1 if group['beta1_dump'] is None else group['beta1_dump']
                    exp_avg.mul_(beta1).add_(1 - beta1_dump, grad)
                    numerator = exp_avg
                else:
                    numerator = grad

                if use_adadelta_lr:
                    delta_numerator = acc_delta.add(eps).sqrt_()
                    if numerator is None:
                        numerator = delta_numerator
                    else:
                        numerator = delta_numerator.mul_(numerator)

                # Decay the second moment running average coefficient if needed
                if use_exp_avg_sq_norm:

                    # update exp_avg_norm
                    if exp_avg_sq_norm_type != 'infinite_l':
                        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    # calculate denominator
                    if exp_avg_sq_norm_type == 'max_past_sq':
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        if not eps_under_root:
                            denom = max_exp_avg_sq.sqrt().add_(eps)
                        else:
                            denom = max_exp_avg_sq.add(eps).sqrt_()


                    elif exp_avg_sq_norm_type == 'infinite_l':
                        # Update the exponentially weighted infinity norm.
                        norm_buf = torch.cat([
                            exp_inf.mul_(beta2).unsqueeze(0),
                            grad.abs().add_(eps).unsqueeze_(0)
                        ], 0)
                        torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                        denom = exp_inf

                    else:
                        if not eps_under_root:
                            denom = exp_avg_sq.sqrt().add_(eps)
                        else:
                            denom = exp_avg_sq.add(eps).sqrt_()

                # calculate lr corrections:
                if use_bias_correction:
                    # calculate bias corrections.
                    if use_exp_avg_norm:
                        bias_correction1 = 1 - beta1 ** state['step']
                    else:
                        bias_correction1 = 1
                    if use_exp_avg_sq_norm and exp_avg_sq_norm_type != 'infinite_l':
                        bias_correction2 = 1 - beta2 ** state['step']
                    else:
                        bias_correction2 = 1
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1

                lr = group['lr']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # MAKE STEP

                if decouple_wd:
                    data_old = p.data.clone()

                if exp_avg_norm_type == 'nesterov':
                    numerator = exp_avg.mul(beta1).add(1 - beta1_dump, grad)

                if denom is not None:
                    # print('denom not none')
                    delta = numerator.div(denom)
                    p.data.addcdiv_(-step_size, numerator, denom)
                else:
                    # print(numerator, exp_avg)
                    delta = numerator
                    p.data.add_(-step_size, numerator)

                if decouple_wd:
                    # group['weight_decay'] is externally decayed
                    p.data = p.data.add(-group['weight_decay'], data_old)

                if use_adadelta_lr:
                    acc_delta.mul_(beta2).addcmul_(1 - beta2, delta, delta)

        return loss
