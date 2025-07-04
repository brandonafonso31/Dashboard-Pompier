import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from networks import *
from ReplayBuffers import *
from IntrinsicCuriosityModule import *
from collective_functions import get_potential_actions

import schedulefree

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_type,
                 layer_size,
                 num_layers,
                 use_batchnorm,
                 n_steps,
                 batch_size,
                 buffer_size,
                 lr,
                 lr_dec,
                 tau,
                 gamma,
                 munchausen,
                 curiosity,
                 curiosity_size,
                 per,
                 rdm,
                 entropy_tau,
                 entropy_tau_coeff,
                 lo,
                 alpha ,
                 N,
                 entropy_coeff,
                 update_every,
                 max_train_steps,
                 device,
                 seed):

        self.state_size = state_size
        self.action_size = action_size
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.seed = seed
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
        self.batch_size = batch_size
        self.Q_updates = 1 # to match with decay update
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.N = N
        self.lr = lr
        self.lr_dec = lr_dec
        self.per = per
        self.rdm = rdm
        # munchausen params
        self.munchausen = munchausen
        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = .1
        self.entropy_tau = entropy_tau
        self.entropy_tau_coeff = entropy_tau_coeff
        self.lo = lo
        self.alpha = alpha   
        self.max_train_steps = max_train_steps # 80k for 10k resp, for lr decay
        self.decay_update =  100 # Q updates % decay update => lr decay
        print("lr decay:", self.lr_dec, "decay_update:", self.decay_update, "PER", self.per)
        self.grad_clip = 1 #1, 10 ?

	    # Q-Network
        if self.layer_type == "noisy":
            self.qnetwork_local = Dueling_QNetwork(state_size, action_size,layer_size, n_steps, seed, num_layers, layer_type, use_batchnorm).to(device)
            self.qnetwork_target = Dueling_QNetwork(state_size, action_size,layer_size, n_steps, seed, num_layers, layer_type, use_batchnorm).to(device)
        else:
            self.qnetwork_local = Dueling_QNetwork(state_size, action_size,layer_size, n_steps, seed, num_layers, layer_type, use_batchnorm).to(device)
            self.qnetwork_target = Dueling_QNetwork(state_size, action_size,layer_size, n_steps, seed, num_layers, layer_type, use_batchnorm).to(device)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        if self.lr_dec == 0:
            self.optimizer = schedulefree.AdamWScheduleFree(self.qnetwork_local.parameters(), lr=lr)
            print("Schedule Free Optimizer")
            # self.optimizer.train()

        print(self.qnetwork_local)
        
        # Replay memory Standard (random simple)
        if self.per == 0:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps, rdm)
        # Replay memory PER
        elif self.per == 1:
            self.memory = PrioritizedReplay(buffer_size, batch_size, seed, gamma, n_steps)
        # Replay memory PER Sum Tree
        elif self.per == 2:
            self.memory = N_Steps_Prioritized_ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps)
            
        # Curiosity
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)
            
    def step(self, state, action, reward, next_state, done):
        
        # state = torch.flatten(state)
        # next_state = torch.flatten(next_state)  

        state = torch.from_numpy(state.flatten()).float()
        next_state = torch.from_numpy(next_state.flatten()).float()
        
        # Save experience in replay memory

        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        if (self.t_step) % self.update_every == 0:

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample() 

                if self.per == 0:
                    loss, icm_loss = self.learn(experiences)
                else:
                    loss = self.learn_per(experiences)
                self.Q_updates += 1  
                # print("Q_updates - learn exp", self.Q_updates, flush=True)
            
    def act(self, state, all_ff_waiting, eps=0., eval=False):

        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        
        if np.random.uniform() > eps:
            # state = state.view(1, self.state_size).to(self.device)
            
            state = torch.from_numpy(state.flatten()).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                q = self.qnetwork_local(state)                
            self.qnetwork_local.train()
            
            q_list = q.cpu().numpy().flatten().tolist()  
            action = filter_q_values(q_list, potential_actions)

        else:
            action = random.choice(potential_actions)
        # if all_ff_waiting:
        #     print(potential_actions)
        #     print(potential_skills)
        #     print(action)

        skill_lvl = potential_skills[potential_actions.index(action)]
        # if all_ff_waiting:
        #     print("agent action", action, "skill_lvl", skill_lvl)

        return action, skill_lvl
    

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    def learn(self, experiences):
        
        icm_loss = 0

        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # calculate curiosity
        if self.curiosity != 0:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"
            if self.curiosity == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()
            icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma**self.n_steps * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) 
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        
        if self.lr_dec != 0:
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        if (self.Q_updates % self.decay_update == 0):

            print("update lr decay")

            if self.lr_dec == 0:
                self.lr_decay_0()
            elif self.lr_dec == 1:
                self.lr_decay_1() 
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
                
        return loss.detach().cpu().numpy(),  icm_loss         
    
    def learn_per(self, experiences):

            self.optimizer.zero_grad()
            states, actions, rewards, next_states, dones, idx, weights = experiences
        
            states = torch.from_numpy(states).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.from_numpy(weights).float().to(self.device)
        
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (self.gamma**self.n_steps * Q_targets_next * (1 - dones))
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(0, actions) # was 1
            # Compute loss
            td_error =  Q_targets - Q_expected
            loss = (td_error.pow(2)*weights).mean().to(self.device)
            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(),1)

            if self.lr_dec != 0:
                self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target)
            
            if (self.Q_updates % self.decay_update == 0):
    
                if self.lr_dec == 0:
                    self.lr_decay_0()
                elif self.lr_dec == 1:
                    self.lr_decay_1() 
                elif self.lr_dec == 2:
                    self.lr_decay_2()
                elif self.lr_dec == 3:
                    self.lr_decay_3()
            # update per priorities
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

            return loss.detach().cpu().numpy()    

    def lr_decay_0(self):
        for p in self.optimizer.param_groups:
            lr_now = p['lr']
        print("step", self.t_step, "current lr :", lr_now)
        
    def lr_decay_1(self):
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self):
        if (self.t_step % 5000 == 0):
            self.lr = self.lr / 2
            for p in self.optimizer.param_groups:
                p['lr'] = self.lr    
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self):
        self.lr = self.lr / 2
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr      
        print("step", self.t_step, "current lr :", self.lr)


class FQF_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 noisy,
                 layer_size,
                 n_steps,
                 batch_size,
                 buffer_size,
                 lr,
                 lr_dec,
                 tau,
                 gamma,
                 munchausen,
                 curiosity,
                 curiosity_size,
                 per,
                 rdm,
                 entropy_tau,
                 entropy_tau_coeff,
                 lo,
                 alpha ,
                 N,
                 entropy_coeff,
                 update_every,
                 device,
                 seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
        self.batch_size = batch_size
        self.Q_updates = 1
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.N = N
        self.lr = lr
        self.lr_dec = lr_dec
        self.per = per
        self.rdm = rdm
        # munchausen params
        self.munchausen = munchausen
        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = .1
        self.entropy_tau = entropy_tau
        self.entropy_tau_coeff = entropy_tau_coeff
        self.lo = lo
        self.alpha = alpha   
        self.noisy = noisy
        self.max_train_steps = 80000 # for 10k resp
        self.decay_update =  100
        print("lr decay", bool(self.lr_dec), "decay_update", self.decay_update, "PER", self.per)
        self.grad_clip = 1 #10 ?

        # FQF-Network
        self.qnetwork_local = QVN(state_size, action_size,layer_size, n_steps, device, seed, noisy, N).to(device)
        self.qnetwork_target = QVN(state_size, action_size,layer_size, n_steps,device, seed, noisy, N).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        print(self.qnetwork_local)

        self.FPN = FPN(layer_size, seed, N, device).to(device)
        print(self.FPN)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=lr*0.000001, alpha=0.95, eps=0.00001)
        
        # Replay memory Standard (random simple)
        if self.per == 0:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps, rdm)
        # Replay memory PER
        elif self.per == 1:
            self.memory = PrioritizedReplay(buffer_size, batch_size, seed, gamma, n_steps)
        # Replay memory PER Sum Tree
        elif self.per == 2:
            self.memory = N_Steps_Prioritized_ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps)
            
        # Curiosity
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)


    def step(self, state, action, reward, next_state, done):
        
        state = torch.flatten(state.cpu())
        next_state = torch.flatten(next_state.cpu())  
        
        # Save experience in replay memory

        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        if (self.t_step) % self.update_every == 0:

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample() 

                if self.per == 0:
                    loss, entropy, icm_loss = self.learn(experiences)
                else:
                    loss, entropy = self.learn_per(experiences)
                self.Q_updates += 1

                
    def act(self, state, all_ff_waiting, eps=0., eval=False):

        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
              
        if np.random.uniform() > eps:
            # state = torch.from_numpy(state).float().to(self.device)
            state = state.view(1, self.state_size)
            self.qnetwork_local.eval()
            with torch.no_grad():
                embedding = self.qnetwork_local.forward(state)
                taus, taus_, _ = self.FPN(embedding)
                F_Z = self.qnetwork_local.get_quantiles(state, taus_, embedding)
                action_values = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z).sum(1)                   
            self.qnetwork_local.train()
            
            q_list = action_values.cpu().numpy().flatten().tolist()  
            action = filter_q_values(q_list, potential_actions)

        else:
            action = random.choice(potential_actions)

        skill_lvl = potential_skills[potential_actions.index(action)]

        return action, int(skill_lvl)
        
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1) 
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())

        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected.shape == (self.batch_size, self.N, 1)
        
        # calc fractional loss 
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss
        
        # calculate icm loss
        if self.curiosity != 0:
            
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"
            if self.curiosity == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()        
        icm_loss = 0
        if self.curiosity != 0:
            icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
        
        # Calculate Q_targets without munchausen 
        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.batch_size, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.gamma**self.n_steps * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
  
        
        # Calculate Q_targets with munchausen
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.gamma**self.n_steps * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.batch_size, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.batch_size, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target
            
        
        # Compute standard loss
        # loss = F.mse_loss(Q_expected, Q_targets) 
        
        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        
        # assert td_error.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)

        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1) 
        loss = loss.mean()

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()
        

        # Minimize standard loss
        # loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        if (self.Q_updates % self.decay_update == 0) and self.lr_dec:
            if self.lr_dec == 1:
                self.lr_decay() 
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
        return loss.detach().cpu().numpy(),  entropy.mean().detach().cpu().numpy(), icm_loss

    def learn_per(self, experiences):

        states, actions, rewards, next_states, dones, idx, weights = experiences
        
        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())
        
        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected.shape == (self.batch_size, self.N, 1)
        # calc fractional loss
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus, weights)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss

        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.batch_size, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.gamma**self.n_steps * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.gamma**self.n_steps * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.batch_size, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.batch_size, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target


        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        # assert td_error.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights
        loss = loss.mean()
        

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),self.grad_clip)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if (self.Q_updates % self.decay_update == 0) and self.lr_dec:

            if self.lr_dec == 1:
                self.lr_decay() 
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
        
        # update priorities
        td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test 
        
        self.memory.update_priorities(idx, td_error.data.cpu().numpy())

                
        return loss.detach().cpu().numpy(), entropy.mean().detach().cpu().numpy()    

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def lr_decay(self):
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self):
        if (self.t_step % 5000 == 0):
            self.lr = self.lr / 2
            for p in self.optimizer.param_groups:
                p['lr'] = self.lr    
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self):
        self.lr = self.lr / 2
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr      
        print("step", self.t_step, "current lr :", self.lr)

    
def calc_fraction_loss(FZ_,FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    assert not gradients.requires_grad
    if weights != None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1)*weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss

def filter_q_values(q_list, potential_actions):
    if potential_actions != [99]:

        dic_q = {k:q_list[k] for k in potential_actions} # dic potential action : q value  

        return max(dic_q, key=dic_q.get) # action with the highest q value
    else:
        return 99
    
# def get_potential_actions(state, all_ff_waiting):

#     # 1st row: rl infos
#     # 2nd row: idx role

#     potential_actions = [99]
#     skill_lvl = 0
#     potential_skills = [0]
#     # state = state.cpu().numpy()
#     col_index = np.argmax(state[1, :] == 1) # role actuel
#     column_values = state[2:, col_index] # ff available for a given role
#     selection = column_values[column_values > 0] # ff having the skill
#     if selection.size > 0: # any ff ?
#         if not all_ff_waiting:
#             cond_met = np.where( (column_values > 0) & (state[2:, -1] == 0) )[0] # ff having any skill lvl > 0
#             potential_skills = column_values[(column_values > 0) & (state[2:, -1] == 0)].tolist()
#         else: # all ff waiting
#             skill_lvl = np.min(selection)
#             cond_met = np.where( (column_values >= skill_lvl) & (state[2:, -1] == -1) )[0] # ff having the best skill lvl
#             cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
#             potential_skills = [skill_lvl]
            
#         if cond_met.size > 0:
#             potential_actions = cond_met.tolist()
        
#     return potential_actions, potential_skills

# def get_potential_actions(state, all_ff_waiting):

#     # 1st row: rl infos
#     # 2nd row: idx role

#     potential_actions = [99]
#     skill_lvl = 0
#     potential_skills = [0]
#     # state = state.cpu().numpy()
#     col_index = np.argmax(state[1, :] == 1) # role actuel
#     column_values = state[2:, col_index] # ff available for a given role
#     selection = column_values[column_values > 0] # ff having the skill
#     if selection.size > 0: # any ff ?
#         if not all_ff_waiting: # standard case
#             cond_met = np.where( (column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1)) )[0] # ff having any skill lvl > 0
#             potential_skills = column_values[(column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1))].tolist()
#         else: # all ff waiting
#             skill_lvl = np.min(selection)
#             cond_met = np.where( (column_values >= skill_lvl) & (state[2:, -2] == 1) )[0] # ff having the best skill lvl
#             cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
#             potential_skills = [skill_lvl]
            
#         if cond_met.size > 0:
#             potential_actions = cond_met.tolist()
#         else:
#             potential_skills = [0]
        
#     return potential_actions, potential_skills

# def get_potential_actions(state, all_ff_waiting):

#     # 1st row: rl infos
#     # 2nd row: idx role

#     potential_actions = [99]
#     skill_lvl = 0
#     potential_skills = [0]
#     # state = state.cpu().numpy()
#     col_index = np.argmax(state[1, :] == 1) # role actuel
#     column_values = state[2:, col_index] # ff available for a given role
#     selection = column_values[column_values > 0] # ff having the skill
#     if selection.size > 0: # any ff ?
#         if not all_ff_waiting: # standard case
#             cond_met = np.where( (column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1)) )[0] # ff having any skill lvl > 0
#             # print("cond_met.size", cond_met.size)
#             potential_skills = column_values[(column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1))].tolist()
#             # print("potential_skills", potential_skills)
#             # subset = state[1:16, np.r_[0:5, -3:0]]
#             # print("Shape:", subset.shape)
#             # print(subset)
#         else: # all ff waiting
#             skill_lvl = np.min(selection)
#             cond_met = np.where( (column_values >= skill_lvl) & (state[2:, -2] == 1) )[0] # ff having the best skill lvl
#             cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
#             potential_skills = [skill_lvl]
#             # print("skill_lvl1", skill_lvl)
            
#         if cond_met.size > 0:
#             potential_actions = cond_met.tolist()
#         else:
#             potential_skills = [0]

#     # print("skill_lvl2", skill_lvl, potential_skills)
        
#     return potential_actions, potential_skills

