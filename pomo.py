from agent import *
from collective_functions import *

#--------------------------------------------------------------------------------------#
#----------------------------------- TEST POMO ----------------------------------------#
#--------------------------------------------------------------------------------------#

class POMO_Agent(DQN_Agent):
    """Extension de DQN_Agent pour le multi-agent POMO"""
    
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
                 alpha,
                 N,
                 entropy_coeff,
                 update_every,
                 max_train_steps,
                 device,
                 seed,
                 num_agents=5):  # Nombre d'agents POMO
        
        super().__init__(
            state_size, action_size, layer_type, layer_size, num_layers, use_batchnorm,
            n_steps, batch_size, buffer_size, lr, lr_dec, tau, gamma, munchausen,
            curiosity, curiosity_size, per, rdm, entropy_tau, entropy_tau_coeff, lo,
            alpha, N, entropy_coeff, update_every, max_train_steps, device, seed
        )
        
        self.num_agents = num_agents
        self.current_agent = 0  # Index de l'agent actif
        
        # Initialiser plusieurs réseaux pour les différents agents
        if self.layer_type == "noisy":
            self.qnetworks_local = [Dueling_QNetwork(state_size, action_size, layer_size, n_steps, seed+i, 
                                                   num_layers, layer_type, use_batchnorm).to(device) 
                                  for i in range(num_agents)]
            self.qnetworks_target = [Dueling_QNetwork(state_size, action_size, layer_size, n_steps, seed+i, 
                                                    num_layers, layer_type, use_batchnorm).to(device) 
                                   for i in range(num_agents)]
        else:
            self.qnetworks_local = [Dueling_QNetwork(state_size, action_size, layer_size, n_steps, seed+i, 
                                                   num_layers, layer_type, use_batchnorm).to(device) 
                                  for i in range(num_agents)]
            self.qnetworks_target = [Dueling_QNetwork(state_size, action_size, layer_size, n_steps, seed+i, 
                                                    num_layers, layer_type, use_batchnorm).to(device) 
                                   for i in range(num_agents)]
            
        # Optimiseur pour tous les agents
        all_params = []
        for net in self.qnetworks_local:
            all_params += list(net.parameters())
            
        if self.lr_dec == 0:
            self.optimizer = schedulefree.AdamWScheduleFree(all_params, lr=lr)
        else:
            self.optimizer = optim.Adam(all_params, lr=lr)
        
        # Pour accès compatible avec le code parent
        self.qnetwork_local = self.qnetworks_local[0]
        self.qnetwork_target = self.qnetworks_target[0]
        
    def step(self, state, action, reward, next_state, done):
        """Override de la méthode step pour gérer plusieurs agents"""
        state = torch.from_numpy(state.flatten()).float()
        next_state = torch.from_numpy(next_state.flatten()).float()
        
        # Sauvegarde de l'expérience pour tous les agents
        for i in range(self.num_agents):
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
    
    def act(self, state, all_ff_waiting, eps=0., eval=False):
        """Override de la méthode act pour utiliser l'agent courant"""
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        
        if np.random.uniform() > eps:
            state = torch.from_numpy(state.flatten()).float().to(self.device)
            
            # Utiliser le réseau de l'agent courant
            self.qnetworks_local[self.current_agent].eval()
            with torch.no_grad():
                q = self.qnetworks_local[self.current_agent](state)
            self.qnetworks_local[self.current_agent].train()
            
            q_list = q.cpu().numpy().flatten().tolist()
            action = filter_q_values(q_list, potential_actions)
        else:
            action = random.choice(potential_actions)
        
        skill_lvl = potential_skills[potential_actions.index(action)]
        
        # Passer à l'agent suivant pour le prochain appel
        self.current_agent = (self.current_agent + 1) % self.num_agents
        
        return action, skill_lvl
    
    def learn(self, experiences):
        """Override de la méthode learn pour entraîner tous les agents"""
        icm_loss = 0
        
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Calcul de la curiosité (identique pour tous les agents)
        if self.curiosity != 0:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            if self.curiosity == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()
            icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
        
        total_loss = 0
        
        # Calcul de la loss pour chaque agent
        for agent_idx in range(self.num_agents):
            # Get max predicted Q values from target model
            Q_targets_next = self.qnetworks_target[agent_idx](next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma**self.n_steps * Q_targets_next * (1 - dones))
            
            # Get expected Q values from local model
            Q_expected = self.qnetworks_local[agent_idx](states).gather(1, actions)
            
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            total_loss += loss
        
        # Moyenne des losses
        total_loss = total_loss / self.num_agents
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.qnetworks_local[agent_idx].parameters(), 1)
        
        if self.lr_dec != 0:
            self.optimizer.step()
        
        # Mise à jour des réseaux cibles
        for agent_idx in range(self.num_agents):
            self.soft_update(self.qnetworks_local[agent_idx], self.qnetworks_target[agent_idx])
        
        # Gestion du decay du learning rate
        if (self.Q_updates % self.decay_update == 0):
            if self.lr_dec == 0:
                self.lr_decay_0()
            elif self.lr_dec == 1:
                self.lr_decay_1()
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
        
        return total_loss.detach().cpu().numpy(), icm_loss