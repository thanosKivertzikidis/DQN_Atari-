import DQN_architecture as my_dqn
import numpy as np
import torch as t
#WHEN USING PRIORITY YOU MUST NOT USE FRAMESTACK
class Agent():

    def __init__(self,steps:int,gamma:float,epsilon:float,lr:float,height:int,width:int,depth:int,batch_size:int,n_actions:int,max_mem_size:int,eps_end:float,ep_dec:float,propability_scale:float)->None:
        """
        Args:
            steps: the number of steps we have to do on Q eval before we can update Q target
            gamma:the γ parameter in bellman equation, higher gamma means bigger trust in future rewards
            epsilon: the starting point of the possibility our agent will take a random act.
            lr: learning rate of DQN
            batch_size: the number of experience that we feed our DQN
            n_actions: number of possible actions
            max_mem_size: experience replay size
            eps_end: the lower limit for epsilon value
            ep_dec: the value that is substracted by epsilon at each step
            propability_scale: A number between [0,1]. This is used in the normalization of the priorities, if it is 0 then we dont use priority.
        """
        self.frameStackLimit=depth
        self.stepLimit=steps
        self.step=0
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=ep_dec
        self.lr=lr
        self.action_space=np.linspace(0,n_actions-1,dtype=np.int32)
        self.mem_size=max_mem_size
        self.batch_size=batch_size


        self.mem_cntr=0
        self.Q_eval=my_dqn.DQN(lr=self.lr,n_actions=n_actions,height=height,width=width,depth=depth)
        self.Q_target=my_dqn.DQN(lr=self.lr,n_actions=n_actions,height=height,width=width,depth=depth)
        self.state_memory=t.tensor(np.zeros((self.mem_size,height,width),dtype=np.float32)).to(self.Q_eval.device)
        self.new_state_memory=np.zeros((self.mem_size,height,width),dtype=np.float32)
        self.action_memory=np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory=np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size,dtype=np.bool_)


    def store_transition(self,state:np.array,action:int,reward:int, stateΝew:np.array,done:np.bool_)->None:
        """
        This method stores a new experience in the agents memory
        Args:
            state: current observation
            action: current action
            reward: the reward of the current action
            stateNew: the next observation (result of the action)
            done: True/False if the new observation is/isn't final
        """
        index=self.mem_cntr%self.mem_size
        self.state_memory[index]=t.tensor(state).to(self.Q_eval.device)
        self.new_state_memory[index]=stateΝew
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.terminal_memory[index]=done
        self.mem_cntr+=1
    def choose_action(self,observation:np.array)->int:
        """
        This method chooses the next action for the agent.
        We produce a random number, if that number is higher that epsilon and we have stacked enough frames to pass the input to the DQN
        we exploit the knowlegde.
        If the number is lower than epsilon then we pick a random action and we explore our enviroment. 
        Args:
            observation (np.array): Current observation based on which we will decide the next action
        Returns:
            int: the number of the action we chose
            
        """
        if np.random.random()>self.epsilon and observation.shape[0]==self.frameStackLimit:
            state = t.tensor(np.array([observation]), dtype=t.float32).to(self.Q_eval.device)
            actions=self.Q_eval.forward(state)
            action=t.argmax(actions).item()
            del state
        else:
            action=np.random.choice(self.action_space)
       
        return action
    def getFrameStacks(self,batch,limit)->tuple[np.array,np.array]:
            """
            We take a self.frameStackLimit sequence of records in the agent's memory and stack them together.
            We do this by taking the self.frameStackLimit-1 previous experiences from a sampled experience.
           
              
            Args:
                batch: the indexes of memories that were randomly sampled
                limit: the most recent observation is the limit that defines the new and the old experiences when the memory has been overwritten.
            Returns:
                a np.array of stacked current-observations
                a np.array of stacked  next-observations
             Note:
                if we are above the limit but closer to it than self.frameStackLimit then we take the self.frameStackLimit number after the limit  
        
            
            """
        
            resultStackState = []
            resultStackNextState = []

            for index in batch:
                if index<limit:
                    if index < self.frameStackLimit:
                        result_indices = np.arange(0, self.frameStackLimit)
                    else:
                        result_indices = np.arange(index - self.frameStackLimit + 1, index + 1)
                else:
                    if index-limit<self.frameStackLimit:
                        result_indices=np.arange(limit+1,limit+1+self.frameStackLimit)
                    else:
                        result_indices = np.arange(index - self.frameStackLimit + 1, index + 1)
                resultStackState.append(self.state_memory[result_indices])    
                resultStackNextState.append(self.new_state_memory[result_indices])

            resultStackState=t.stack(resultStackState).to(self.Q_eval.device)
            tensors = [t.tensor(x) for x in resultStackNextState]
            resultStackNextState = t.stack(tensors).to(self.Q_eval.device)
            
            
            return resultStackState, resultStackNextState

    def learn(self)->None:
        """
            The method responsible for enabling the Agent to learn from its experiences.

            This method performs experience replay and updates the Q-network based on priority sampling. 
            If the memory buffer has not accumulated enough samples (less than batch_size), the function returns without learning.
            Otherwise, it selects a batch of experiences from memory based on priority, computes the Q-values, and applies the Bellman equation to update the Q-network.

            The learning process proceeds as follows:
            1. Syncs the target network with the evaluation network after a fixed number of steps (stepLimit).
            2. Chooses a batch of experiences from the replay memory using prioritized experience sampling.
            3. Calculates Q-values from the evaluation network (Q_eval) and the target network (Q_target).
            4. Uses the Bellman equation to compute the target Q-values, incorporating rewards and future state Q-values.
            5. Updates the network by calculating the loss between the predicted and target Q-values, using importance sampling weights to scale the loss.
            6. Adjusts the priorities in the replay memory based on the TD-error.
            7. Performs a gradient descent step to minimize the loss and update the evaluation network's parameters.
            8. Decays the exploration rate (`epsilon`) until it reaches a minimum threshold (`eps_min`).

            Notes:
            ------
            - If the agent hasn't collected enough experiences (i.e., `mem_cntr < batch_size`), learning is skipped.
            - The target network (`Q_target`) is synchronized with the evaluation network (`Q_eval`) every `stepLimit` steps.
            - The method leverages frame stacking (`getFrameStacks`) when required by the environment.
            - The exploration-exploitation balance is managed using an epsilon-greedy policy with a decaying epsilon value.
            
            Performance optimizations:
            - Gradients are reset using `zero_grad()` before backpropagation.
            - Unnecessary variables are deleted at the end of the function to manage memory.
            - The method ensures the correct handling of terminal states by setting their Q-targets to -1.
        """
        if self.mem_cntr-self.frameStackLimit+1 < self.batch_size:
            return
        if(self.stepLimit==0 or self.step%self.stepLimit==0):

            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            self.step=0
        else:
             self.step+=1

        self.Q_eval.optimizer.zero_grad()
        self.Q_target.optimizer.zero_grad()

        if(self.mem_cntr<self.mem_size):
            memoryLimit=self.mem_cntr
        else:
            memoryLimit=self.mem_size
        
        batch = np.random.choice(memoryLimit-self.frameStackLimit+1,self.batch_size, replace=False)
        action_batch = self.action_memory[batch]
        reward_batch = t.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = t.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.frameStackLimit==1:
            state_batch = t.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = t.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        else:
            state_batch,new_state_batch=self.getFrameStacks(batch,self.mem_cntr%self.mem_size-1)
            
        reward_batch[terminal_batch]=-1
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)

        q_target = reward_batch + self.gamma * t.max(q_next, dim=1)[0]
        q_target[terminal_batch]=-1
        self.Q_eval.train
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon=self.eps_min
        del state_batch
        del new_state_batch
        del action_batch
        del q_eval
        del q_next
        del loss
        del batch
        del batch_index
        del q_target
        del reward_batch
        del terminal_batch
        del memoryLimit