import torch as t
from importlib import reload
import torchvision.transforms.functional as Fv
import time
import Agent
reload(Agent)
import numpy as np
import gymnasium as gym

import plotting_fun
reload(plotting_fun)
def resize_observation(obs,shape)                                                                                                                                                                                                                                                            :

    obs.resize(1,1,shape[0],shape[1])

    obs = t.tensor(obs, dtype=t.float32)
    obs = Fv.resize(obs,(84, 84))

    obs=obs.numpy()
    return obs[0][0]
def main(n_games:int=100,
         attari_title:str="Asterix-v4",
         gamma:float=0.99,
         epsilon:float=1e-5,
         batch_size:int=64,
         eps_end:float=0.1,
         lr:float=1e-5,
         max_mem_size:int=400000,
         ep_dec:float=1e-4,
         steps:int=2000,
         framesStack:int=4,
         frameSkip:int=1,
         plotname:str="training_result",
         propability_scale:float=1.0)->None:
    """
    This function initialize the agent that we will train and then starts the training loop.

    Args:
         n_games (int): the number of games the agent will experience.
         attari_title (str): the name of the game our agent will be trained on.
         gamma (float) (0,1): bellman equation parameter, higher value means future rewards are important
         epsilon (float): the initial value of the possibily that our agent will choose a random action (exploration)
         batch_size (int): size of the batches that we will train our model with.
         eps_end (float): the lower limit that epsilon can have as value
         lr (float):  learning rate of our model
         max_mem_size (int):  the count of discrete experiences we will store in memory
         ep_dec (float): the value we reduce the epsilon by after each step of the enviroment
         steps (int): its the number of steps we update our target model with
         framesStack (int): the count of sequential frames that are stacked together in order to give the agent the perception of movement
         frameSkip (int): the amount of frames the agent will not chose a action.
         plotname (str): name of the png file that will depict the training size
         propability_scale (float): [0,1]
    """

    env=gym.make(attari_title,obs_type="grayscale")
    n_actions = env.action_space.n
    agent=Agent.Agent(gamma=gamma,epsilon=epsilon,batch_size=batch_size,
                n_actions=n_actions,eps_end=eps_end,height=84,width=84,depth=framesStack,lr=lr,max_mem_size=max_mem_size,
                ep_dec=ep_dec,steps=steps,propability_scale=propability_scale)




    max=0
    frame=0
    trainingTimerStart=time.time()
    scores,eps_history,time_history,avgScore,avgTime=[],[],[],[],[]
    for i in range(n_games):
        timerStart=time.time()
        score=0
        done=False
        truncated=False
        observation=env.reset()[0]
        observation = observation
        observation=resize_observation(observation,observation.shape)


        neuralNetTime=0
        stackedObs=[]
        stackedNewObs=[]

        actionSkip=0
        countingNOOPS=0
        index=0
        maxreward=0
        while not (done or truncated):
            a=time.time()
            if(frameSkip==0 or index%frameSkip==0):
                action=agent.choose_action(np.array(stackedObs))
                actionSkip=action
            else:
                 action=actionSkip

            if(action==0):
                countingNOOPS+=1
            else:
                countingNOOPS=0

            if(countingNOOPS==30):
                action=np.random.choice(n_actions)
                countingNOOPS=0
                actionSkip=action
            neuralNetTime=time.time()-a
            observationNext, reward, done,_,info =env.step(action)
            observationNext=resize_observation(observationNext,observationNext.shape)
            if reward>maxreward:
                maxreward=reward
            if maxreward!=0:
                reward/=maxreward
            score+=reward
            if(max<=score):
                max=score
                maxIndex=i



            lives=info['lives']
            a=time.time()
            stackedObs.append(observation)
            stackedNewObs.append(observationNext)
            agent.store_transition(observation,action,reward,observationNext,done)
            agent.learn()
            frame+=1
            if(frame%10000==0):
                print(f"frame {frame}")

            if not index<framesStack-1:
                #κρατάω μια στοίβα απο τις πιο πρόσφατες καταστάσεις για να τροφοδοτήσω το δίκτυο
                stackedObs.pop(0)
                stackedNewObs.pop(0)
            observation=observationNext
            neuralNetTime=time.time()-a
            index+=1

        time_history.append(time.time()-timerStart-neuralNetTime)
        scores.append(score)
        eps_history.append(agent.epsilon)

        anv_score=np.mean(scores[-100:])
        anv_time=np.mean(time_history[-100:])
        avgScore.append(anv_score)
        avgTime.append(anv_time)
        print(f"episode {i} score {score}, average score of the last 100 games {anv_score}, epsilon {agent.epsilon}")
    env.close()
    plotting_fun.plotTrainingInfo(scores,avgScore,time_history,avgTime,"Training Process Of Agent.",plotname)
    print(f"Training took {(time.time()-trainingTimerStart)/3600.0} hours")
    print(f"Max score :{max} achieved at episode {maxIndex}")
    print(f"Average episode score {sum(scores)/n_games}")
    print(f"standard deviation of rewards {np.std(scores)}")
    print(f"Average survival time of episodes{sum(time_history)/n_games}")

if __name__=="__name__":
    main()