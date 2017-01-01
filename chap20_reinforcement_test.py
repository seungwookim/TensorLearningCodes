
import tensorflow as tf
import random

pile = [0, 0, 2, 0, 0, 0, 1, 1, 0, 0,
        0, 2, 2, 2, 0, 1, 1, 1, 3, 0,
        1, 1, 1, 1, 1, 1, 1, 3, 3, 3]

class EnvironmentModule :
    """
    Game Rule
    number 0 : empty
    number over 1 :groups
    gain puls score : when there is no other group on the top on index you pick
    gain minus score :when there is other group on the top on index you pick
    game end : when all index become 0
    """
    def updateState(self, index):
        global pile
        reward = self.getReward(pile, index)
        pile[index] = 0
        flag = self.getEndFlag(pile)
        print("action : {0}".format(index))
        print("state : {0}".format(pile))
        print("reward : {0}".format(reward))
        print("end flag : {0}".format(flag))
        return reward, flag

    def getEndFlag(self, pile):
        for val in pile:
            if val > 0 :
                return False
        return True


    def getReward(self, pile, index):
        if(pile[index] == 0) :
            return -1
        elif(index - 10 > 0 and (pile[index] == pile[index-10] or pile[index-10] == 0)):
            return 1
        else :
            return -1

    def getState(self):
        global pile
        return pile


class TensorModule :
    def train_reward(self):
        return ""

    def predict_action(self, state):
        return ""

class AgentModule :

    def train_reward(self, reward):
        print("train net using reward")

    def get_prediction(self, state):
        return random.randrange(0, 30)


class SimulThread :
    def run_thread(self):
        env_manager = EnvironmentModule()
        agent_manager = AgentModule()
        while(True):
            """
            Agent : set state on agent
            Agent : decide what action to do
            Env : update with action => state
            Reward : get reawrd from Env
            Train : train model with reward
            """
            state = env_manager.getState()
            action = agent_manager.get_prediction(state)
            reward, flag = env_manager.updateState(action)
            agent_manager.train_reward(reward)

            if(flag):
                break


SimulThread().run_thread()