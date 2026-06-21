# import tensorflowa as tf
from keras.layers import Dense, Activation, Conv2D, Flatten, AveragePooling2D, Input,Concatenate
from keras.models import load_model

from keras.optimizers import Adam
import numpy as np
from keras.models import Model
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, 17,40,1))
        self.new_state_memory = np.zeros((self.mem_size, 17,40,1))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=dtype)
        self.action_memory2 = np.zeros((self.mem_size,1), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action[0]
            self.action_memory2[index] = action[1]
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions1 = self.action_memory[batch]
        actions2 = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions1, actions2, rewards, states_, terminal

def build_dqn(lr, window=40):

    input_1 = Input(shape=(17,40,1))
    x = Conv2D(8, (1,11), padding="same",activation='relu')(input_1)
    x = Conv2D(8, (5,1), padding="same",activation='relu')(x)
    x = Conv2D(8, (1,11), padding="valid")(x)
    x = Conv2D(8, (5,1), padding="valid")(x)

    x = Conv2D(16, (1,9), padding="same",activation='relu')(x)
    x = Conv2D(16, (9,1), padding="same",activation='relu')(x)
    x = Conv2D(16, (1,9), padding="valid")(x)
    x = Conv2D(16, (5,1), padding="valid")(x)

    x = Conv2D(32, (1,5), padding="same",activation='relu')(x)
    x = Conv2D(32, (5,1), padding="same",activation='relu')(x)
    x = Conv2D(32, (5,1), padding="same",activation='relu')(x)
    x = Conv2D(32, (1,5), padding="valid")(x)
    x = Conv2D(32, (5,1), padding="valid")(x)

    x=AveragePooling2D(pool_size=(1, 2),strides=(1, 2))(x)

    x = Conv2D(64, (1,3), padding="same",activation='relu')(x)
    x = Conv2D(64, (1,3), padding="same",activation='relu')(x)
    x = Conv2D(64, (3,3), padding="valid")(x)

    x = Conv2D(128, (5,1), padding="same",activation='relu')(x)
    x = Conv2D(128, (5,1), padding="same",activation='relu')(x)
    x1 = Conv2D(128, (3,3), padding="valid")(x)

    x2 = Conv2D(256, (1,1), padding="same",activation='relu')(x1)
    x2 = Conv2D(256, (1,1), padding="same",activation='relu')(x1)

    x3 = Conv2D(256, (1,1), padding="same",activation='relu')(x1)
    x3 = Conv2D(256, (1,1), padding="same",activation='relu')(x1)

    f1 = Flatten()(x2)
    f2 = Flatten()(x3)

    out1 = Dense (5, activation='softmax')(f1)
    out2 = Dense (1)(f2)

    mymodel = Model(inputs=input_1, outputs=[out1,out2])
    mymodel.compile(optimizer=Adam(lr=lr), loss={'Action':'categorical_crossentropy',
                                                'n_LOT':'mse'}, metrics=['accuracy'])

    return mymodel



class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                    input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                    mem_size=900000, fname='ddqn_model.h5', replace_target=50):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                    discrete=False)

        self.q_eval = build_dqn(alpha)
        self.q_target = build_dqn(alpha)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state=state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            x = np.zeros((1, 5))
            J = np.random.choice(5, 1)
            x[np.arange(1), J] = 1
            action = x[0]
        else:
            actions = self.q_eval.predict(state, verbose=0)
            x = np.zeros((1, 5))
            J = np.argmax(actions)
            x[np.arange(1), J] = 1
            action = x[0]

        return action1, action2

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action1, action2, reward, new_state, done = \
                                            self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices1 = np.dot(action1, action_values)
            action_indices = action_indices1.astype(int)


            q_next = self.q_target.predict(new_state, verbose=0)
            q_eval = self.q_eval.predict(new_state, verbose=0)
            q_pred = self.q_eval.predict(state, verbose=0)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            # print("=========>", action_indices)
            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()
