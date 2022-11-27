# from keras.layers import Dense, Activation
# from keras.models import Sequential, load_model
# from keras.optimizers import Adam
import numpy as np
import pandas as pd

class forex:
    def __init__(self):
        self.dataSet = pd.read_csv('Indicator_train_date_scale.csv')
        self.window = 40
        self.data_step = 0 # full data basis
        self.full_data = len(self.dataSet)- self.window
        self.done = False


        self.reward = 0
        self.Lot = 1000   #lot=100_000    mini=10_000    micro= 1_000
        self.commission=0

        self.opened_trade = 0
        self.previous_trade_position=0
        self.previous_action = False

    def save_previous_action(self):
        self.previous_trade_position= self.dataSet['Raw_Close'][self.data_step+self.window-1]


    def step(self, action):
        done=False
        reward=0
        ##################  calculate profit ################
        ######### open down#############
        if action[0]==1:
            if self.opened_trade == 1:
                reward=-10
            else:
                reward=1
                self.opened_trade=0
                #save position
                self.save_previous_action()
                self.previous_action = False
        ######### open up ###########
        elif action[1]==1:
            if self.opened_trade == 1:
                reward=-10
            else:
                reward=1
                self.opened_trade=1
                #save position
                self.save_previous_action()
                self.previous_action = True
        ##### null ##############
        elif action[2]==1:
            reward=0
        ##### close down ###################
        elif action[3]==1:
            if self.opened_trade == 1:
                reward = self.profit_close(self.previous_trade_position, self.dataSet['Raw_Close'][self.data_step+self.window-1], self.previous_action)
                self.opened_trade=0
            else:
                reward=-10

        # ######### close up ###########
        elif action[4]==1:
            if self.opened_trade == 1:
                reward = self.profit_close(self.previous_trade_position, self.dataSet['Raw_Close'][self.data_step+self.window-1], self.previous_action)
                self.opened_trade=0
            else:
                reward=-10

        ##################  calculate Next observation ################
        observation=self.prep_data(self.data_step)

        ##################  calculate counter ################
        if self.dataSet['Minute'][self.data_step]==1 and self.dataSet['Hour'][self.data_step]==1:
            done = True
            self.data_step += 1
            if self.opened_trade==1:
                self.opened_trade=0
                self.previous_trade_position=0
                reward=-20
            else:
                self.opened_trade=0
                self.previous_trade_position=0
        elif self.data_step == self.full_data - self.window:
            done = True
            self.data_step=0
            if self.opened_trade==1:
                self.opened_trade=0
                self.previous_trade_position=0
                reward=-20
            else:
                self.opened_trade=0
                self.previous_trade_position=0
        else:
            self.data_step += 1
            done = False

        return observation, reward, done,

    def prep_data(self, data_step):

        c=np.full((40), self.opened_trade)
        a=np.stack((
            self.dataSet["Hour"][data_step:data_step+self.window],
            self.dataSet["Weekday"][data_step:data_step+self.window],
            self.dataSet["Month"][data_step:data_step+self.window],

            self.dataSet["Close"][data_step:data_step+self.window],
            self.dataSet["Open"][data_step:data_step+self.window],
            self.dataSet["Low"][data_step:data_step+self.window],
            self.dataSet["High"][data_step:data_step+self.window],
            self.dataSet["Volume"][data_step:data_step+self.window],

            self.dataSet["EMA_9"][data_step:data_step+self.window],
            self.dataSet["EMA_21"][data_step:data_step+self.window],
            self.dataSet["EMA_200"][data_step:data_step+self.window],

            # self.dataSet["RSI_5"][data_step:data_step+self.window],
            self.dataSet["RSI_14"][data_step:data_step+self.window],

            # self.dataSet["bb_bbm"][data_step:data_step+self.window],
            self.dataSet["bb_bbh"][data_step:data_step+self.window],
            self.dataSet["bb_bbl"][data_step:data_step+self.window],
            # self.dataSet["bb_bbhi"][data_step:data_step+self.window],
            # self.dataSet["bb_bbli"][data_step:data_step+self.window],
            self.dataSet["bb_bbw"][data_step:data_step+self.window],
            # self.dataSet["bb_bbp"][data_step:data_step+self.window],

            self.dataSet["williams_r"][data_step:data_step+self.window],
            c
            ))

        b = np.reshape(a, (17, 40, 1))
        return b


    def episode_reset(self):
        aa=self.prep_data(self.data_step)
        self.opened_trade = 0
        self.previous_trade_position=0
        self.previous_action = False
        return aa

    def reset_data(self):
        self.data_step=0

    def profit_close(self, open, close, down_up):
        if down_up: # means predicted up
            profits=((close-open)*self.Lot)-self.commission
            self.opened_trade = 0
        else:  # predicted down
            profits=((open-close)*self.Lot)-self.commission
            self.opened_trade = 0
        return profits

    def profit_open(self):
        self.opened_trade = 1
        profits = 1
        return profits
