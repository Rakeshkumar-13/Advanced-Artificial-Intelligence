# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

import world
import random
import utils
import config
import numpy as np
import mdptoolbox
from utils import Directions

class Tallon():
    grid_size = (config.worldBreadth, config.worldLength)
    remaining_side = (1 - config.directionProbability)/2
    # reward values for the bonus, pit, whitecell and meanies

    # These rewards are used to give the result of the tallon score.
    meanies_reward= -1.0 
    white_reward = -0.04       
    pit_reward = -1.0 
    bonus_reward= 1.0          

    action_n_s_e_w_probability=(remaining_side, remaining_side, config.directionProbability, 0.) #probability of moving in the intended direction

    def __init__(self, arena):
        # Make a copy of the world an attribute, so that Tallon can

        # query the state of the world

        self.gameWorld = arena

        # What moves are possible.

        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

     # The below line of codes are used to display the position og the characters in the game.   
    def makeMove(self):
        try:
            self.number_of_states = self.grid_size[0] * self.grid_size[1]
            self.number_of_actions = 4

            self.pits = self.gameWorld.getPitsLocation() 
            self.tallon_location = self.gameWorld.getTallonLocation()  
            self.meanies = self.gameWorld.getMeanieLocation()  
            self.bonuses = self.gameWorld.getBonusLocation()    
            
            

            P,R = self.fill_in_probs()

            mdptoolbox.util.check(P, R)
            value_iteration2 = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
            value_iteration2.run()
            print(' The Policy Iteration :\n', value_iteration2.policy)

            tallon_position = lambda x: np.ravel_multi_index(x, self.grid_size)
            tallon_position_in_world = int(tallon_position((self.tallon_location.y,self.tallon_location.x)))

            print("Tallon's position in grid:",tallon_position_in_world)

            if int(value_iteration2.policy[tallon_position_in_world]) == 0:
                print("In position",tallon_position_in_world," tallon is going to", value_iteration2.policy[tallon_position_in_world],"north direction")
                return Directions.NORTH
            if int(value_iteration2.policy[tallon_position_in_world]) == 1:
                print("In position",tallon_position_in_world," tallon is going to",value_iteration2.policy[tallon_position_in_world],"south direction")
                return Directions.SOUTH
            if int(value_iteration2.policy[tallon_position_in_world]) == 2:
                print("In position",tallon_position_in_world,"tallon is going to", value_iteration2.policy[tallon_position_in_world],"east direction")
                return Directions.EAST
            if int(value_iteration2.policy[tallon_position_in_world]) == 3:
                print("In position",tallon_position_in_world,"tallon is going to",value_iteration2.policy[tallon_position_in_world],"west direction")
                return Directions.WEST
        except Exception as e:
            print(" Tallon can't locate any bonus because tallon is far away from the bonus. So tallon will move randomly")
            random_movement = utils.pickRandomPose(self.tallon_location.y,self.tallon_location.x)
            if random_movement.x > self.tallon_location.x:
                return Directions.EAST
            if random_movement.x < self.tallon_location.x:
                return Directions.WEST
            if random_movement.y > self.tallon_location.y:
                return Directions.NORTH
            if random_movement.y < self.tallon_location.y:
                return Directions.SOUTH 

    def fill_in_probs(self):
        try:
            available_meanies=[] 
            available_bonus=[]                 
            available_pits=[] 

            P = np.zeros((self.number_of_actions, self.number_of_states, self.number_of_states))  
            R = np.zeros((self.number_of_states, self.number_of_actions))                   
                                                                       

            tallon_position = lambda x: np.ravel_multi_index(x, (self.grid_size))
            tallon_position_in_world = tallon_position((self.tallon_location.y,self.tallon_location.x))

            pit_pos = lambda x: np.ravel_multi_index(x,self.grid_size)

            meanie_pos = lambda x: np.ravel_multi_index(x,self.grid_size)

            bonus_pos = lambda x: np.ravel_multi_index(x,self.grid_size)

            
             #get the pit that is close to tallon

            for pit in range(len(self.pits)):
                available_pits.append(pit_pos((self.pits[pit].y,self.pits[pit].x)))
                c_pit = min(available_pits,key=lambda x:abs(x-tallon_position_in_world))
                c_pit = str(c_pit)
                c_pit = (int(c_pit[0]),int(c_pit[1])) if(len(c_pit)>1) else (0,int(c_pit))
                currentpit = c_pit
            print("closest pit to Tallon is: ",c_pit)

            #get the bonus that is close to tallon

            for bonus in range(len(self.bonuses)):
                available_bonus.append(bonus_pos((self.bonuses[bonus].y,self.bonuses[bonus].x)))
                c_bonus = min(available_bonus,key=lambda x:abs(x-tallon_position_in_world))
                c_bonus = str(c_bonus)
                c_bonus = (int(c_bonus[0]),int(c_bonus[1])) if (len(c_bonus)>1) else (0,int(c_bonus))
                currentbonus = c_bonus
            print("closest bonus to Tallon is: ",c_bonus)

            #get the bonus that is meanie to tallon

            for meanie in range(len(self.meanies)):
                available_meanies.append(meanie_pos((self.meanies[meanie].y,self.meanies[meanie].x)))
                c_meanie = min(available_meanies,key=lambda x:abs(x-tallon_position_in_world))
                c_meanie = str(c_meanie)
                c_meanie = (int(c_meanie[0]),int(c_meanie[1])) if (len(c_meanie)>1) else (0,int(c_meanie))
                currentmeanie = c_meanie
            print("closest meanie to Tallon is: ",c_meanie)

           

            #convert grid to 1d for processing  
              
            convert_world_to_1d = lambda x: np.ravel_multi_index(x, self.grid_size)

            def hit_the_wall(cell):
                try:
                    convert_world_to_1d(cell)
                except ValueError as e:
                    return True
                return False

            # make probs for each action
            North = [self.action_n_s_e_w_probability[i] for i in (0, 1, 2, 3)]
            South = [self.action_n_s_e_w_probability[i] for i in (1, 0, 3, 2)]
            West = [self.action_n_s_e_w_probability[i] for i in (2, 3, 1, 0)]
            East = [self.action_n_s_e_w_probability[i] for i in (3, 2, 0, 1)]
            actions = [North, South, East, West]
            for i, a in enumerate(actions):
                actions[i] = {'North':a[2], 'South':a[3], 'West':a[0], 'East':a[1]}
            
            def update_the_Probability_and_Rewards(cell, next_cell, index, action_prob):
                if cell == currentmeanie:
                    P[index, convert_world_to_1d(cell), convert_world_to_1d(cell)] = 1.0
                    R[convert_world_to_1d(cell), index] = self.meanies_reward
                
                elif cell == currentpit:
                    P[index, convert_world_to_1d(cell), convert_world_to_1d(cell)] = 1.0
                    R[convert_world_to_1d(cell), index] = self.pit_reward

                elif cell == currentbonus:
                    P[index, convert_world_to_1d(cell), convert_world_to_1d(cell)] = 1.0
                    R[convert_world_to_1d(cell), index] = self.bonus_reward

                elif hit_the_wall(next_cell):
                    P[index, convert_world_to_1d(cell), convert_world_to_1d(cell)] += action_prob
                    R[convert_world_to_1d(cell), index] = self.white_reward

                else:
                    P[index, convert_world_to_1d(cell), convert_world_to_1d(next_cell)] = action_prob
                    R[convert_world_to_1d(cell), index] = self.white_reward

            for index, action in enumerate(actions):
                for cell in np.ndindex(self.grid_size):
                    #North
                    next_cell = (cell[0]-1, cell[1])
                    update_the_Probability_and_Rewards(cell, next_cell, index, action['North'])

                    #South
                    next_cell = (cell[0]+1, cell[1])
                    update_the_Probability_and_Rewards(cell, next_cell, index, action['South'])

                    #West
                    next_cell = (cell[0], cell[1]-1)
                    update_the_Probability_and_Rewards(cell, next_cell, index, action['West'])

                    #East
                    next_cell = (cell[0], cell[1]+1)
                    update_the_Probability_and_Rewards(cell, next_cell, index, action['East'])
            return P,R

        except Exception as e:
            print()