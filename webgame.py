from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete

class WebGame(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,140,200), dtype=np.uint8)
        self.action_space = Discrete(8)
        # Capture game frames
        self.cap = mss()
        self.play_location = {'top': 600, 'left': -1380, 'width': 840, 'height': 600}
        self.score_location = {'top': 1328, 'left': -1820, 'width': 48, 'height': 30}
        self.done_location = {'top': 880, 'left': -1024, 'width': 128, 'height': 40}

    def step(self, action):
        
        #check score before performing action
        try:
            score_before_action = int(self.get_score().strip())
        except ValueError:
            score_before_action = 0
        
        #action mappings
        if action==0:
            pydirectinput.moveTo(x=-960, y=600, duration=0.1)
        elif action==1:
            pydirectinput.moveTo(x=-1260, y=900, duration=0.1)
        elif action==2:
            pydirectinput.moveTo(x=-960, y=1200, duration=0.1)
        elif action==3:
            pydirectinput.moveTo(x=-660, y=900, duration=0.1)
        elif action==4:
            pydirectinput.click(x=-960, y=600)
        elif action==5:
            pydirectinput.click(x=-1260, y=900)
        elif action==6:
            pydirectinput.click(x=-960, y=1200)
        elif action==7:
            pydirectinput.click(x=-660, y=900)
            
        #check if the game is done
        done, done_cap = self.get_done()
        
        #get next observation
        next_observation = self.get_observation()
        
        #check the score after action is performed
        if score_before_action ==0:
            score_after_action = 0
        else:
            try:  
                score_after_action = int(self.get_score().strip())
            except ValueError:
                score_after_action = score_before_action
        
        #reward given if score is increased
        if 0<(score_after_action-score_before_action)<=9:
            reward =1
        elif 10<=(score_after_action-score_before_action)<=30:
            reward =10
        elif 31<=(score_after_action-score_before_action)<=100:
            reward = 50
        elif (score_after_action-score_before_action)==0:
            reward =0
        else:
            reward =0
        
        info = {}
        
        return next_observation, reward, done, info
    
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=-960, y=900)
        #sleep 4 seconds coz score starts showing after 4 seconds
        time.sleep(4)
        
        return self.get_observation()
        
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.play_location))[:,:,:3]).astype(np.uint8)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
         
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        play_area = np.array(self.cap.grab(self.play_location))[:,:,:3]
        
        #image processing:
        gray = cv2.cvtColor(play_area, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (200,140))
        channel = np.reshape(resized, (1,140,200))
        
        return channel
    
    def get_done(self):
        done_img = np.array(self.cap.grab(self.done_location))
        done_strings = ['Play']
        done = False
        res = pytesseract.image_to_string(done_img)[:4]
        if res in done_strings:
            done = True
        
        return done, done_img
    
    def get_score(self):
        score_img = np.array(self.cap.grab(self.score_location))
        
#         gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (3,3), 0)
#         thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#         invert = 255 - opening

        score = pytesseract.image_to_string(score_img, lang='eng', config='--psm 6')
        
        return score