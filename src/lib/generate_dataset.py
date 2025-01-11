import tqdm
import pandas as pd
from src.dnn.classifier import *


class GenerateDataset_OR_GetStructuredRow(object):
    def __init__(self, mat,model_path:str,instant_input:bool):
        self.instant_input = instant_input
        self.model_path=model_path
        self.names = {
            'Nose_X': [], 'Nose_Y': [],
            'Left_Eye_X': [], 'Left_Eye_Y': [],
            'Right_Eye_X': [], 'Right_Eye_Y': [],
            'Left_Ear_X': [], 'Left_Ear_Y': [],
            'Right_Ear_X': [], 'Right_Ear_Y': [],
            'Left_Shoulder_X': [], 'Left_Shoulder_Y': [],
            'Right_Shoulder_X': [], 'Right_Shoulder_Y': [],
            'Left_Elbow_X': [], 'Left_Elbow_Y': [],
            'Right_Elbow_X': [], 'Right_Elbow_Y': [],
            'Left_Wrist_X': [], 'Left_Wrist_Y': [],
            'Right_Wrist_X': [], 'Right_Wrist_Y': [],
            'Left_Hip_X': [], 'Left_Hip_Y': [],
            'Right_Hip_X': [], 'Right_Hip_Y': [],
            'Left_Knee_X': [], 'Left_Knee_Y': [],
            'Right_Knee_X': [], 'Right_Knee_Y': [],
            'Left_Ankle_X': [], 'Left_Ankle_Y': [],
            'Right_Ankle_X': [], 'Right_Ankle_Y': [],
        }
        if not self.instant_input:
            self.names['Class'] = [] 
            self.label = 'Attack'

        self.result_keypoint = mat

    def generate_(self):
        for keypoints in self.result_keypoint:
            self.names['Nose_X'].append(keypoints[0][0])
            self.names['Nose_Y'].append(keypoints[0][1])
            self.names['Left_Eye_X'].append(keypoints[1][0])
            self.names['Left_Eye_Y'].append(keypoints[1][1])
            self.names['Right_Eye_X'].append(keypoints[2][0])
            self.names['Right_Eye_Y'].append(keypoints[2][1])
            self.names['Left_Ear_X'].append(keypoints[3][0])
            self.names['Left_Ear_Y'].append(keypoints[3][1])
            self.names['Right_Ear_X'].append(keypoints[4][0])
            self.names['Right_Ear_Y'].append(keypoints[4][1])
            self.names['Left_Shoulder_X'].append(keypoints[5][0])
            self.names['Left_Shoulder_Y'].append(keypoints[5][1])
            self.names['Right_Shoulder_X'].append(keypoints[6][0])
            self.names['Right_Shoulder_Y'].append(keypoints[6][1])
            self.names['Left_Elbow_X'].append(keypoints[7][0])
            self.names['Left_Elbow_Y'].append(keypoints[7][1])
            self.names['Right_Elbow_X'].append(keypoints[8][0])
            self.names['Right_Elbow_Y'].append(keypoints[8][1])
            self.names['Left_Wrist_X'].append(keypoints[9][0])
            self.names['Left_Wrist_Y'].append(keypoints[9][1])
            self.names['Right_Wrist_X'].append(keypoints[10][0])
            self.names['Right_Wrist_Y'].append(keypoints[10][1])
            self.names['Left_Hip_X'].append(keypoints[11][0])
            self.names['Left_Hip_Y'].append(keypoints[11][1])
            self.names['Right_Hip_X'].append(keypoints[12][0])
            self.names['Right_Hip_Y'].append(keypoints[12][1])
            self.names['Left_Knee_X'].append(keypoints[13][0])
            self.names['Left_Knee_Y'].append(keypoints[13][1])
            self.names['Right_Knee_X'].append(keypoints[14][0])
            self.names['Right_Knee_Y'].append(keypoints[14][1])
            self.names['Left_Ankle_X'].append(keypoints[15][0])
            self.names['Left_Ankle_Y'].append(keypoints[15][1])
            self.names['Right_Ankle_X'].append(keypoints[16][0])
            self.names['Right_Ankle_Y'].append(keypoints[16][1])
            if not self.instant_input:
                self.names['Class'].append(self.label)
    
    def to_dataframe(self):
        return pd.DataFrame(self.names)

    def save_to_csv(self, filename: str):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
