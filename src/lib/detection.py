import cv2
import numpy as np
from src.dnn.classifier import *
from src.lib.fileschecker import *
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
from src.lib.plotgraph import *
import time
import math
import geocoder

def get_location():
    g = geocoder.ip('me')
    if g.ok:
        return (g.city,g.country)
    else:
        print("Could not retrieve location based on IP.")
        return None

def get_animated_color(frame_count):
    r = int(127 * (math.sin(0.1 * frame_count) + 1))
    g = int(127 * (math.sin(0.1 * frame_count + 2) + 1))
    b = int(127 * (math.sin(0.1 * frame_count + 4) + 1))
    return (b, g, r)

def detect_behaviour(video_file_path: str, model_file_path_1: str, model_file_path_2: str):

    try:

        if len(os.listdir(os.path.join(os.getcwd(), 'model_1'))) == 0 \
            and len(os.listdir(os.path.join(os.getcwd(), 'model_2'))) == 0 \
            and len(os.listdir(os.path.join(os.getcwd(), 'upload'))) == 0 and \
            os.path.exists(os.path.join(os.getcwd(),video_file_path)) and \
            os.path.exists(os.path.join(os.getcwd(),model_file_path_1)) and \
            os.path.exists(os.path.join(os.getcwd(),model_file_path_2)):

            check_path_or_not_video = file_save(video_file_path, 'upload')
            check_path_or_not_model_1 = file_save(model_file_path_1, 'model_1')
            check_path_or_not_model_2 = file_save(model_file_path_2, 'model_2')


        elif len(os.listdir(os.path.join(os.getcwd(), 'model_1'))) >0 \
            and len(os.listdir(os.path.join(os.getcwd(), 'model_2'))) >0 \
            and len(os.listdir(os.path.join(os.getcwd(), 'upload'))) >0:
            check_path_or_not_video = video_file_path
            check_path_or_not_model_1 = model_file_path_1
            check_path_or_not_model_2 = model_file_path_2

        else:
            print('Video or model file did not upload properly!')
            sys.exit(1)

        if not os.path.exists(os.path.join(os.getcwd(),'output')):

            os.makedirs(os.path.join(os.getcwd(),'output'),exist_ok=True)

        now = datetime.now()
        curr_time = now.strftime('%I:%M %p')
        
        if os.path.exists(check_path_or_not_video) and os.path.exists(check_path_or_not_model_1) and os.path.exists(check_path_or_not_model_2):
            video_file = os.listdir('upload')[0]
            model_file_1 = os.listdir('model_1')[0]
            model_file_2 = os.listdir('model_2')[0]

            classes = ['Attack', 'Neutral', 'Suspicious']

            attack_count,suspicious_count = 0,0
            # time_stamp_list = {
            #                    'Attacker Seen':[],
            #                    'Suspecious Act Seen':[]
            # }
            
            parent_keypoints_dict = {}
            keypoints_dict = {
                'id':[],
                'label':[],
                'keypoints':[]
            }


            video_path = os.path.join('upload', video_file)
            model_1_path = os.path.join('model_1', model_file_1)
            model_2_path = os.path.join('model_2', model_file_2)

            classifier = KeypointClassification(model_2_path)
            model_yolo = YOLO(model_1_path)
            cap = cv2.VideoCapture(video_path)
            
            out_file_save = cv2.VideoWriter(os.path.join(os.getcwd(),'output',f'out_1.mp4'),  
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        20.0, (900,600))
            
            print('Total Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            sidebar_width = 200
            width , height = 900,600
            start_time = time.time() 

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_tot , frame_count , flag = 0 , 0 , 0 # for fps purpose
            idx1 = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Create a black sidebar
                sidebar = np.zeros((height, sidebar_width, 3), dtype=np.uint8)
                results = model_yolo(frame, verbose=False)
                annotated_frame = results[0].plot(boxes=False)

                for r in results:
                    bound_box = r.boxes.xyxy
                    conf = r.boxes.conf.tolist()
                    keypoints = r.keypoints.xyn.tolist()
                    # print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

                    for index, box in enumerate(bound_box):
                        if conf[index] > 0.75:
                            x1, y1, x2, y2 = box.tolist()
                            data = {}

                            if len(keypoints[index]) == 17:
                                for j in range(len(keypoints[index])):
                                    data[f'x{j}'] = keypoints[index][j][0]
                                    data[f'y{j}'] = keypoints[index][j][1]

                                df = pd.DataFrame(data, index=[0])
                                input_keypoint = df.to_numpy().flatten()

                                results_ = classifier(input_keypoint)

                                if results_ == classes[0]:
                                    conf_text = f'{classes[0]} ({conf[index]:.2f})'
                                    color = (255, 7, 58)
                                    
                                elif results_ == classes[1]:
                                    conf_text = f'{classes[1]} ({conf[index]:.2f})'
                                    color = (57, 255, 20)
                                else:
                                    conf_text = f'{classes[2]} ({conf[index]:.2f})'
                                    color = (57, 20, 255)
                                
                                unique_id = f"{idx1}_{index}"
                                if tuple(input_keypoint) not in keypoints_dict.get('keypoints') and results_ not in keypoints_dict.get('label') or unique_id not in keypoints_dict.get('id'):
                                    keypoints_dict.get('id').append(unique_id)
                                    keypoints_dict.get('keypoints').append(tuple(input_keypoint))
                                    keypoints_dict.get('label').append(results_)
                                    flag = 1
                                else:
                                    flag = 0

                                if unique_id not in parent_keypoints_dict and flag==1:
                                    parent_keypoints_dict[unique_id] = {'label':results_,
                                                                        'keypoints':tuple(input_keypoint)}
                                    
                                
                                if unique_id in parent_keypoints_dict and flag==1:
                                    if parent_keypoints_dict[unique_id]['label'] == classes[0]:
                                        attack_count += 1
                                        
                                    elif parent_keypoints_dict[unique_id]['label'] == classes[2]:
                                        suspicious_count += 1

                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                            else:
                                print(f"Skipping frame {frame_tot} bounding box {index} due to incorrect keypoint length")

                frame_tot += 1
                idx1 = frame_tot
                frame_count += 1
                animated_color = get_animated_color(frame_tot)
                
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                fps = frame_tot / elapsed_time  # Calculate FPS

                # Display FPS on the sidebar
                cv2.putText(sidebar, f'FPS: {fps:.2f}', (5, 70),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,animated_color, 1)
                cv2.putText(sidebar, f'FRAME:{frame_count}/{total_frames}', (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,animated_color, 1)

                # Reset for the next interval
                frame_tot = 0
                start_time = time.time()  # Reset start_time

                # Display current time on the sidebar
                now = datetime.now()
                curr_time = now.strftime('%I:%M %p')
                
                if get_location():

                    city,country = get_location()
                    cv2.putText(sidebar, f'LOC:{city.upper()}.{country.upper()}', (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, animated_color, 1)
                else:
                    cv2.putText(sidebar, "LOC:KOLKATA.IN", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, animated_color, 1)
                    
                cv2.putText(sidebar, f'TIME:{curr_time}', (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, animated_color, 1)
                cv2.putText(sidebar, f'ATK COUNT:{attack_count}', (5, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, animated_color, 1)
                cv2.putText(sidebar, f'SUSP COUNT:{suspicious_count}', (5, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, animated_color, 1)

                # Resize the frame to fit the new dimensions
                frame_resized = cv2.resize(annotated_frame, (width, height))

                # Combine sidebar and frame
                combined_frame = cv2.cvtColor(np.hstack((sidebar, frame_resized)),cv2.COLOR_BGR2RGB)
                ret, jpeg = cv2.imencode('.jpg', combined_frame)
                if not ret:
                    break
                frame_data = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
                
            cap.release()
            
        else:
            print('Video or model file did not upload properly!')
            sys.exit(1)

    except Exception as e:
        print('An exception occurred:', type(e).__name__)
        print('Error message:', str(e))




