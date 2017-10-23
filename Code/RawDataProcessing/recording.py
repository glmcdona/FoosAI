import cv2
import xml.etree.ElementTree as ET
import os
from rod import *

import pprint
pp = pprint.PrettyPrinter(depth=6)

def xml_load_rect(rect):
    return (int(rect.find("X").text), int(rect.find("Y").text), int(rect.find("W").text), int(rect.find("H").text))

def xml_load_rgb(text):
    tokens = text.split(",")
    return (int(tokens[0]), int(tokens[1]), int(tokens[2]))
    

global refPt
refPt = (0,0)

def click_callback(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)
        cropping = True

class Recording(object):
    def __init__(self, recording_file):
        self.recording_file = recording_file
        self.tree = ET.parse(recording_file)
        self.root = self.tree.getroot()
        base_path = os.path.dirname(recording_file)
        
        # CROP
        self.crop = xml_load_rect(self.root.find("CROP").find("RECT"))
        
        
        # Files
        self.file_avi = os.path.join(base_path, self.root.find("RECORDING").find("AVI").text)
        self.frame_start = int(self.root.find("RECORDING").find("ACCELRATIONFRAMESTART").text)
        self.frame_end = int(self.root.find("RECORDING").find("ENDFRAME").text)
        
        # Rod tracking settings
        rod_tracking_alignment = self.root.find("RODS").find("TRACKING").find("ALIGNMENT").text
        rod_tracking_gap_colour = xml_load_rgb(self.root.find("RODS").find("TRACKING").find("GAP_COLOUR").text)
        rod_tracking_gap_colour_distance = int(self.root.find("RODS").find("TRACKING").find("GAP_COLOUR_DISTANCE").text)
        rod_tracking_gap_min_size = int(self.root.find("RODS").find("TRACKING").find("GAP_MIN_SIZE").text)
        rod_tracking_rod_width = int(self.root.find("RODS").find("TRACKING").find("ROD_WIDTH").text)
        
        # Video file
        self.cap = cv2.VideoCapture(self.file_avi)
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame = 0
        self.has_more = False
        
        # Rods
        self.rods = {}
        for rod in self.root.find("RODS").iter("ROD"):
            self.rods[rod.find("NAME").text] = Rod( xml_load_rect(rod.find("RECT")),rod.find("NAME").text, rod_tracking_gap_colour, rod_tracking_gap_colour_distance, rod_tracking_gap_min_size, rod_tracking_rod_width)
    
    def __del__(self):
        self.cap.release()
    
    def process(self):
        # Extract the rod positions
        pass
    
    def initialize(self):
        self._initialize_rod_gaps()
        self.cap.set(1,0)
        self.frame = 0
        self.has_more = True
    
    def get_next_frame(self):
        # Returns:
        # (cropped frame, {rod name = rod positions})
        if( self.cap.isOpened() and self.has_more ):
            ret, frame = self.cap.read()
            if ret==True:
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray,50,150,apertureSize = 3)

                # Cropped frame
                frame_cropped = frame[self.crop[1]:(self.crop[1]+self.crop[3]), self.crop[0]:(self.crop[0]+self.crop[2])]
                
                # Calculate rod positions
                rod_positions = {}
                for rod_name, rod in self.rods.items():
                    rod_positions[rod_name] = None

                    if self.frame % 120 == 0 or rod.rod_line == None:
                        rod.update_rod_line(edges)

                    failure_count = 0
                    if rod.rod_line != None:
                        (new_x, line, success) = rod.track_rod_position(frame)
                        if( not success ):
                            failure_count += 1
                        rod_positions[rod_name] = new_x

                self.frame += 1
                return (frame_cropped, rod_positions, failure_count)
            else:
                print("frame read failure")
                self.has_more = False
                return (None, None, 0)
        
        return (None, None, 0)
    
    def _initialize_rod_gaps(self):
        print("Loading rod gap tracking sizes...")
        key_frame_count = 30
        #key_frame_count = 5
        small_frame_set = []
        for i in range(key_frame_count):
            frame = i * (self.num_frames/key_frame_count)
            self.cap.set(1,frame)
            ret, frame = self.cap.read()
            small_frame_set.append(frame)
        
        # Load the rod gap sizes to track from a small subset of the frames
        for rod_name, rod in self.rods.items():
            size = rod.find_gap_size(small_frame_set)
            print("\t%s: gap size %i" % (rod_name, size))
    
    def play(self):
        self.initialize()
        
        # Extract the rod positions
        cv2.namedWindow("image")
        
        global refPt
        rod_offence_last_frame_x = 0
        rod_defence_last_frame_x = 0
        rod_goalie_last_frame_x = 0
        self.cap.set(1,3000)
        count = 0
        while(self.cap.isOpened()):
            cv2.setMouseCallback("image", click_callback)
            ret, frame = self.cap.read()
            frame_original = frame.copy()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,50,150,apertureSize = 3)
            
            # Crop region
            box = self.crop
            cv2.rectangle(frame, (box[0], box[1]),
                                (box[0] + box[2], box[1] + box[3]), 
                                (0,255,0) );
            
            # Draw the rods
            for rod_name, rod in self.rods.items():
                box = rod.box
                cv2.rectangle(frame, (box[0], box[1]),
                                    (box[0] + box[2], box[1] + box[3]), 
                                    (255,0,0) );
                
                if count % 120 == 0:
                    rod.update_rod_line(edges)
                
                rod_line = rod.rod_line
                
                if rod_line != None:
                    cv2.line(frame,rod_line[0],rod_line[1],(0,0,255),2)
                    
                    (new_x, line, success) = rod.track_rod_position(frame_original)
                    
                    if line != None:
                        rod_offence_last_frame_x = new_x
                        cv2.line(frame, line[0], line[1], (255,0,255), thickness=4)
            
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'%i,%i: rgb: %i,%i,%i' % (refPt[0],refPt[1],frame[refPt[1]][refPt[0]][0],frame[refPt[1]][refPt[0]][1],frame[refPt[1]][refPt[0]][2]),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
            
            cv2.imshow('image',frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey()
                
            count += 1
        
        cv2.destroyAllWindows()
        pass