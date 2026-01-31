import sys
import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
from collections import deque, Counter
from model.board_definition import BoardDefinition
from model.level_config import LevelConfig
from settings import *
from levels.level_content_initializer import LevelContentInitializer
from model.direction import Direction

# -------------------------
# Configuration
# -------------------------
CAMERA_INDEX = 0
SIDEBAR_WIDTH = 320
CAM_HEIGHT = 240

# --- FACE JOYSTICK SETTINGS ---
# Lower thresholds because head movement is smaller than hand movement
SMOOTHING_WINDOW = 4       
DEADZONE_THRESHOLD = 0.03  # Head must move 3% of screen width to register
LOCK_ANGLE_BUFFER = 0.35   
MOUTH_OPEN_THRESHOLD = 0.04 # Threshold to detect mouth open (for pause)

class Game:
    def __init__(self):
        pygame.init()
        self.game_width = RESOLUTION[0]
        self.game_height = RESOLUTION[1]
        self.canvas_width = self.game_width + SIDEBAR_WIDTH
        self.canvas_height = max(self.game_height, CAM_HEIGHT)
        self.screen = pygame.display.set_mode((1100, 600), pygame.RESIZABLE)
        pygame.display.set_caption("Pac-Man Face Controller")
        self.canvas = pygame.Surface((self.canvas_width, self.canvas_height))
        self.game_surface = pygame.Surface((self.game_width, self.game_height))
        self.timer = pygame.time.Clock()
        
        self.game_engine = self.init_game(self.game_surface)
        self.game_start_sfx = pygame.mixer.Sound('media/game_start.wav')

        # --- CHANGED: FACE MESH SETUP ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cam_surface = None
        
        # State
        self.gesture_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_committed_direction = None 
        self.waiting_for_start = True
        self.is_paused = False
        
        # --- NEW: CALIBRATION STATE ---
        # We need to know where "Center" is for your nose.
        self.neutral_nose_pos = None # (x, y)
        
        self.game_surface.fill([12, 2, 25])
        self.game_engine.tick() 
        print("JUDGE MODE: Face Joystick. Center head and press SPACE.")

    def init_game(self, surface_to_draw_on):
        board = BOARD.copy()
        board_definition = BoardDefinition(board)
        level_1 = LevelConfig(wall_color='blue', gate_color='white',
                              board_definition=board_definition, power_up_limit=POWER_UP_LIMIT)
        level_init = LevelContentInitializer(level_1, surface_to_draw_on)
        return level_init.init_game_engine()

    def check_mouth_open(self, lm):
        """
        Pause Gesture: Is mouth open?
        Landmark 13: Upper Lip
        Landmark 14: Lower Lip
        """
        upper_lip = lm[13]
        lower_lip = lm[14]
        distance = math.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)
        return distance > MOUTH_OPEN_THRESHOLD

    def process_input(self):
        if not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        status_text = "NEUTRAL"
        status_color = (200, 200, 200)

        if result.multi_face_landmarks:
            # Get the first face detected
            face_landmarks = result.multi_face_landmarks[0]
            lm = face_landmarks.landmark
            
            # --- LANDMARK DEFINITIONS ---
            #
            # ID 1 is the tip of the nose
            nose_x, nose_y = lm[1].x, lm[1].y
            
            # Draw the nose tip
            cv2.circle(frame, (int(nose_x*w), int(nose_y*h)), 5, (255, 0, 255), -1)

            # --- WAITING / CALIBRATION ---
            if self.waiting_for_start or self.game_engine.game_over:
                # While waiting, we constantly update the neutral position
                # so that the moment they press space, that position is locked.
                self.neutral_nose_pos = (nose_x, nose_y)
                
                # Draw calibration UI
                cv2.putText(frame, "CENTER HEAD", (int(w/2)-80, int(h/2)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, (int(nose_x*w), int(nose_y*h)), 20, (0, 255, 255), 2)

            else:
                # --- GAMEPLAY LOGIC ---
                
                # 1. Pause Check (Mouth Open)
                if self.check_mouth_open(lm):
                    self.is_paused = True
                    cv2.putText(frame, "MOUTH OPEN: PAUSE", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    self.is_paused = False
                    
                    # 2. FACE JOYSTICK LOGIC
                    # Compare current nose pos to the LOCKED neutral pos
                    if self.neutral_nose_pos is not None:
                        ref_x, ref_y = self.neutral_nose_pos
                        
                        # Draw the "Anchor" (Where your head started)
                        cv2.circle(frame, (int(ref_x*w), int(ref_y*h)), 5, (100, 100, 100), -1)
                        cv2.circle(frame, (int(ref_x*w), int(ref_y*h)), 40, (50, 50, 50), 1) # Deadzone ring
                        
                        # Draw Line connecting Anchor to Nose
                        cv2.line(frame, (int(ref_x*w), int(ref_y*h)), (int(nose_x*w), int(nose_y*h)), (255, 255, 0), 2)

                        # Vector Calculation
                        dx = nose_x - ref_x
                        dy = nose_y - ref_y
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        detected_direction = None

                        # 3. Deadzone Check
                        if distance > DEADZONE_THRESHOLD:
                            angle = math.atan2(dy, dx)
                            new_direction = None
                            
                            # 4. Sticky Angle Logic (Same as before)
                            if self.last_committed_direction == Direction.UP:
                                 if (-2.35 - LOCK_ANGLE_BUFFER) < angle <= (-0.78 + LOCK_ANGLE_BUFFER): new_direction = Direction.UP
                            elif self.last_committed_direction == Direction.DOWN:
                                 if (0.78 - LOCK_ANGLE_BUFFER) <= angle < (2.35 + LOCK_ANGLE_BUFFER): new_direction = Direction.DOWN
                            elif self.last_committed_direction == Direction.RIGHT:
                                 if (-0.78 - LOCK_ANGLE_BUFFER) < angle < (0.78 + LOCK_ANGLE_BUFFER): new_direction = Direction.RIGHT
                            elif self.last_committed_direction == Direction.LEFT:
                                 if (2.35 - LOCK_ANGLE_BUFFER) < angle or angle <= (-2.35 + LOCK_ANGLE_BUFFER): new_direction = Direction.LEFT

                            if new_direction is None:
                                if -0.78 < angle < 0.78: new_direction = Direction.RIGHT
                                elif 0.78 <= angle < 2.35: new_direction = Direction.DOWN
                                elif -2.35 < angle <= -0.78: new_direction = Direction.UP
                                else: new_direction = Direction.LEFT
                            
                            self.gesture_history.append(new_direction)
                            most_common_dir, count = Counter(self.gesture_history).most_common(1)[0]
                            
                            # 5. Continuous Input
                            if count >= 2:
                                self.game_engine.direction_command = most_common_dir
                                self.last_committed_direction = most_common_dir
                                
                                # Visuals
                                if most_common_dir == Direction.LEFT: 
                                    status_text = "LEFT"
                                    self.highlight_slice(frame, "LEFT", (255, 255, 0), ref_x, ref_y)
                                elif most_common_dir == Direction.RIGHT: 
                                    status_text = "RIGHT"
                                    self.highlight_slice(frame, "RIGHT", (0, 255, 255), ref_x, ref_y)
                                elif most_common_dir == Direction.UP: 
                                    status_text = "UP"
                                    self.highlight_slice(frame, "UP", (0, 255, 0), ref_x, ref_y)
                                elif most_common_dir == Direction.DOWN: 
                                    status_text = "DOWN"
                                    self.highlight_slice(frame, "DOWN", (0, 0, 255), ref_x, ref_y)
                                status_color = (0, 255, 0)
                        else:
                            self.last_committed_direction = None
                    
        cv2.putText(frame, f"{status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        frame_resized = cv2.resize(frame, (SIDEBAR_WIDTH, CAM_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        self.cam_surface = pygame.surfarray.make_surface(frame_rgb)

    def highlight_slice(self, frame, direction, color, cx_rel, cy_rel):
        """Draws wedges relative to the LOCKED CALIBRATION CENTER."""
        h, w, _ = frame.shape
        cx = int(cx_rel * w)
        cy = int(cy_rel * h)
        overlay = frame.copy()
        pts = []
        if direction == "RIGHT": pts = np.array([[cx, cy], [w, -h], [w, 2*h]])
        elif direction == "DOWN": pts = np.array([[cx, cy], [2*w, h], [-w, h]])
        elif direction == "LEFT": pts = np.array([[cx, cy], [0, 2*h], [0, -h]])
        elif direction == "UP": pts = np.array([[cx, cy], [-w, 0], [2*w, 0]])
        if len(pts) > 0:
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def update(self):
        self.timer.tick(FPS)
        self.canvas.fill([12, 2, 25])
        
        if not self.waiting_for_start:
            self.game_surface.fill([12, 2, 25])
            if not self.is_paused:
                self.game_engine.tick() 
        
        sidebar_x = self.game_width
        pygame.draw.rect(self.canvas, (30, 30, 40), (sidebar_x, 0, SIDEBAR_WIDTH, self.canvas_height))
        self.canvas.blit(self.game_surface, (0, 0))
        
        if self.cam_surface is not None:
            cam_x, cam_y = sidebar_x, 50
            self.canvas.blit(self.cam_surface, (cam_x, cam_y))
            pygame.draw.rect(self.canvas, (0, 255, 0), (cam_x, cam_y, SIDEBAR_WIDTH, CAM_HEIGHT), 2)
            
            # --- UI LOGIC ---
            current_time = pygame.time.get_ticks()
            
            if self.waiting_for_start or self.game_engine.game_over:
                label_text = "LOCK HEAD & PRESS SPACE"
                if current_time % 1000 < 500:
                    label_color = (255, 255, 0)
                else:
                    label_color = (255, 255, 255)
            elif self.is_paused:
                label_text = "!!! MOUTH OPEN - PAUSED !!!"
                label_color = (255, 0, 0)
            else:
                label_text = ">>> FACE LINK ACTIVE <<<"
                if current_time % 500 < 250:
                    label_color = (0, 255, 255) 
                else:
                    label_color = (200, 255, 255) 
            
            font = pygame.font.SysFont('Arial', 16, bold=True)
            label = font.render(label_text, True, label_color)
            text_rect = label.get_rect(center=(cam_x + SIDEBAR_WIDTH//2, cam_y + CAM_HEIGHT + 20))
            self.canvas.blit(label, text_rect)

        scaled_surface = pygame.transform.smoothscale(self.canvas, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def reset_game(self):
        self.game_start_sfx.play()
        self.game_engine = self.init_game(self.game_surface)
        self.gesture_history.clear()
        self.waiting_for_start = False
        self.last_committed_direction = None
        self.is_paused = False
        # Note: We do NOT reset neutral_nose_pos here so the user doesn't lose calibration mid-game.
        # They can re-calibrate by restarting.

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                sys.exit()
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.waiting_for_start: 
                        # LOCK CALIBRATION NOW
                        self.waiting_for_start = False
                        self.game_start_sfx.play()
                    elif self.game_engine.game_over: 
                        self.reset_game()
                    else:
                        self.is_paused = not self.is_paused
            
            if event.type == GHOST_EATEN_EVENT:
                self.game_engine.play_ghost_runsaway_sound()
            if event.type == PLAYER_EATEN_EVENT:
                self.game_engine.play_player_eaten_sound()

    def cleanup(self):
        self.cap.release()
        pygame.quit()

    def run(self):
        while True:
            self.check_events()
            self.process_input()
            self.update()

if __name__ == '__main__':
    game = Game()
    game.run()