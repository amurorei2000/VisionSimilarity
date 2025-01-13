import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import PoseDetecting
import cv2
import torch
from ultralytics import YOLO
import numpy as np


form_class = uic.loadUiType('PoseUI.ui')[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('../models/yolov8m-pose.pt')
        self.draw_points = True
        self.confidence_threshold = 0.5
        self.onFrame = False
        self.movieCam = None
        self.webcam = None
        self.image1_path = ""
        self.image2_path = ""
        
        self.log_name = ['Head', 'Left Arm', 'Right arm', 'Body', 'Left Leg', 'Right Leg', 'Mean']
        self.text_boxes = [self.text_Head, self.text_LeftArm, self.text_RightArm, self.text_Body,
                           self.text_LeftLeg, self.text_RightLeg, self.text_Mean]
        
        # 버튼에 함수 바인딩하기
        self.btn_compare_images.clicked.connect(self.On_Clicked_Compare_Images)
        self.btn_compare_movie_cam.clicked.connect(self.On_Clicked_Compare_MovieCam)
        self.btn_clear_cam.clicked.connect(self.On_Clicked_Off_movieCam)
        self.btn_select_img1.clicked.connect(self.OpenFileExplorer_to_window1)
        self.btn_select_img2.clicked.connect(self.OpenFileExplorer_to_window2)
        
        # 버튼 색상 입히기
        self.btn_compare_images.setStyleSheet('QPushButton {background-color: #62a8de; color: white}')
        self.btn_compare_movie_cam.setStyleSheet('QPushButton {background-color: #62a8de; color: white}')
        self.btn_clear_cam.setStyleSheet('QPushButton {background-color: #62a8de; color: white}')
        
    def On_Clicked_Compare_Images(self):
        # self.On_Clicked_Off_movieCam()
        
        if not (self.image1_path == "" or self.image2_path == ""):
            all_similarities = PoseDetecting.compare_images_similarity(self.model,
                                                                       self.image1_path[0],
                                                                       self.image2_path[0],
                                                                       self.device,
                                                                       self.confidence_threshold)
        
            # 로그 텍스트를 윈도우에 출력
            if all_similarities != None and len(all_similarities) == 7:
                self.Show_Simliarity(all_similarities)
                
        else:
            self.text_log.setText(f'{self.text_log.toPlainText()}\n비교할 이미지를 선택해 주세요')
    
    # 이미지를 1번 윈도우에 출력
    def OpenFileExplorer_to_window1(self):
        self.image1_path = QFileDialog.getOpenFileName(self)
        self.text_log.setText(f'{self.text_log.toPlainText()}\n{self.image1_path[0]}')
        
        load_img = PoseDetecting.load_img(self.image1_path[0])
        height, width, _ = load_img.shape
        bytes_per_line = width * 3
        q_image = QImage(load_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.img_01.setPixmap(QPixmap(pixmap))
    
    # 이미지를 2번 윈도우에 출력            
    def OpenFileExplorer_to_window2(self):
        self.image2_path = QFileDialog.getOpenFileName(self)
        self.text_log.setText(f'{self.text_log.toPlainText()}\n{self.image2_path[0]}')
        
        load_img = PoseDetecting.load_img(self.image2_path[0])
        height, width, _ = load_img.shape
        bytes_per_line = width * 3
        q_image = QImage(load_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.img_02.setPixmap(QPixmap(pixmap))
        
    def On_Clicked_Compare_MovieCam(self):
        self.text_log.setText(f'{self.text_log.toPlainText()}\n캠에 잘 보일 수 있게 서 주세요.')
        if not self.onFrame:
            # 영상 및 캠 Open
            self.movieCam = cv2.VideoCapture("./movie/Movie.mp4")
            self.webcam = cv2.VideoCapture(0)
            self.onFrame = True
            
            if self.movieCam.get(cv2.CAP_PROP_POS_FRAMES) == self.movieCam.get(cv2.CAP_PROP_FRAME_COUNT):
                self.movieCam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
            while self.onFrame and self.movieCam.isOpened() and self.webcam.isOpened():
                ret1, movie_frame = self.movieCam.read()
                ret2, webcam_frame = self.webcam.read()
                
                if ret1 and ret2:
                    # 키 포인트 탐지
                    movie_frame = cv2.resize(movie_frame, (640, 480))
                    keypoints1 = PoseDetecting.extract_keypoints(self.model, movie_frame, self.device)
                    
                    webcam_frame = cv2.flip(webcam_frame, 1)
                    keypoints2 = PoseDetecting.extract_keypoints(self.model, webcam_frame, self.device)
                    
                    # 키 포인트 그리기
                    if self.draw_points:
                        movie_frame = PoseDetecting.draw_keypoints(keypoints1, movie_frame, self.confidence_threshold)
                        webcam_frame = PoseDetecting.draw_keypoints(keypoints2, webcam_frame, self.confidence_threshold)
                    
                    movie_frame = cv2.cvtColor(movie_frame, cv2.COLOR_BGR2RGB)
                    webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
                    
                    # 코사인 유사도 체크
                    all_similarities = PoseDetecting.calculate_similarity(keypoints1, keypoints2, self.confidence_threshold)
                
                # 빈 이미지 예외 처리    
                elif ret1 == False and ret2 == True:
                    movie_frame = np.zeros(shape=(480, 640, 3), dtype=np.int8)
                    movie_frame = cv2.cvtColor(movie_frame, cv2.COLOR_BGR2RGB)
                
                elif ret1 == True and ret2 == False:
                    webcam_frame = np.zeros(shape=(480, 640, 3), dtype=np.int8)
                    webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
                else:
                    movie_frame = np.zeros(shape=(480, 640, 3), dtype=np.int8)
                    webcam_frame = np.zeros(shape=(480, 640, 3), dtype=np.int8)

                # 영상 출력
                height1, width1, _ = movie_frame.shape
                bytes_per_line1 = width1 * 3
                q_image1 = QImage(movie_frame.data, width1, height1, bytes_per_line1, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(q_image1)
                self.img_01.setPixmap(QPixmap(pixmap1))
                
                # 캠 출력
                height2, width2, _ = webcam_frame.shape
                bytes_per_line2 = width2 * 3
                q_image2 = QImage(webcam_frame.data, width2, height2, bytes_per_line2, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(q_image2)
                self.img_02.setPixmap(QPixmap(pixmap2))
                
                # 로그 출력
                if all_similarities != None and len(all_similarities) == 7:
                    self.Show_Simliarity(all_similarities)
                    
                if cv2.waitKey(10) == ord('q'):
                    break
                    
    def On_Clicked_Off_movieCam(self):
        self.onFrame = False
        self.img_01.clear()
        self.img_02.clear()
        self.text_log.clear()
        
        for i, tb in enumerate(self.text_boxes):
            tb.setText(f'{self.log_name[i]}: 0.00%')
        
        if not self.movieCam == None:
            self.movieCam.release()
        if not self.webcam == None:
            self.webcam.release()
            
    def Show_Simliarity(self, all_sims):
        for i, val in enumerate(all_sims):
            if val > 0.85: 
                self.text_boxes[i].setText(f'{self.log_name[i]}: <span style="color: #1e8a6d;">{val * 100:.2f}%</span><br>')
            elif val > 0.5: 
                self.text_boxes[i].setText(f'{self.log_name[i]}: <span style="color: #e38222;">{val * 100:.2f}%</span><br>')
            else:
                self.text_boxes[i].setText(f'{self.log_name[i]}: <span style="color: #ff0000;">{val * 100:.2f}%</span><br>')

        
if __name__ == '__main__':
    # 창 띄우기
    app = QApplication(sys.argv)
    view_window = MyWindow()
    view_window.show()
    view_window.On_Clicked_Off_movieCam()
    sys.exit(app.exec_())

