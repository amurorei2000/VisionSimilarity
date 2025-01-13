import os
import cv2
import numpy as np
# from ultralytics import YOLO
# from pytubefix import YouTube
# from pytubefix.cli import on_progress

# def download_video(url, filename, resolution):
#     yt = YouTube(url, on_progress_callback=on_progress)
#     print(yt.title)
    
#     yt.streams.filter(adaptive=True, res=resolution, file_extension='mp4').first().download(output_path='.', filename=filename)

# os.makedirs('./images', exist_ok=True)
# video_url = 'https://www.youtube.com/watch?v=Bf1wC33FDvE'
# download_video(video_url, 'movie/Movie.mp4', '720p')


def extract_keypoints(model, img, device='cpu'):
    results = model.predict(img, device=device)
    keypoints = results[0].keypoints.data.squeeze().cpu().numpy()
    return keypoints


def draw_keypoints(keypoints, frame, confidence_threshold):
    connections = [
      ([4, 2, 0, 1, 3], (0, 255, 0)),         # 얼굴(초록선)
      ([10, 8, 6, 5, 7, 9], (255, 0, 0)),     # 양 팔(파랑선)
      ([6, 12, 11, 5], (255, 0, 255)),        # 몸통(보라선)
      ([12, 14, 16], (0, 165, 255)),          # 오른다리(주황선)
      ([11, 13, 15], (0, 165, 255)),          # 왼다리(주황선)
    ]
    
    if len(keypoints) < 17:     # 예외 처리
        return frame
    
    for group, color in connections:
        for i in range(len(group) - 1):
            idx1, idx2 = group[i], group[i+1]
            x1, y1, score1 = keypoints[idx1]
            x2, y2, score2 = keypoints[idx2]
                
            if score1 > confidence_threshold and score2 > confidence_threshold:
                point1 = (int(x1), int(y1))
                point2 = (int(x2), int(y2))
                    
                # 점과 선 그리기
                cv2.circle(frame, point1, 3, (0, 0, 255), cv2.FILLED)
                # cv2.putText(frame, str(idx1), point1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                cv2.circle(frame, point2, 3, (0, 0, 255), cv2.FILLED)
                # cv2.putText(frame, str(idx2), point2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                cv2.line(frame, point1, point2, color, 2)
                    
    return frame


def play_webcam():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print('웹 캠을 연결해주세요!')
    else:
        while True:
            ret, show_frame1 = webcam.read()
            if ret:
                cv2.imshow('MyCamera', show_frame1)
                width = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                if cv2.waitKey(10) == ord('q'):
                    break
            else:
                print('웹캠 프레임을 읽는데 실패했습니다...')
                break

    webcam.release()
    cv2.destroyAllWindows()


# 부위별 방향 계산 및 정규화 함수
def normalize_keypoints(parts_idx, keypoints, confidence_threshold):
    norm_vec = []
    
    for i in range(len(parts_idx) -1):
        a1 = keypoints[parts_idx[i]]
        a2 = keypoints[parts_idx[i+1]]
        
        if a1[2] > confidence_threshold and a2[2] > confidence_threshold:
            calculate_vec= (a2[:2] - a1[:2]) / np.linalg.norm(a2[:2] - a1[:2])
            norm_vec.append(calculate_vec)
        else:
            norm_vec.append(np.zeros(shape=(2,), dtype=np.float32))
        
    return norm_vec


# 부위별 코사인 유사도 계산 함수
def cos_similiarity(v1, v2):
    sum_similarity = 0
    count = 0
    for i in range(len(v1)):
        if not np.allclose(v1[i], 0, atol=1e-5) and not np.allclose(v2[i], 0, atol=1e-5):
            dot_product = np.dot(v1[i], v2[i])
            # l2_norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            # sum_similarity += dot_product/l2_norm
            sum_similarity += dot_product
            count += 1
            
    if count != 0:
        return sum_similarity / count
    else:
        return 0
    

# 파츠별 유사도 측정 함수
def compute_similarity(keypoints1, keypoints2, confidence_threshold):
      head_idx = [4, 2, 0, 1, 3]
      left_arm_idx = [5, 7, 9]
      right_arm_idx = [6, 8, 10]
      body_idx = [6, 5, 12, 11]
      left_leg_idx = [12, 14, 16]
      right_leg_idx = [11, 13, 15]
      sims = []
      
      if len(keypoints1) < 17 or len(keypoints2) < 17:
            return sims
            
      # 1번 키포인트 해석
      head1 = normalize_keypoints(head_idx, keypoints1, confidence_threshold)
      left_arm1 = normalize_keypoints(left_arm_idx, keypoints1, confidence_threshold)
      right_arm1 = normalize_keypoints(right_arm_idx, keypoints1, confidence_threshold)
      body1 = normalize_keypoints(body_idx, keypoints1, confidence_threshold)
      left_leg1 = normalize_keypoints(left_leg_idx, keypoints1, confidence_threshold)
      right_leg1 = normalize_keypoints(right_leg_idx, keypoints1, confidence_threshold)

      # 2번 키포인트 해석
      head2 = normalize_keypoints(head_idx, keypoints2, confidence_threshold)
      left_arm2 = normalize_keypoints(left_arm_idx, keypoints2, confidence_threshold)
      right_arm2 = normalize_keypoints(right_arm_idx, keypoints2, confidence_threshold)
      body2 = normalize_keypoints(body_idx, keypoints2, confidence_threshold)
      left_leg2 = normalize_keypoints(left_leg_idx, keypoints2, confidence_threshold)
      right_leg2 = normalize_keypoints(right_leg_idx, keypoints2, confidence_threshold)
      
      # 1번과 2번의 파츠별 코사인 유사도 계산
      sims.append(cos_similiarity(head1, head2))
      sims.append(cos_similiarity(left_arm1, left_arm2))
      sims.append(cos_similiarity(right_arm1, right_arm2))
      sims.append(cos_similiarity(body1, body2))
      sims.append(cos_similiarity(left_leg1, left_leg2))
      sims.append(cos_similiarity(right_leg1, right_leg2))
      total_sim = np.mean(np.array(sims))
      sims.append(total_sim)
      
      return sims


# 캠 화면에 유사도를 표시하는 함수
def show_similarity_logs(img, sims):
    if sims != None and len(sims) == 7:
        log_name = ['Head', 'Left Arm', 'Right arm', 'Body', 'Left Leg', 'Right Leg', 'Mean Similarity']
        for i, val in enumerate(sims):
            # 일치율에 따라 폰트 색상 변경
            if val > 0.85:
                font_color = (0, 255, 0)
            elif val > 0.5:
                font_color = (0, 255, 255)
            else:
                font_color = (0, 0, 255)
                
            img = cv2.putText(img, f'{log_name[i]}: {val*100:.2f}%', (670, 40 + (i*25)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
            img = cv2.putText(img, f'{log_name[i]}: {val*100:.2f}%', (670, 40 + (i*25)), cv2.FONT_HERSHEY_COMPLEX, 0.8, font_color, 1)
    
    return img


# 이미지 2개 실행 함수
def compare_images_similarity(model, img1_path, img2_path, device, confidence_threshold):
    print(img1_path)
    img1 = cv2.imread(img1_path)
    img1 = cv2.resize(img1, (640, 480))
    keypoints1 = extract_keypoints(model, img1, device)
    
    print(img2_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (640, 480))
    keypoints2 = extract_keypoints(model, img2, device)
    
    all_similarities = compute_similarity(keypoints1, keypoints2, confidence_threshold)
    # all_similarities = ""
    return all_similarities


# 이미지 로드 함수
def load_img(file_path):
    load_img = cv2.imread(file_path)
    load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
    load_img = cv2.resize(load_img, (640, 480))
    return load_img

