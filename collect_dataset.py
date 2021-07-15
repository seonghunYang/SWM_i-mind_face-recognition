import cv2, time, torch, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from retinaface import RetinaFace
from preprocess import alignImage, preprocessImage
from inference import imgToEmbedding, identifyFace, calculateDistance
from visualization import drawFrameWithBbox
from backbones import get_model
from utils.utils import checkExtenstion 


def collectFaceImageWithSeed(input_video_path, model_path, seed, threshold):
    
    cap = cv2.VideoCapture(input_video_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 갯수: ", frame_cnt)
    btime = time.time()

    detect_model = RetinaFace.build_model()
    if type(model_path) == str:
        identify_model = loadModel("r50", model_path)
    else:
        identify_model = model_path
    
    collect_img = [[] for _ in range(len(seed['labels']))]
    
    frame_idx = 0
    while True:
        has_frame, img_frame = cap.read()
        if not has_frame:
            print("처리 완료")
            break
        stime = time.time()
        try:
            detect_faces = RetinaFace.detect_faces(img_path=img_frame, model=detect_model)
            if (type(detect_faces) == dict):
                crop_face_imgs = []
                for key in detect_faces.keys():
                    face = detect_faces[key]
                    facial_area = face['facial_area']
                    crop_face = cropFullFace(img_frame, facial_area)
                    crop_face = cv2.resize(crop_face, (112, 112))
                    crop_face_imgs.append(crop_face)

                for face_img in crop_face_imgs:
                    process_face_img, process_face_img_flip = preprocessImage(face_img)
                    embedding = imgToEmbedding(process_face_img, identify_model, img_flip=process_face_img_flip)
                    # 신원 확인이 아니라 비교 하고 이미지 수집해야함
                    seed_idx, seed_distance = verifyWithSeed(embedding, seed, threshold)
                    if seed_idx >= 0:
                        collect_img[seed_idx].append([face_img, seed_distance])
        except:
            print("에러가 발생했습니다. 현재까지 상황을 저장합니다")
            break
        print('frame별 detection 수행 시간:', round(time.time() - stime, 4),frame_idx)
        frame_idx += 1
    cap.release()

    print("최종 완료 수행 시간: ", round(time.time() - btime, 4))
    return collect_img

def verifyWithSeed(embedding, seed, threshold):
    min_distance = threshold
    min_idx = -1

    for idx, seed_embedding in enumerate(seed['embedding']):
        distance = calculateDistance(embedding, seed_embedding)
        
        if distance < min_distance:
            min_idx = idx
            min_distance = distance

    return min_idx, min_distance
    
def cropFullFace(img, area):
    img_copy = img.copy()
    left = area[0]
    top = area[1]
    right = area[2]
    bottom = area[3]
    center = [(top + bottom) // 2, (left + right) // 2]
    y_half_len = (bottom - top) // 2
    full_left = max(0, center[1] - y_half_len) 
    full_right = min(center[1] + y_half_len, len(img_copy[1])) 
    return img_copy[top: bottom, full_left: full_right, ::-1]

def createEmbedingSeed(root_path, folder_name , model, img_show=False):
    seed = {
        "labels": [],
        "embedding": []
    }
    label = folder_name
    labels = []
    seed_img_folder = root_path + "/" + folder_name
    seed_img_list = os.listdir(seed_img_folder)
    for seed_img in seed_img_list:
        if not checkExtenstion(seed_img):
            continue
        img = cv2.imread(seed_img_folder + "/" + seed_img)
        detect_faces = RetinaFace.detect_faces(img_path=img)
    # bbox
        if (type(detect_faces) == dict):
            crop_face_imgs = []
            for key in detect_faces.keys():
                face = detect_faces[key]
                facial_area = face['facial_area']
                crop_face = cropFullFace(img, facial_area)
                crop_face = cv2.resize(crop_face, (112, 112))
                crop_face_imgs.append(crop_face)
        # embedding
            for face_img in crop_face_imgs:
                process_face_img, process_face_img_flip = preprocessImage(face_img)
                embedding = imgToEmbedding(process_face_img, model, img_flip=process_face_img_flip)
                seed["embedding"].append(embedding)
                seed['labels'].append(label)
                labels.append(label)
                if img_show:
                    plt.imshow(face_img)
                    plt.show()
    seed['embedding'] = normalize(seed['embedding'])
    return seed