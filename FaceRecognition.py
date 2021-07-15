import cv2, time
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from retinaface import RetinaFace
from preprocess import alignImage, preprocessImage
from inference import imgToEmbedding, identifyFace
from visualization import drawFrameWithBbox
from backbones import get_model

def loadModel(backbone_name, weight_path, fp16=False):
    model = get_model(backbone_name, fp16=fp16)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model = model.cuda()
    return model

def faceRecognition(input_video_path, out_video_path, model_path, db):
    
    cap = cv2.VideoCapture(input_video_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    vid_writer = cv2.VideoWriter(out_video_path, codec, vid_fps, vid_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 갯수: ", frame_cnt)
    btime = time.time()

    detect_model = RetinaFace.build_model()
    if type(model_path) == str:
        identify_model = loadModel("r50", model_path)
    else:
        identify_model = model_path
    
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
              align_face_imgs = alignImage(img_frame, detect_faces)
              identities = []
              for face_img in align_face_imgs:
                  process_face_img, process_face_img_flip = preprocessImage(face_img)
                  embedding = imgToEmbedding(process_face_img, identify_model, img_flip=process_face_img_flip)
                  identity = identifyFace(embedding, db)
                  identities.append(identity)
              img_frame = drawFrameWithBbox(img_frame, detect_faces, identities)
        except:
            print("에러가 발생했습니다. 현재까지 상황을 저장합니다")
            break
        print('frame별 detection 수행 시간:', round(time.time() - stime, 4),frame_idx)
        frame_idx += 1
        vid_writer.write(img_frame)
    vid_writer.release()
    cap.release()

    print("최종 완료 수행 시간: ", round(time.time() - btime, 4))

def createEmbedingDB(db_folder_path, model, img_show=False):
    db = {
        "labels": [],
        "embedding": []
    }
    face_folders = os.listdir(db_folder_path)
    for face_folder in face_folders:
        label = face_folder
        img_folder_path = db_folder_path + "/" + face_folder
        img_folders = os.listdir(img_folder_path)
        print(label)
        labels = []
        for img_path in img_folders:
            img = cv2.imread(img_folder_path + "/" + img_path)
            faces = RetinaFace.detect_faces(img_path=img)
        # bbox
            if (type(faces) == dict):
                alignment_face_imgs = alignImage(img, faces)

            # embedding
                for face_img in alignment_face_imgs:
                    process_face_img, process_face_img_flip = preprocessImage(face_img)
                    embedding = imgToEmbedding(process_face_img, model, img_flip=process_face_img_flip)
                    db["embedding"].append(embedding)
                    db['labels'].append(label)
                    labels.append(label)
                if img_show:
                    img_bbox = drawFrameWithBbox(img, faces, labels)
                    plt.imshow(img_bbox)
                    plt.show()
    db['embedding'] = normalize(db['embedding'])
    return db