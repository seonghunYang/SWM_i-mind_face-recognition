import cv2, time, torch, os, traceback, logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from retinaface import RetinaFace
from preprocess import alignImage, preprocessImage
from inference import imgToEmbedding, identifyFace, calculateDistance
from visualization import drawFrameWithBbox
from backbones import get_model
from utils.utils import checkImgExtenstion 


def collectFaceImageWithSeed(input_video_path, model_path, seed, threshold):
    
    cap = cv2.VideoCapture(input_video_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 갯수: ", frame_cnt)
    btime = time.time()

    detect_model = RetinaFace.build_model()
    if type(model_path) == str:
        recognition_model = loadModel("r50", model_path)
    else:
        recognition_model = model_path
    
    collect_img = [[] for _ in range(len(seed['labels']))]
    
    frame_idx = 0
    while True:
        cap.set(1, frame_idx)
        has_frame, img_frame = cap.read()
        if not has_frame:
            print("처리 완료")
            break
        stime = time.time()
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
                process_face_img, process_flip_face_img = preprocessImage(face_img)
                embedding = imgToEmbedding(process_face_img, recognition_model, img_flip=process_flip_face_img)
                # 신원 확인이 아니라 비교 하고 이미지 수집해야함
                seed_idx_list, seed_distance_list = verifyWithSeed(embedding, seed, threshold)
                if len(seed_idx_list) > 0:
                    for idx, seed_idx in enumerate(seed_idx_list):
                        collect_img[seed_idx].append([face_img, seed_distance_list[idx]])
        print('frame별 detection 수행 시간:', round(time.time() - stime, 4),frame_idx)
        frame_idx += 3
    cap.release()

    print("최종 완료 수행 시간: ", round(time.time() - btime, 4))
    
    return collect_img

def verifyWithSeed(embedding, seed, threshold):
    idx_list = []
    distance_list = []
    for idx, seed_embedding in enumerate(seed['embedding']):
        distance = calculateDistance(embedding, seed_embedding)
        
        if distance < threshold:
            idx_list.append(idx)
            distance_list.append(distance)

    return idx_list, distance_list
    
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
    labels = []
    seed_folder_path = root_path + "/" + folder_name
    seed_img_list = os.listdir(seed_folder_path)
    for idx, seed_img in enumerate(seed_img_list):
        if not checkImgExtenstion(seed_img):
            continue
        img = cv2.imread(seed_folder_path + "/" + seed_img)
        detect_face = RetinaFace.detect_faces(img_path=img)
        if (type(detect_face) == dict):
            key = list(detect_face.keys())[0]
            face = detect_face[key]
            facial_area = face['facial_area']
            crop_face = cropFullFace(img, facial_area)
            crop_face = cv2.resize(crop_face, (112, 112))
            
            #embedding
            process_face_img, process_flip_face_img = preprocessImage(crop_face)
            embedding = imgToEmbedding(process_face_img, model, img_flip=process_flip_face_img)
            seed["embedding"].append(embedding)
            seed['labels'].append(idx)
            if img_show:
                plt.imshow(crop_face)
                plt.show()
    seed['embedding'] = normalize(seed['embedding'])
    return seed

def filterCosineSimilarity(collect_img, model):
    embeddings = []
    for i in range(len(collect_img)):
        img = collect_img[i][0]
        im, flip_im = preprocessImage(img)
        embedding = imgToEmbedding(im, model, img_flip=flip_im)
        embeddings.append(embedding)

    similarity_metrics = cosine_similarity(embeddings)

    duplicate_image_idx = set()

    for i in range(len(similarity_metrics[0])):
        if i in duplicate_image_idx:
            continue
        for j in range(i+1, len(similarity_metrics[1])):
            if similarity_metrics[i][j] >= 0.8:
                duplicate_image_idx.add(j)
    new_collect_img = []

    for i in range(len(collect_img)):
        if i in duplicate_image_idx:
            continue
        new_collect_img.append(collect_img[i])
    return new_collect_img

def selectNearImageAndSave(filter_collect_img, number):
    select_img = filter_collect_img.copy()
    if len(select_img) > 60:
      select_img.sort(key=lambda x: x[1])
      select_img = select_img[:60]
    for idx, img in enumerate(select_img):
      directory = "../our_children_changed/dataset/{}".format(number)
      if not os.path.exists(directory):
        os.makedirs(directory)
      plt.imsave(directory + "/{}.jpg".format(idx), img[0])


def collectDatasetPipeline(root_path, page, number, model):
    #seed
    logging.basicConfig(filename=root_path + "/log/page{}.log".format(page), format="%(asctime)s %(levelname)s %(message)s")
    try:
        seed = createEmbedingSeed(root_path + "/seed", str(number), model, img_show=True)
        print("시드 라벨 개수: ", len(seed['labels']))
        #filter#1
        collect_img = collectFaceImageWithSeed(root_path + "/video/page{}/{}.mp4".format(page, number), model, seed, 1)
        print("1차 필터링 완료")
        for i in range(len(collect_img)):
            print("{}번: {}개".format(i,len(collect_img[i])))
        #filter#2 and filter#3,
        for i in range(len(collect_img)):
            filter_collect_img = filterCosineSimilarity(collect_img[i], model)
            print("{}번 2차필터링 완료: {}개".format(i ,len(filter_collect_img)) )
            #save
            selectNearImageAndSave(filter_collect_img, number+i)
            print("{}번 3차 필터링 완료 저장 끝".format(i))
    except:
        logging.error(traceback.format_exc())
        traceback.print_exc()