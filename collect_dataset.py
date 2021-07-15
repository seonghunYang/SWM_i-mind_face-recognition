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
from utils.utils import checkExtenstion 

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
        faces = RetinaFace.detect_faces(img_path=img)
    # bbox
        if (type(faces) == dict):
            alignment_face_imgs = alignImage(img, faces)
        # embedding
            for face_img in alignment_face_imgs:
                process_face_img, process_face_img_flip = preprocessImage(face_img)
                embedding = imgToEmbedding(process_face_img, model, img_flip=process_face_img_flip)
                seed["embedding"].append(embedding)
                seed['labels'].append(label)
                labels.append(label)
            if img_show:
                img_bbox = drawFrameWithBbox(img, faces, labels)
                plt.imshow(img_bbox)
                plt.show()
    seed['embedding'] = normalize(seed['embedding'])
    return seed