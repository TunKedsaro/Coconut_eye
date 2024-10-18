import cv2
import onnxruntime as ort
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as trans
from keras.models import load_model
import pickle


class FaceModel:
    def __init__(self, embedding_model_path):
        self.session = ort.InferenceSession(embedding_model_path)

    def preprocess_image(self, img_input):
        # Check if the input is a file path (string) or a NumPy array (image)
        if isinstance(img_input, str):  # If it's a file path
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Image not found at path: {img_input}")
        elif isinstance(img_input, np.ndarray):  # If it's an image (NumPy array)
            img = img_input
        else:
            raise TypeError("Input must be a file path (str) or an image (np.ndarray)")

        # Resize the image to (112, 112)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def get_embedding(self, img):
        inputs = self.session.get_inputs()
        outputs = self.session.run(None, {inputs[0].name: img})
        embedding = outputs[0]
        return embedding[0]
class FacePreprocessor:
    def __init__(self,image_size='112,112',margin=44):
        self.image_size = [int(x) for x in image_size.split(',')]
        if len(self.image_size) == 1:
            self.image_size = [self.image_size[0],self.image_size[0]]
        self.margin = margin
        assert len(self.image_size) == 2
        assert self.image_size[0] == 112 and (self.image_size[1] == 112 or self.image_size[1] == 96)
    def read_image(self,img_path,mode='rgb',layout='HWC'):
        if mode == 'gray': # gray -> gray
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if mode == 'rgb':
                img = img[..., ::-1]
            if layout == 'CHW':
                img = np.transpose(img,(2,0,1))
        return img
    def preprocess(self, img, bbox=None, landmark=None):
        if isinstance(img, str):
            img = self.read_image(img)

        M = None
        if landmark is not None:
            assert len(self.image_size) == 2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)

            if self.image_size[1] == 112:
                src[:, 0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]

        if M is None:
            return self._center_crop(img, bbox)
        else:
            return self._warp_image(img, M)

    def _center_crop(self, img, bbox):
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - self.margin // 2, 0)
        bb[1] = np.maximum(det[1] - self.margin // 2, 0)
        bb[2] = np.minimum(det[2] + self.margin // 2, img.shape[1])
        bb[3] = np.minimum(det[3] + self.margin // 2, img.shape[0])

        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(self.image_size) > 0:
            ret = cv2.resize(ret, (self.image_size[1], self.image_size[0]))
        return ret

    def _warp_image(self, img, M):
        assert len(self.image_size) == 2
        warped = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        return warped

def findCosineDistance(vector1, vector2):
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T,vec2)
    b = np.dot(vec1.T,vec1)
    c = np.dot(vec2.T,vec2)

    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))
def CosineSimilarity(test_vec, source_vecs):
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

embedding_model_path = r"C:\Users\Acer\Desktop\FaceRecAPI\models\w600k_r50.onnx"
face_model = FaceModel(embedding_model_path)
detector = MTCNN()
preprocessor = FacePreprocessor(image_size='112,112',margin=44)
# Load the classifier model
mymodel = r"C:\Users\Acer\Desktop\FaceRecAPI\models\normal_model.h5"
model = load_model(mymodel)
# Load label
embeddings = r"C:\Users\Acer\Desktop\FaceRecAPI\models\embeddings_datasets.pickle"
le = r"C:\Users\Acer\Desktop\FaceRecAPI\models\le.pickle"
with open(embeddings,"rb") as f:
    data = pickle.load(f)
with open(le,"rb") as f:
    le = pickle.load(f)
embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])         # แทนชื่อคนด้วยตัวเลข
print(labels)

def fn(img, cosine_threshold = 0.95, proba_threshold = 0.85, comparing_num = 5):
    # input -> RGB
    # img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Rescale size with factor
    h,w = img.shape[:2]
    Fct = 1000/h
    Nh  = int(h*Fct)
    Nw  = int(w*Fct)
    img = cv2.resize(img, (Nw,Nh))
    # Face detection
    detector = MTCNN()
    bboxes   = detector.detect_faces(img)
    # # Original img
    # plt.figure(figsize=(10,7))
    # plt.imshow(img)
    # plt.show()
    try:
        # Switching
        if len(bboxes) == 0:
            text = "0"
        elif len(bboxes) > 0:
            if len(bboxes) == 1:              # case I  : Single face
                biggest_face = bboxes[0]
            elif len(bboxes) > 1:             # case II : multiple face
                max_area = 0
                for face in bboxes:
                    x, y, width, height = face['box']
                    area = width*height
                    if area > max_area:
                        max_area = area
                        biggest_face = face
            # print(biggest_face)
            bbox = biggest_face['box']
            bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            landmarks = biggest_face['keypoints']
            landmarks = np.array([landmarks["left_eye"][0],landmarks["right_eye"][0],landmarks["nose"][0],landmarks["mouth_left"][0],landmarks["mouth_right"][0],landmarks["left_eye"][1],landmarks["right_eye"][1],landmarks["nose"][1],landmarks["mouth_left"][1],landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2,5)).T

            nimg = preprocessor.preprocess(img,bbox,landmarks)                      # ได้หน้าของแต่ละคน
            # plt.imshow(nimg)
            # plt.show()

            prep_img = face_model.preprocess_image(nimg)
            embedding = face_model.get_embedding(prep_img).reshape(1,-1)
            # Class predictive
            text = "?"
            preds = model.predict(embedding)                                        # [[9.9969018e-01 3.0968967e-04 1.1600605e-07]]
            preds = preds.flatten()                                                 # [9.9969018e-01 3.0968967e-04 1.1600605e-07]
            j = np.argmax(preds)                                                            # 0,1,2 class ที่มากสุด
            proba = preds[j]                                                                # เอาเปอร์เซ้นของตัวที่มากที่สุดมา
            # similarity
            match_class_idx = np.where(labels == j)[0]
            selected_idx = np.random.choice(match_class_idx, 20)
            # selected_idx
            compare_embeddings = embeddings[selected_idx]
            # compare_embeddings
            cos_similarity = CosineSimilarity(embedding, compare_embeddings)
            print("if cos_similarity < cosine_threshold and proba > proba_threshold:")
            print(f"if {cos_similarity} < {cosine_threshold} and {proba} > {proba_threshold}:")
            # if cos_similarity < cosine_threshold and proba > proba_threshold:
            if cos_similarity < cosine_threshold:
                name = le.classes_[j]
                text = f"{name}"
                print(f"Recognized: {name} <{proba*100:.2f}>")
            # y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
            # cv2.putText(img, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
            # # break
            # plt.figure(figsize=(15,10))
            # plt.imshow(img)
            # plt.show()

    except:
        text = "Something Error"
    return  text, img.shape, len(bboxes)

# fn("/content/drive/MyDrive/coconut/Face_recognition_project/Test_dataset/SC651Ice/225.png")

# img_name, predict_name, img_shape, len_boxes = fn(r"C:\Users\Acer\Desktop\FaceRecAPI\test_images\001.png")
# img_name, predict_name, img_shape, len_boxes



def test():
    print("Yep")
