from deepface import DeepFace
from pathlib import Path

# Given a path to an attack image, finds the original image and returns the distance between them.
# Also includes optional parameters matching those of DeepFace.verify, which default to the default parameters of that function
def distance_to_original_img(atk_path, model_name = 'VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):
    if not atk_path.exists():
        print("invalid path to attack image")
    file_name = Path(atk_path).name # filename of attack image, which should also be the filename of the original image
    face_name = file_name[:-9] # name of the person in the file, obtained by removing "_<idx>.png" from the filename
    org_path = 'data/lfw-py/lfw_funneled/' + face_name + '/' + file_name # path to the original image
    if not org_path.exists():
        print("invalid path to original image")
    obj = DeepFace.verify(atk_path, org_path, model_name, distance_metric, model, enforce_detection, detector_backend, align, prog_bar, normalization)
    return obj['distance']
