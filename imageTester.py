def test_attack(atk_img, original_img):
    backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
    models = ["Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
    face_detected = []
    for backend in backends:
        try:
            detected_face = DeepFace.detectFace(atk_img, detector_backend = backend)
            face_detected.append(1)
        except:
            face_detected.append(0)
    distances = []
    for model in models:
        distances.append([])
    i = 0
    for model in models:
        for backend in backends:
            obj = DeepFace.verify(atk_img, original_img, model_name = model, detector_backend = backend)
            distances[i].append(obj['distance'])
        i += 1
    return(face_detected, distances)