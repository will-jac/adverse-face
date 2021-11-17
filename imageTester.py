import DeepFace

def test_attack(atk_img, original_img, output):
    backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
    models = ["Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
    face_detected = []
    for backend in backends:
        try:
            detected_face = DeepFace.detectFace(atk_img, detector_backend = backend)
            face_detected.append(1)
            if output:
                print(backend, "did detect a face")
        except:
            face_detected.append(0)
            if output:
                print(backend, "did not detect a face")
    distances = []
    for model in models:
        distances.append([])
    i = 0
    for model in models:
        worst_distance = 1000
        worst_backend = ""
        for backend in backends:
            obj = DeepFace.verify(atk_img, original_img, model_name = model, detector_backend = backend)
            distances[i].append(obj['distance'])
            if(obj['distance'] < worst_distance):
                worst_distance = obj['distance']
                worst_backend = backend
        if output:
            print("The closest fit for", model, "was", worst_distance, "distance using", backend, "as a backend")
        i += 1
    return(face_detected, distances)
