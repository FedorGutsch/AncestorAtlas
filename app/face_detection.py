from realutils.face.insightface import (
    isf_analysis_faces,
    isf_face_batch_similarity,
    isf_faces_visualize
)
from PIL import Image
import io
import base64
import numpy as np


def analyze_photo(image_path: str):
    """
    Полный анализ фото через realutils:
    1. Детекция всех лиц
    2. Сравнение лиц между собой на одном фото
    3. Визуализация результатов
    """
    # Анализируем все лица на фото
    faces = isf_analysis_faces(image_path)
    
    if not faces:
        return {
            "faces_count": 0,
            "faces": [],
            "similarity_matrix": None,
            "visualization": None,
            "message": "Лица не найдены"
        }
    
    # Сравниваем лица между собой (если >1 лицо)
    similarity_matrix = None
    if len(faces) > 1:
        embeddings = [face.embedding for face in faces]
        similarity_matrix = isf_face_batch_similarity(embeddings).tolist()
    
    # Визуализируем результаты
    vis_image = isf_faces_visualize(image_path, faces)
    
    # Конвертируем в base64 для отображения в браузере
    buffered = io.BytesIO()
    if isinstance(vis_image, np.ndarray):
        # Если возвращается numpy array (OpenCV формат)
        import cv2
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        vis_image = Image.fromarray(vis_image)
    
    if vis_image.mode == 'RGBA':
        vis_image = vis_image.convert('RGB')
    
    vis_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Формируем данные о лицах
    faces_data = []
    for i, face in enumerate(faces):
        bbox = getattr(face, 'bbox', None)
        kps = getattr(face, 'kps', None)
        det_score = getattr(face, 'det_score', None)
        
        faces_data.append({
            "id": i + 1,
            "bbox": list(bbox) if bbox is not None else None,
            "det_score": float(det_score) if det_score is not None else None,
            "embedding_shape": face.embedding.shape if hasattr(face, 'embedding') else None,
            "embedding_sample": face.embedding[:5].tolist() if hasattr(face, 'embedding') else None
        })
    
    return {
        "faces_count": len(faces),
        "faces": faces_data,
        "similarity_matrix": similarity_matrix,
        "visualization": img_str,
        "message": f"Найдено {len(faces)} лиц(а)"
    }