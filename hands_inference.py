import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from flask import Flask, render_template, Response
import os
import urllib.request

app = Flask(__name__)

# Conexões das mãos (índices dos landmarks conectados)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),      # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12), # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # Mindinho
    (5, 9), (9, 13), (13, 17)            # Conexões na palma
]

def draw_hand_landmarks(image, hand_landmarks, connections, landmark_color=(0, 255, 0), 
                        connection_color=(0, 255, 255), thickness=2, circle_radius=2):
    """Desenha os landmarks e conexões das mãos na imagem"""
    h, w, _ = image.shape
    
    # Converter landmarks para coordenadas de pixel
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
    
    # Desenhar conexões
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], connection_color, thickness)
    
    # Desenhar pontos dos landmarks
    for point in points:
        cv2.circle(image, point, circle_radius, landmark_color, -1)

class HandGestureDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Dicionário de tradução dos gestos para português
        self.gesture_translations = {
            'Closed_Fist': 'Rock and Roll',
            'Open_Palm': 'Tchau! Tchau!',
            'Pointing_Up': 'Obrigado senhor!',
            'Thumb_Down': 'Eliminado',
            'Thumb_Up': 'Aprovado',
            'Victory': 'Paz e amor',
            'ILoveYou': 'Eu Te Amo'
        }
        
        # Dicionário de tradução para handedness (mão esquerda/direita)
        # Como a imagem é espelhada, invertemos a lógica para ficar correto do ponto de vista do usuário
        self.handedness_translations = {
            'Left': 'Direita',   # Inverte porque a imagem é espelhada
            'Right': 'Esquerda'  # Inverte porque a imagem é espelhada
        }
        
        # Download dos modelos se não existirem
        self._download_models_if_needed()
        
        # Configurar Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=4,  # Detectar até 4 mãos
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        # Configurar Gesture Recognizer
        gesture_base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
        gesture_options = vision.GestureRecognizerOptions(
            base_options=gesture_base_options,
            num_hands=6,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.gesture_detector = vision.GestureRecognizer.create_from_options(gesture_options)
        print("✅ MediaPipe Hand Landmarker e Gesture Recognizer inicializados!")
    
    def _download_models_if_needed(self):
        """Download dos modelos hand_landmarker e gesture_recognizer se não existirem"""
        models = [
            {
                'path': 'hand_landmarker.task',
                'name': 'Hand Landmarker',
                'url': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            },
            {
                'path': 'gesture_recognizer.task',
                'name': 'Gesture Recognizer',
                'url': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
            }
        ]
        
        for model in models:
            if not os.path.exists(model['path']):
                print(f"📥 Baixando modelo {model['name']}...")
                try:
                    urllib.request.urlretrieve(model['url'], model['path'])
                    print(f"✅ Modelo {model['name']} baixado com sucesso!")
                except Exception as e:
                    print(f"❌ Erro ao baixar modelo {model['name']}: {e}")
                    raise Exception(f"Não foi possível baixar o modelo {model['name']}")
    
    def detect_hands_and_gestures(self, frame):
        """Detecta mãos e gestos na imagem e desenha os landmarks"""
        # Converter frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detectar mãos
        hand_result = self.hand_detector.detect(mp_image)
        
        # Detectar gestos
        gesture_result = self.gesture_detector.recognize(mp_image)
        
        hand_count = 0
        gestures_detected = []
        
        # Processar landmarks das mãos
        if hand_result.hand_landmarks:
            hand_count = len(hand_result.hand_landmarks)
            
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                # Determinar se é mão esquerda ou direita
                handedness = hand_result.handedness[idx][0]
                hand_label_en = handedness.category_name
                hand_score = handedness.score
                # Traduzir para português
                hand_label = self.handedness_translations.get(hand_label_en, hand_label_en)
                
                # Desenhar landmarks das mãos
                draw_hand_landmarks(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    landmark_color=(0, 255, 0),
                    connection_color=(0, 255, 255),
                    thickness=2,
                    circle_radius=2
                )
                
                # Adicionar label da mão
                h, w, c = frame.shape
                cx = int(hand_landmarks[0].x * w)
                cy = int(hand_landmarks[0].y * h)
                cv2.putText(frame, f'{hand_label} ({hand_score:.2f})', 
                           (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Processar gestos detectados
        if gesture_result.gestures:
            for idx, gesture_list in enumerate(gesture_result.gestures):
                if gesture_list:
                    gesture = gesture_list[0]
                    gesture_name_en = gesture.category_name
                    gesture_score = gesture.score
                    # Traduzir o nome do gesto para português
                    gesture_name_pt = self.gesture_translations.get(gesture_name_en, gesture_name_en)
                    gestures_detected.append(f"{gesture_name_pt} ({gesture_score:.2f})")
        
        # Adicionar informações na tela
        cv2.putText(frame, f'Maos detectadas: {hand_count}/4', 
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar gestos detectados
        y_offset = 110
        for gesture in gestures_detected:
            cv2.putText(frame, f'Gesto: {gesture}', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def generate_frames(self):
        """Gera frames para streaming"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip horizontal para efeito espelho
            frame = cv2.flip(frame, 1)
            
            # Detectar mãos e gestos
            frame = self.detect_hands_and_gestures(frame)
            
            # Codificar frame para JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

# Instância global do detector
detector = None

def get_detector():
    """Inicializa o detector de mãos e gestos sob demanda"""
    global detector
    if detector is None:
        detector = HandGestureDetector()
    return detector

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Feed de vídeo"""
    return Response(get_detector().generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Iniciando Hand Landmarker e Gesture Recognition - CIAg")
    print("Detecção de até 6 mãos simultâneas")
    print("Reconhecimento de gestos em PORTUGUÊS")
    print("Acesse: http://localhost:8888")
    
    app.run(debug=True, host='0.0.0.0', port=8888)
