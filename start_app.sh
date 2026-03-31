#!/bin/bash

echo "Iniciando Hand Landmarker & Gesture Recognition - CIAg"
echo "======================================================"

# Ativar ambiente virtual
if [ -d "venv" ]; then
    echo "Ativando ambiente virtual..."
    source venv/bin/activate
fi

# Verificar dependências
python3 -c "import cv2, mediapipe, flask, numpy; print('✅ Dependências OK!')" 2>/dev/null || {
    echo "Instalando dependências..."
    pip install opencv-python mediapipe flask numpy
}

# Executar aplicação
echo "Iniciando servidor..."
echo "✋ Detecção de mãos: até 4 mãos simultâneas"
echo "🤝 Reconhecimento de gestos em tempo real"
echo "Acesse: http://localhost:8888"
echo "Pressione Ctrl+C para parar"
echo "======================================================"

python3 hands_inference.py