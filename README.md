# Hand Landmarker & Gesture Recognition - CIAg

Uma aplicação web em tempo real para detecção de mãos e reconhecimento de gestos usando **MediaPipe Hand Landmarker** e **Gesture Recognizer** com **Flask**, desenvolvida para a aula de Computer Vision do Synapse e CIAg.



![Hand Landmarker Demo](https://img.shields.io/badge/Status-Funcionando-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)
![Flask](https://img.shields.io/badge/Flask-3.0+-red)

## Funcionalidades

- **Detecção de múltiplas mãos**: Até **6 mãos simultaneamente**
- **Reconhecimento de gestos em tempo real**: Usando MediaPipe Gesture Recognizer
- **Hand landmarks**: Detecção precisa de 21 pontos em cada mão
- **Classificação de mão**: Identifica mão esquerda/direita
- **FPS counter**: Monitoramento de performance em tempo real
- **Controles por teclado**: Atalhos para melhor experiência
- **Interface web responsiva**: Funciona em desktop e mobile


## Tecnologias Utilizadas

- **Backend**: Python 3.7+, Flask, MediaPipe Tasks
- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Computer Vision**: 
  - MediaPipe Hand Landmarker
  - MediaPipe Gesture Recognizer
- **Streaming**: MJPEG over HTTP

## Pré-requisitos

- Python 3.7 ou superior
- Webcam funcionando
- Navegador web moderno (Chrome, Firefox, Safari, Edge)
- Sistema operacional: Linux, macOS, Windows

## Instalação

### 1. Clone ou baixe o projeto

```bash
git clone [URL_DO_PROJETO]
cd AULA5-HandLandmarker
```

### 2. Instale as dependências:

```bash
pip install opencv-python mediapipe flask numpy
```

### 3. Verifique a estrutura do projeto:

```
AULA5-HandLandmarker/
├── pose_inference.py      # Aplicação principal
├── start_app.sh          # Script de inicialização
├── requirements.txt      # Dependências
├── templates/
│   └── index.html       # Interface web
├── hand_landmarker.task    # Modelo (baixado automaticamente)
└── gesture_recognizer.task # Modelo (baixado automaticamente)
```

## Como Usar

### Método 1: Script de inicialização (Recomendado)

```bash
./start_app.sh
```

### Método 2: Execução direta

```bash
python pose_inference.py
```

### Método 3: Com ambiente virtual

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
python3 pose_inference.py
```

## Funcionalidades Detalhadas

### Hand Landmarker
- Detecta até 6 mãos simultaneamente
- 21 pontos de landmark por mão
- Classificação automática: mão esquerda/direita
- Confidence score para cada detecção

### Gesture Recognizer
- Reconhece gestos padrão do MediaPipe
- Gestos suportados: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
- Score de confiança para cada gesto
- Detecção em tempo real

### Controles
- **F**: Alternar tela cheia
- **R**: Recarregar página
- **ESC**: Sair da tela cheia

### Interface
- Stream de vídeo em tempo real
- Contador de FPS
- Indicadores visuais das mãos detectadas
- Labels dos gestos reconhecidos

