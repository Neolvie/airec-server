#!/bin/sh
# Скачивает модели диаризации sherpa-onnx в ./models
set -e
cd "$(dirname "$0")/.."
mkdir -p models
cd models

if [ ! -f sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  echo "Скачиваю pyannote segmentation..."
  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
fi

if [ ! -f 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
  echo "Скачиваю 3dspeaker embedding..."
  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
fi

echo "Модели готовы: $(pwd)"
