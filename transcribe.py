import whisper
import sys
import os
import torch
from datetime import timedelta

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def transcribe_with_timestamps(audio_path, output_txt="transcription.txt", model_size="base", language="ko"):
    if not os.path.exists(audio_path):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_path}", flush=True)
        sys.exit(1)

    print("모델 로딩 중...", flush=True)
    model = whisper.load_model(model_size)
    print("모델 로딩 완료 ✅", flush=True)

    print("텍스트 변환 중 (시간 포함)...", flush=True)
    result = model.transcribe(audio_path, language=language, verbose=False)

    print(f"변환 완료! 저장 중: {output_txt}", flush=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"[{start} - {end}]\n{text}\n\n")

    print("✅ 저장 완료:", output_txt, flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python transcribe.py <오디오 파일 경로> [출력 파일명] [모델 사이즈] [언어코드]", flush=True)
        sys.exit(1)

    audio_file = sys.argv[1]
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "transcription.txt"
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"
    language = sys.argv[4] if len(sys.argv) > 4 else "en"

    if torch.cuda.is_available():
        print("🎉 GPU 사용 중:", torch.cuda.get_device_name(0), flush=True)
        print("CUDA 버전:", torch.version.cuda, flush=True)
    else:
        print("⚠️ GPU 사용 불가 (CPU로 실행됨)", flush=True)

    transcribe_with_timestamps(audio_file, output_txt, model_size, language)


"""
<사용법>
python transcribe.py "mp3 파일 명" "txt 파일 명" 모델종류 언어

1. .\venv\Scripts\Activate.ps1
2. (예시) python transcribe.py "interior4-4.mp3" "test.txt" small en
"""