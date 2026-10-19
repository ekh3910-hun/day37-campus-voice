
# day37_practice.py
# 캠퍼스 시설 안내 음성 도우미
# 녹음 → STT → 시설 안내 응답 → TTS → 재생 → 실행 이력 저장
import os
import psycopg
from dotenv import load_dotenv
import asyncio
import io
import json
import math
import struct
import tempfile
import time
import traceback
import uuid
import wave

from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. 기본 설정
# ==================================================

MAX_BYTES = 10 * 1024 * 1024

LOG_PATH = Path(__file__).with_name("voice_runs_campus.jsonl")

MODEL_SIZE = "base"

VOICE_ID = "ko-KR-SunHiNeural"

load_dotenv(Path(__file__).with_name(".env"), override=True)


# 연습용 캠퍼스 시설 정보
FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}


# ==================================================
# 2. WAV 파일 검사
# ==================================================

def validate_wav(audio):

    if not audio:
        raise ValueError("녹음된 음성이 없습니다.")

    if len(audio) > MAX_BYTES:
        raise ValueError("음성 파일은 10MB 이하만 가능합니다.")

    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:

            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype()

    except (wave.Error, EOFError, OSError) as e:
        raise ValueError("올바른 WAV 파일이 아닙니다.") from e

    if (
        compression != "NONE"
        or channels not in (1, 2)
        or sample_rate <= 0
        or sample_width not in (1, 2, 3, 4)
        or frames <= 0
    ):
        raise ValueError("지원하지 않는 WAV 형식입니다.")

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "bit_depth": sample_width * 8,
        "duration_sec": round(frames / sample_rate, 2),
    }


# ==================================================
# 3. Whisper STT 모델 로딩
# ==================================================

@st.cache_resource(show_spinner="Whisper 모델 로딩 중...")
def load_stt_model(model_size):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


# ==================================================
# 4. 실제 STT
# PyAV 호환성 오류 우회 버전
# ==================================================

def transcribe_wav(audio, model_size):

    model = load_stt_model(model_size)

    # 임시 WAV 파일 생성
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as f:

        f.write(audio)

        wav_path = Path(f.name)

    try:

        # PyAV로 WAV 파일 직접 열기
        # metadata_errors 인자를 사용하지 않음
        with av.open(str(wav_path), mode="r") as container:

            audio_stream = container.streams.audio[0]

            # Whisper 입력 규격으로 변환
            # 16kHz / mono / signed 16-bit PCM
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=16000,
            )

            samples = []

            # 음성 프레임 디코딩
            for frame in container.decode(audio_stream):

                resampled_frames = resampler.resample(frame)

                for resampled in resampled_frames:

                    array = resampled.to_ndarray().flatten()

                    samples.append(array)

            # 리샘플러에 남은 데이터 처리
            for resampled in resampler.resample(None):

                array = resampled.to_ndarray().flatten()

                samples.append(array)

        if not samples:
            raise ValueError("음성 데이터를 읽지 못했습니다.")

        # NumPy 배열 결합 및 float32 변환
        audio_array = np.concatenate(samples)

        audio_array = (
            audio_array.astype(np.float32) / 32768.0
        )

        # 파일 경로 대신 NumPy 배열을 전달
        segments, info = model.transcribe(
            audio_array,
            language="ko",
            vad_filter=True,
        )

        # 구간별 인식 결과 결합
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return transcript

    finally:

        # 성공/실패와 관계없이 임시 파일 삭제
        wav_path.unlink(missing_ok=True)


# ==================================================
# 5. 데모 STT
# ==================================================

def demo_stt(hint):

    return hint.strip()


# ==================================================
# 6. 캠퍼스 시설 안내 응답 생성
# ==================================================

def build_reply(transcript):

    clean = " ".join(transcript.split())

    if not clean:
        return "음성 인식을 하지 못했습니다. 다시 말씀해 주세요."

    # 질문에 시설명이 포함되어 있는지 확인
    for facility, description in FACILITIES.items():

        if facility in clean:
            return description

    # 시설명을 찾지 못한 경우
    names = ", ".join(FACILITIES.keys())

    return (
        f"문의하신 시설을 찾지 못했습니다. "
        f"안내 가능한 시설은 {names}입니다."
    )


# ==================================================
# 7. 실제 TTS - Edge TTS
# ==================================================

async def generate_tts_mp3(text, voice_id):

    import edge_tts

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    ) as f:

        mp3_path = Path(f.name)

    try:

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_id,
        )

        await communicate.save(str(mp3_path))

        return mp3_path.read_bytes()

    finally:

        mp3_path.unlink(missing_ok=True)


# ==================================================
# 8. 데모 TTS - 비프음 생성
# ==================================================

def generate_beep():

    sample_rate = 16000
    duration = 0.4
    frequency = 660
    amplitude = 10000

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        samples = bytearray()

        for i in range(int(sample_rate * duration)):

            value = int(
                amplitude
                * math.sin(
                    2 * math.pi * frequency * i / sample_rate
                )
            )

            samples.extend(
                struct.pack("<h", value)
            )

        wav.writeframes(bytes(samples))

    return buffer.getvalue()


# ==================================================
# 9. TTS 실행 함수
# ==================================================

def synthesize_speech(text, tts_mode, voice_id):

    # 데모 TTS
    if tts_mode == "데모 TTS":

        return generate_beep(), "audio/wav"

    # 실제 TTS
    mp3_bytes = asyncio.run(
        generate_tts_mp3(text, voice_id)
    )

    return mp3_bytes, "audio/mpeg"


# ==================================================
# 10. JSONL 실행 이력 저장
# ==================================================

def store_meta(meta):

    with LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                meta,
                ensure_ascii=False,
            ) + "\n"
        )
def store_meta_db(meta):

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다.")

    with psycopg.connect(
        database_url,
        connect_timeout=10
    ) as conn:

        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO public.campus_voice_runs (
                    request_id,
                    created_at,
                    status,
                    error_type,
                    stt_mode,
                    tts_mode,
                    model_version,
                    stt_ms,
                    reply_ms,
                    tts_ms,
                    total_ms
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
            """, (
                meta["request_id"],
                meta["created_at"],
                meta["status"],
                meta["error_type"],
                meta["stt_mode"],
                meta["tts_mode"],
                meta["model_version"],
                meta["stt_ms"],
                meta["reply_ms"],
                meta["tts_ms"],
                meta["total_ms"]
            ))

# ==================================================
# 11. 최근 실행 이력 조회
# ==================================================

def load_history(limit=20):

    if not LOG_PATH.exists():
        return []

    rows = []

    with LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return rows[-limit:][::-1]


# ==================================================
# 12. 전체 음성 처리 파이프라인
# ==================================================

def run_pipeline(
    audio,
    stt_mode,
    tts_mode,
    demo_text,
    model_size,
    voice_id,
):

    # 요청 고유 ID
    request_id = str(uuid.uuid4())

    # UTC 시각
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    # 단계별 처리시간
    stt_ms = 0.0
    reply_ms = 0.0
    tts_ms = 0.0

    transcript = ""
    reply = ""

    audio_result = None
    mime_type = None
    wav_info = None

    status = "success"
    error_type = ""
    error_detail = ""

    # 전체 처리 시작 시간
    total_start = time.perf_counter()

    try:

        # ------------------------------------------
        # 1. WAV 검증
        # ------------------------------------------

        wav_info = validate_wav(audio)

        # ------------------------------------------
        # 2. STT
        # ------------------------------------------

        stt_start = time.perf_counter()

        try:

            if stt_mode == "데모 STT":

                transcript = demo_stt(demo_text)

            else:

                transcript = transcribe_wav(
                    audio,
                    model_size,
                )

        finally:

            stt_ms = (
                time.perf_counter() - stt_start
            ) * 1000

        # ------------------------------------------
        # 3. 응답 생성
        # ------------------------------------------

        reply_start = time.perf_counter()

        try:

            reply = build_reply(transcript)

        finally:

            reply_ms = (
                time.perf_counter() - reply_start
            ) * 1000

        # ------------------------------------------
        # 4. TTS
        # ------------------------------------------

        tts_start = time.perf_counter()

        try:

            audio_result, mime_type = synthesize_speech(
                reply,
                tts_mode,
                voice_id,
            )

        finally:

            tts_ms = (
                time.perf_counter() - tts_start
            ) * 1000

    except Exception as e:

        status = "error"

        error_type = type(e).__name__
        error_detail = str(e)

        st.error(
            f"처리 실패: {error_type}: {e}"
        )

        # 오류 상세 정보
        st.code(traceback.format_exc())

    finally:

        # ------------------------------------------
        # 5. 전체 처리시간
        # ------------------------------------------

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        # 개인정보 및 음성 원본은 저장하지 않음
        meta = {
            "request_id": request_id,
            "created_at": created_at,
            "status": status,
            "error_type": error_type,
            "stt_mode": stt_mode,
            "tts_mode": tts_mode,
            "model_version": (
                model_size
                if stt_mode == "실제 STT"
                else "demo"
            ),
            "stt_ms": round(stt_ms, 2),
            "reply_ms": round(reply_ms, 2),
            "tts_ms": round(tts_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        try:
            store_meta(meta)
        except OSError as e:
            st.warning(f"JSONL 저장 실패: {e}")

        try:
            store_meta_db(meta)
        except Exception as e:
            st.warning(f"Neon DB 저장 실패: {type(e).__name__}: {e}")
    return {
        "transcript": transcript,
        "reply": reply,
        "audio_result": audio_result,
        "mime_type": mime_type,
        "wav_info": wav_info,
        "meta": meta,
        "error_detail": error_detail,
    }


# ==================================================
# 13. Streamlit 화면
# ==================================================

def main():

    st.set_page_config(
        page_title="캠퍼스 시설 안내 음성 도우미",
        page_icon="🎤",
        layout="wide",
    )

    st.title("🎤 캠퍼스 시설 안내 음성 도우미")

    st.write(
        "음성으로 캠퍼스 시설을 문의하고 "
        "안내를 받아보세요."
    )

    st.info(
        "시설 위치와 운영시간은 실습용 예시 정보입니다."
    )

    # ------------------------------------------
    # 시설 정보
    # ------------------------------------------

    with st.expander("안내 가능한 시설"):

        for name, description in FACILITIES.items():

            st.write(
                f"**{name}**: {description}"
            )

    st.divider()

    # ------------------------------------------
    # 모델 선택
    # ------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        stt_mode = st.selectbox(
            "STT 모델",
            ["데모 STT", "실제 STT"],
        )

    with col2:

        tts_mode = st.selectbox(
            "TTS 모델",
            ["데모 TTS", "실제 TTS"],
        )

    # ------------------------------------------
    # 데모 질문 선택
    # ------------------------------------------

    demo_text = ""

    if stt_mode == "데모 STT":

        demo_text = st.selectbox(
            "데모 질문 선택",
            [
                "도서관 운영시간 알려줘",
                "학생식당 어디야?",
                "취업지원센터 이용하고 싶어",
                "행정실 위치 알려줘",
                "운동장에 대해서 알려줘",
            ],
        )

    st.divider()

    # ------------------------------------------
    # 마이크 녹음
    # ------------------------------------------

    audio_file = st.audio_input(
        "마이크로 음성을 녹음하세요"
    )

    audio_bytes = None

    if audio_file is not None:

        audio_bytes = audio_file.getvalue()

        st.audio(
            audio_bytes,
            format="audio/wav",
        )

        st.caption(
            f"녹음 파일 크기: {len(audio_bytes):,} bytes"
        )

    # ------------------------------------------
    # 음성 안내 실행
    # ------------------------------------------

    if st.button(
        "음성 안내 실행",
        type="primary",
        use_container_width=True,
    ):

        if audio_bytes is None:

            st.error(
                "먼저 음성을 녹음해 주세요."
            )

        elif len(audio_bytes) > MAX_BYTES:

            st.error(
                "음성 파일은 10MB 이하만 가능합니다."
            )

        else:

            with st.spinner("음성 AI 처리 중..."):

                result = run_pipeline(
                    audio=audio_bytes,
                    stt_mode=stt_mode,
                    tts_mode=tts_mode,
                    demo_text=demo_text,
                    model_size=MODEL_SIZE,
                    voice_id=VOICE_ID,
                )

            st.session_state["last_result"] = result

    # ------------------------------------------
    # 결과 표시
    # ------------------------------------------

    result = st.session_state.get("last_result")

    if result is not None:

        st.divider()

        st.subheader("처리 결과")

        meta = result["meta"]

        if meta["status"] == "success":

            st.success("음성 처리 완료")

            st.write(
                "**STT 인식 결과:**",
                result["transcript"],
            )

            st.write(
                "**시설 안내 답변:**",
                result["reply"],
            )

            if result["audio_result"] is not None:

                st.audio(
                    result["audio_result"],
                    format=result["mime_type"],
                )

        else:

            st.error(
                f"처리 실패: {meta['error_type']}"
            )

        # ------------------------------------------
        # 처리시간 표시
        # ------------------------------------------

        st.subheader("단계별 처리 시간")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "STT",
            f"{meta['stt_ms']:.2f} ms",
        )

        col2.metric(
            "응답",
            f"{meta['reply_ms']:.2f} ms",
        )

        col3.metric(
            "TTS",
            f"{meta['tts_ms']:.2f} ms",
        )

        col4.metric(
            "전체",
            f"{meta['total_ms']:.2f} ms",
        )

        # ------------------------------------------
        # 막대그래프
        # ------------------------------------------

        chart_df = pd.DataFrame(
            {
                "단계": [
                    "STT",
                    "응답 생성",
                    "TTS",
                ],
                "처리시간(ms)": [
                    meta["stt_ms"],
                    meta["reply_ms"],
                    meta["tts_ms"],
                ],
            }
        )

        st.bar_chart(
            chart_df.set_index("단계")
        )

    # ------------------------------------------
    # 최근 실행 이력
    # ------------------------------------------

    st.divider()

    st.subheader("최근 실행 이력")

    history = load_history(limit=20)

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "아직 실행 이력이 없습니다."
        )


# ==================================================
# 14. 프로그램 실행
# ==================================================

if __name__ == "__main__":
    main()

