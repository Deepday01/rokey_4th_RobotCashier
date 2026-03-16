import asyncio
import edge_tts
import os
import tempfile

VOICE = "ko-KR-SunHiNeural"

async def _speak_async(text: str):
    output_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            output_file = fp.name

        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)

        os.system(f'mpv "{output_file}"')
    finally:
        if output_file and os.path.exists(output_file):
            os.remove(output_file)

def speak(text: str):
    asyncio.run(_speak_async(text))