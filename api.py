from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from LLMmain import get_docs, generate_answer
from trans import sinhalaToEnglish, englishToSinhala
import speech_recognition as sr
import os
import tempfile

app = FastAPI()

user_intro: str = ""  # Global temporary store

class QueryRequest(BaseModel):
    query: str

class IntroRequest(BaseModel):
    intro_text: str

# --------------------------
# Endpoint 1: Set User Intro
@app.post("/set_user_intro/")
def set_user_intro(request: IntroRequest):
    global user_intro
    user_intro = request.intro_text
    return {"message": "User intro saved successfully."}

# --------------------------
# Endpoint 2: Get Answer (English or Sinhala)
@app.post("/get_answer/")
def get_answer(request: QueryRequest):
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        answer = generate_answer(request.query, docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_sinhala/")
def get_answer_sinhala(request: QueryRequest):
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        english_answer = generate_answer(english_query, docs)
        sinhala_answer = englishToSinhala(english_answer)
        return {"answer": sinhala_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# Endpoint 2: Get Answer (English or Sinhala)
@app.post("/get_answer_voice/")
async def get_answer_voice(audio_file: UploadFile = File(...)):
    temp_audio_path = None
    try:
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        content = await audio_file.read()
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(content)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text_query = recognizer.recognize_google(audio_data)
        docs = get_docs(text_query, top_k=5)
        answer = generate_answer(text_query, docs)
        return {"query": text_query, "answer": answer}
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech could not be understood")
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Speech recognition service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))  # Use Railway's PORT env or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)



# POST /set_user_intro/
# {
#   "intro_text": "EPDS score: 17 (moderate depression risk). The mother is feeling isolated and has trouble sleeping."
# }
