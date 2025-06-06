# ─────────────────────────────────────────────────────────────────────────────
#  Dermatology multi‑agent system for healthcare - MASH - demo (sorry for few comments)
# ─────────────────────────────────────────────────────────────────────────────

# How to Run:

# Export Open AI API-key into environment
# Required Python dependencies:
#   pip install --upgrade streamlit openai transformers pillow requests torch accelerate

# Then run with
# streamlit run MASHvoice.py


import os, io, json, base64, requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from transformers import pipeline
from PIL import Image

# 0) OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) HF skin‑cancer classifier pipeline
pipe = pipeline(
    "image-classification",
    model="Anwarkh1/Skin_Cancer-Image_Classification",   # or "./Classifier"
    device=0                                            # use -1 for CPU
)

# 2) Voices & helper to autoplay audio invisibly
ASSISTANT_VOICE = "shimmer"
DERM_VOICE      = "ash"

def play_audio(text: str, voice: str) -> None:
    """Synthesize speech and autoplay it without showing controls."""
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    audio_bytes = resp.content              # raw WAV bytes
    b64 = base64.b64encode(audio_bytes).decode()
    html = (
        f'<audio autoplay style="display:none">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        f"</audio>"
    )
    components.html(html, height=0)

# 3) Prompts
ASSISTANT_SYS_PROMPT = (
    "You are the virtual front-desk assistant of a dermatology practice. "
    "Greet the user, gather basic info, and then transfer the patient to the dermatologist AI agent to check sysmptoms to figure out what kind of appointment they need."
    "when it's time to check their skin symptoms, call the transfer_agent function with {\"agent\":\"Dermatologist\"}. "
    "Before transfering, tell the user that if it's okay you will transfer to the dermatologist and why you will do that"
    "Today is 05.05.2025. The next available appointment is Thursday 08.05.2025 at 11AM."
    "At the very end of the conversation, When the appointment is made and the date is fix, send a confirmation email by calling send_email(to,subject,body). "
    "Before sending the email, ask for the user to confirm the details"
    "Be concise"
    "If the possible diagnosis or classification was cancerous, don't be so upbeat."
)

DERM_SYS_PROMPT = (
    "You are a dermatologist AI. You have access to a classification result from classify_skin." \
    "Your job is to consult people on their skin symptoms. Then evaluate their symptoms, and decide what kind of appointment (type and length) they need."
    "When a patient describes symptoms, ask them to send a photo of it"
    "After you have recommended an appointment, call the transfer agent to finalise the appointment booking"
    "Call the transfer_agent with {\"agent\":\"Assistant\"}."
    "Before transfering, tell the user that if it's okay you will transfer to the Assistant and why you will do that"
    "If the classification is benign, still recommend a 10-minute check-up and transfer back to the Assistant "
    "If the classification is actually cancer, then only say that there may be a possibility that this is this diagnosis and recommend 30 min appointment"
)

# 4) Function-call metadata for the Chat API
EMAIL_FUNCTION = {
    "name": "send_email",
    "description": "Send an email to the user with a given subject and body",
    "parameters": {
        "type": "object",
        "properties": {
            "to":      {"type": "string"},
            "subject": {"type": "string"},
            "body":    {"type": "string"}
        },
        "required": ["to", "subject", "body"]
    }
}

TRANSFER_FUNCTION = {
    "name": "transfer_agent",
    "description": "Route the conversation to a different agent",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": ["Assistant", "Dermatologist"]}
        },
        "required": ["agent"]
    }
}

# 5) Helper utilities
def wrap_message_content(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else content

def unwrap_content(content):
    if isinstance(content, list):
        return "".join(p.get("text","") for p in content if p.get("type")=="text")
    return content

def to_api_messages(history):
    msgs=[]
    for m in history:
        role=m["role"]
        if role in ("system","user"):
            msgs.append({"role":role,"content":wrap_message_content(m["content"])})
        elif role in ("assistant","dermatologist"):
            msgs.append({"role":"assistant","content":wrap_message_content(m["content"])})
        else:   # function
            msgs.append({"role":"function","name":m["name"],"content":m["content"]})
    return msgs

# 6) Email + classifier implementations
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY") # My free trial has expired
EMAIL_FROM       = os.getenv("EMAIL_FROM")

def send_email(to:str, subject:str, body:str)->str:
    url = "https://api.sendgrid.com/v3/mail/send"
    hdr = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": EMAIL_FROM},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}]
    }
    preview = (body[:75] + "…") if len(body) > 75 else body
    return f" Confirmation email sent to {to} · subject: '{subject}' · body preview: '{preview}' ✅"

def classify_skin(image_bytes:bytes)->str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label = pipe(img, top_k=1)[0]["label"]
    return label

# 7) Agent wrappers
def assistant_agent(api_msgs):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":wrap_message_content(ASSISTANT_SYS_PROMPT)}]+api_msgs[1:],
        functions=[EMAIL_FUNCTION, TRANSFER_FUNCTION],
        function_call="auto"
    )
    m = resp.choices[0].message
    if getattr(m,"function_call",None):
        name=m.function_call.name
        args=json.loads(m.function_call.arguments)
        if name=="send_email":
            return send_email(**args)
        if name=="transfer_agent":
            return {"transfer_to":args["agent"]}
    text=unwrap_content(m.content)
    play_audio(text, ASSISTANT_VOICE)
    return text

def dermatologist_agent(api_msgs):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":wrap_message_content(DERM_SYS_PROMPT)}]+api_msgs[1:],
        functions=[TRANSFER_FUNCTION],
        function_call="auto"
    )
    m = resp.choices[0].message
    if getattr(m,"function_call",None) and m.function_call.name=="transfer_agent":
        return {"transfer_to":json.loads(m.function_call.arguments)["agent"]}
    text=unwrap_content(m.content)
    play_audio(text, DERM_VOICE)
    return text

# 8) Streamlit UI
st.set_page_config(page_title="Dermatology Practice", layout="wide")
st.title("Dermatology Practice 🏥")

if "history" not in st.session_state:
    st.session_state.history=[{"role":"system","content":ASSISTANT_SYS_PROMPT}]
if "last_classification" not in st.session_state:
    st.session_state.last_classification=None

for m in st.session_state.history:
    if m["role"]=="system": continue
    disp=m["role"].capitalize()
    if m["role"] in ("user","assistant","dermatologist"):
        st.chat_message(disp).write(m["content"])
    else:
        st.chat_message("Function").write(f"{m['name']}: {m['content']}")

if st.session_state.last_classification:
    st.markdown(f"**Last classification:** {st.session_state.last_classification}")

user_input = st.chat_input(
    placeholder="Ask for appointments, describe symptoms, or attach a photo…",
    accept_file=True,
    file_type=["png","jpg","jpeg"]
)

if user_input is not None:
    text, files = (user_input, []) if isinstance(user_input,str) else (user_input.text or "", user_input.files or [])

    if text:
        st.session_state.history.append({"role":"user","content":text})
        st.chat_message("User").write(text)

    for f in files:
        raw=f.read()
        img=Image.open(io.BytesIO(raw)).convert("RGB")
        label=classify_skin(raw)
        st.session_state.history.append({"role":"function","name":"classify_skin","content":label})
        st.session_state.last_classification=label
        st.chat_message("User").image(img,caption=f.name)

    api_msgs = to_api_messages(st.session_state.history)

    who = "Dermatologist" if files else "Assistant"
    out = assistant_agent(api_msgs) if who=="Assistant" else dermatologist_agent(api_msgs)

    if isinstance(out,dict) and "transfer_to" in out:
        tgt=out["transfer_to"]
        st.session_state.history.append({"role":"function","name":"transfer_agent","content":json.dumps({"agent":tgt})})
        api_msgs = to_api_messages(st.session_state.history)
        out = assistant_agent(api_msgs) if tgt=="Assistant" else dermatologist_agent(api_msgs)
        who = tgt

    reply = out if isinstance(out,str) else ""
    st.session_state.history.append({"role":who.lower(),"content":reply})
    st.chat_message(who).write(reply)