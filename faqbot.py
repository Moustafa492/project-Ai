import numpy as np
import pandas as pd
import requests
import os

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class FAQBot:
    def __init__(self, excel_path: str, threshold: float = 0.35):

        self.excel_path = excel_path
        self.threshold = threshold  

        self.questions_en = []
        self.answers_en = []
        self.questions_ar = []
        self.answers_ar = []

        self.vectorizer_en = None
        self.matrix_en = None
        self.vectorizer_ar = None
        self.matrix_ar = None

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.cache = {}
        self.history = {}

        self.backend_url = "https://graduationproject48.runasp.net/api"
        self.token = None

        self._load_data()
        self._build_models()

# ================= REQUEST =================
    def _safe_get(self, url):
        try:
            headers = {
                "Authorization": f"Bearer {self.token}"
            }

            r = requests.get(url, headers=headers)

            print("URL:", url)
            print("Status:", r.status_code)
            print("Response:", r.text)

            if r.status_code == 200:
                return r.json()

            return None

        except Exception as e:
            print("ERROR:", e)
            return None

# ================= DATA =================
    def _load_data(self):
        ext = os.path.splitext(self.excel_path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(self.excel_path)
        else:
            df = pd.read_excel(self.excel_path)

        self.questions_en = df["question"].dropna().tolist()
        self.answers_en = df["answer"].dropna().tolist()

        self.questions_ar = df["question_ar"].dropna().tolist()
        self.answers_ar = df["answer_ar"].dropna().tolist()

# ================= MODELS =================
    def _build_models(self):
        self.vectorizer_en = TfidfVectorizer()
        self.matrix_en = self.vectorizer_en.fit_transform(self.questions_en)

        self.vectorizer_ar = TfidfVectorizer()
        self.matrix_ar = self.vectorizer_ar.fit_transform(self.questions_ar)

# ================= LANG =================
    def _detect_lang(self, text):
        try:
            return "ar" if detect(text) == "ar" else "en"
        except:
            return "en"

# ================= BACKEND =================

    def _get_gpa(self):
        url = f"{self.backend_url}/Student/gpa"
        data = self._safe_get(url)

        if data and "data" in data:
            return data["data"].get("gpa")

        return None

    def _get_current_courses(self):
        url = f"{self.backend_url}/Enrollment/current-courses"
        data = self._safe_get(url)

        if data and "data" in data:
            return data["data"]

        return []

    def _get_previous_courses(self):
        url = f"{self.backend_url}/Enrollment/previous-courses"
        data = self._safe_get(url)

        if data and "data" in data:
            return data["data"]

        return []

# ================= AI =================
    def _ask_ai(self, question, student_id=None):
        try:
            memory = self._get_memory(student_id) if student_id else ""

            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {
                    "role": "system",
                    "content": f"""
You are a university assistant.

Rules:
- Answer in the SAME language as the user (Arabic or English).
- Keep the answer SHORT and clear.
- Do NOT use markdown.
- Do NOT use symbols like # or *.
- Answer like a normal chatbot.
- If the user asks about GPA, courses, or study, be helpful and direct.

Previous conversation:
{memory}
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.4
        )

            return res.choices[0].message.content.strip()

        except Exception as e:
            print("AI Error:", str(e))
            return "Something went wrong"
        
# ================ Genrate Title ==================
    def generate_title(self, question):
        try:
            prompt = f"""
Generate a very short title (max 5 words) for this question:
{question}

Only return the title.
"""

            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            return res.choices[0].message.content.strip()

        except:
            return question[:30]
        
# =================== Detect Intent ================
    def _detect_intent(self, question):
       prompt = f"""
Classify the user question into EXACTLY ONE of these intents:

gpa
current_courses
previous_courses
study_plan
faq
unknown

IMPORTANT:
- Return ONLY one word
- Use exactly the same spelling

Examples:
Q: what is my gpa → gpa
Q: what subjects am I taking → current_courses
Q: what did I finish → previous_courses

Question: {question}
"""

       try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            intent = res.choices[0].message.content.strip().lower()
            return intent
       
       except Exception as e:
            print("Intent Error:", e)
            return "unknown"
       
# ================= HISTORY =================
    def _save_history(self, student_id, q, a):
        if student_id not in self.history:
            self.history[student_id] = []

        self.history[student_id].append({
            "q": q,
            "a": a
        })

    def _get_memory(self, student_id):
        history = self.history.get(student_id, [])[-10:]  
        return "\n".join([f"Q: {h['q']} A: {h['a']}" for h in history])

# ================= FAQ =================
    def _faq(self, q, lang):
        vec = self.vectorizer_ar if lang == "ar" else self.vectorizer_en
        mat = self.matrix_ar if lang == "ar" else self.matrix_en
        answers = self.answers_ar if lang == "ar" else self.answers_en

        sims = cosine_similarity(vec.transform([q]), mat)[0]
        idx = np.argmax(sims)

        return answers[idx], sims[idx]

# ================= MAIN =================
    def answer(self, question, student_id=None):
        try:
            intent = self._detect_intent(question)
            intent = intent.replace(" ", "_")
            print("Detected Intent:", intent)

            answer = None  # 🔥 لازم جوه try

        # ================= GPA =================
            if intent == "gpa":
                gpa = self._get_gpa()
                answer = f"Your GPA is: {gpa}" if gpa else "No GPA found"

        # ================= CURRENT COURSES =================
            elif intent == "current_courses":
                data = self._get_current_courses()
                if not data:
                    answer = "No current courses found"
                else:
                    courses = [c.get("courseName", "Unknown") for c in data]
                    answer = "Your current courses:\n" + "\n".join(courses)

        # ================= PREVIOUS COURSES =================
            elif intent == "previous_courses":
                data = self._get_previous_courses()
                if not data:
                    answer = "No previous courses found"
                else:
                    courses = [c.get("courseName", "Unknown") for c in data]
                    answer = "Your completed courses:\n" + "\n".join(courses)

        # ================= Study Plan =================
            elif intent == "study_plan":
                gpa = self._get_gpa()
                answer = self._ask_ai(f"My GPA is {gpa}. Give me a study plan.", student_id)

        # ================= FAQ + AI =================
            else:
                faq, score = self._faq(question, self._detect_lang(question))
                if score > 0.7:
                    answer = faq
                else:
                    answer = self._ask_ai(question, student_id)

            # 🔥 حفظ الهستوري
            self._save_history(student_id, question, answer)

            return {"answer": answer}

        except Exception as e:
            print("MAIN ERROR:", str(e))
            return {"answer": "Something went wrong"}