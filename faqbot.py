import numpy as np
import pandas as pd
import requests
import os

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class FAQBot:
    def __init__(self, excel_path: str, threshold: float = 0.5):

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

        self.history = {}

        self.backend_url = "https://graduationproject48.runasp.net/api"
        self.token = None

        self._load_data()
        self._build_models()

# ================= LOAD DATA =================
    def _load_data(self):
        ext = os.path.splitext(self.excel_path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(self.excel_path)
        else:
            df = pd.read_excel(self.excel_path)

        self.questions_en = df["question"].fillna("").tolist()
        self.answers_en = df["answer"].fillna("").tolist()

        self.questions_ar = df.get("question_ar", pd.Series()).fillna("").tolist()
        self.answers_ar = df.get("answer_ar", pd.Series()).fillna("").tolist()

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

# ================= REQUEST =================
    def _safe_get(self, url):
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            r = requests.get(url, headers=headers)

            if r.status_code == 200:
                return r.json()

            return None
        except Exception as e:
            print("ERROR:", e)
            return None

# ================= BACKEND =================
    def _get_gpa(self):
        data = self._safe_get(f"{self.backend_url}/Student/gpa")
        return data.get("data", {}).get("gpa") if data else None

    def _get_current_courses(self):
        data = self._safe_get(f"{self.backend_url}/Enrollment/current-courses")
        return data.get("data", []) if data else []

    def _get_previous_courses(self):
        data = self._safe_get(f"{self.backend_url}/Enrollment/previous-courses")
        return data.get("data", []) if data else []

# ================= FIX BUG هنا 🔥 =================
    def _extract_names(self, courses):
        if not isinstance(courses, list):
            return []

        names = []

        for c in courses:
            if isinstance(c, dict):
                names.append(
                    c.get("courseName") or 
                    c.get("name") or 
                    c.get("title") or 
                    "Unknown"
                )

        return names

# ================= COURSE GRAPH (من الصورة) =================
    COURSE_GRAPH = [
        {"name": "Intro to Computer Science", "prereq": []},
        {"name": "Computer Programming", "prereq": ["Intro to Computer Science"]},
        {"name": "Discrete Mathematics", "prereq": []},
        {"name": "Linear Algebra", "prereq": []},
        {"name": "Electronics", "prereq": []},
        {"name": "Physics", "prereq": []},

        {"name": "Data Structures", "prereq": ["Computer Programming"]},
        {"name": "Logic Design", "prereq": ["Electronics"]},

        {"name": "Algorithms", "prereq": ["Data Structures"]},
        {"name": "Databases", "prereq": ["Data Structures"]},
        {"name": "Operating Systems", "prereq": ["Algorithms"]},
        {"name": "Artificial Intelligence", "prereq": ["Algorithms"]},

        {"name": "Machine Learning", "prereq": ["Artificial Intelligence"]},
        {"name": "Distributed Systems", "prereq": ["Operating Systems"]},
        {"name": "Cloud Computing", "prereq": ["Networks"]},
        {"name": "Cyber Security", "prereq": ["Operating Systems"]},
    ]

# ================= SMART RECOMMEND =================
    def _recommend_smart(self):
        completed = self._extract_names(self._get_previous_courses())
        current = self._extract_names(self._get_current_courses())
        gpa = self._get_gpa()

        available = []

        for course in self.COURSE_GRAPH:

            if course["name"] in completed or course["name"] in current:
                continue

            if all(p in completed for p in course["prereq"]):
                available.append(course["name"])

        if gpa:
            if gpa < 2:
                return available[:2]
            elif gpa < 3:
                return available[:4]
            else:
                return available[:6]

        return available

# ================= ROADMAP =================
    def _generate_roadmap(self):
        completed = self._extract_names(self._get_previous_courses())

        roadmap = []
        temp = completed.copy()

        for _ in range(8):
            term = []

            for c in self.COURSE_GRAPH:
                if c["name"] in temp:
                    continue

                if all(p in temp for p in c["prereq"]):
                    term.append(c["name"])

            if not term:
                break

            roadmap.append(term[:5])
            temp.extend(term)

        return roadmap

# ================= AI =================
    def _ask_ai(self, prompt, student_id=None):
        memory = self._get_memory(student_id)

        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a smart academic advisor.

Use GPA and courses.

Memory:
{memory}
"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return res.choices[0].message.content.strip()

# ================= FAQ =================
    def _faq(self, q, lang):
        vec = self.vectorizer_ar if lang == "ar" else self.vectorizer_en
        mat = self.matrix_ar if lang == "ar" else self.matrix_en
        answers = self.answers_ar if lang == "ar" else self.answers_en

        sims = cosine_similarity(vec.transform([q]), mat)[0]
        idx = np.argmax(sims)

        return answers[idx], sims[idx]

# ================= HISTORY =================
    def _save_history(self, sid, q, a):
        if not sid:
            return

        self.history.setdefault(sid, []).append({"q": q, "a": a})

    def _get_memory(self, sid):
        return "\n".join(
            [f"{h['q']} -> {h['a']}" for h in self.history.get(sid, [])[-5:]]
        )

# ================= INTENT =================
    def _detect_intent(self, q):
        q = q.lower()

        if "gpa" in q or "معدل" in q:
            return "gpa"
        if "current" in q or "باخد" in q:
            return "current"
        if "previous" in q or "خلصت" in q:
            return "previous"
        if "plan" in q or "خطة" in q:
            return "study_plan"
        if "recommend" in q or "اخد ايه" in q:
            return "recommend"
        if "roadmap" in q or "تخرج" in q:
            return "roadmap"

        return "faq"

# ================= MAIN =================
    def answer(self, question, student_id=None):

        lang = self._detect_lang(question)
        intent = self._detect_intent(question)

        if intent == "gpa":
            gpa = self._get_gpa()
            answer = f"GPA: {gpa}" if gpa else "No GPA"

        elif intent == "current":
            names = self._extract_names(self._get_current_courses())
            answer = "\n".join(names) if names else "No courses"

        elif intent == "previous":
            names = self._extract_names(self._get_previous_courses())
            answer = "\n".join(names) if names else "No previous"

        elif intent == "study_plan":
            gpa = self._get_gpa()
            current = self._extract_names(self._get_current_courses())

            prompt = f"""
Student GPA: {gpa}
Current courses: {current}

Give smart study plan.
"""
            answer = self._ask_ai(prompt, student_id)

        elif intent == "recommend":
            rec = self._recommend_smart()
            answer = "Recommended:\n" + "\n".join(rec) if rec else "No available courses"

        elif intent == "roadmap":
            roadmap = self._generate_roadmap()

            txt = ""
            for i, term in enumerate(roadmap, 1):
                txt += f"Term {i}:\n" + "\n".join(term) + "\n\n"

            answer = txt

        else:
            faq_answer, score = self._faq(question, lang)

            if score > self.threshold:
                answer = faq_answer
            else:
                answer = self._ask_ai(question, student_id)

        self._save_history(student_id, question, answer)

        return {"answer": answer}