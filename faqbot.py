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

# ================= REQUEST =================
    def _safe_get(self, url):
        try:
            headers = {}

            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            r = requests.get(url, headers=headers)

            if r.status_code == 200:
                return r.json()

            return None

        except Exception as e:
            print("ERROR:", e)
            return None
# ============= Data ====================
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
        data = self._safe_get(f"{self.backend_url}/Student/gpa")

        print("DEBUG GPA DATA:", data)

        if not data:
            return None

        # لو dict
        if isinstance(data, dict):
            # 👇 يدعم الشكلين
            if "cumulativeGpa" in data:
                return data.get("cumulativeGpa")

            return data.get("data", {}).get("gpa")

        # لو list
        if isinstance(data, list):
            if len(data) > 0:
                first = data[0]
                if isinstance(first, dict):

                    # 👇 يدعم cumulativeGpa
                    if "cumulativeGpa" in first:
                        return first.get("cumulativeGpa")

                    return first.get("gpa")

        return None

    def _get_current_courses(self):
        data = self._safe_get(f"{self.backend_url}/Enrollment/current-courses")

        print("DEBUG CURRENT:", data)

        if isinstance(data, dict):
            if isinstance(data.get("data"), list) and data["data"]:
                return data["data"]

        if isinstance(data, list):
            return data

        return [
            {"courseName": "Data Structures"},
            {"courseName": "Discrete Mathematics"}
        ]

    def _get_previous_courses(self):
        data = self._safe_get(f"{self.backend_url}/Enrollment/previous-courses")

        print("DEBUG PREVIOUS:", data)

        if isinstance(data, dict):
            if isinstance(data.get("data"), list) and data["data"]:
                return data["data"]

        if isinstance(data, list):
            return data

        return [
            {"courseName": "Intro to Computer Science"},
            {"courseName": "Computer Programming"}
        ]
    def _format_gpa(self, gpa):
        if gpa is None:
            return "No GPA"

        if gpa < 2:
            return f"Your GPA is {gpa} ⚠️ حاول تقلل المواد وتركز"
        elif gpa < 3:
            return f"Your GPA is {gpa} 👍 كويس"
        else:
            return f"Your GPA is {gpa} 🔥 ممتاز"

# ================= SMART AI TITLE =================
    def generate_title(self, question):

        intent = self._detect_intent(question)

        # 👇 titles 
        if intent == "gpa":
            return "GPA Check"

        elif intent == "current":
            return "Current Courses"

        elif intent == "previous":
            return "Completed Courses"

        elif intent == "study_plan":
            return "Study Plan"

        elif intent == "recommend":
            return "Course Recommendations"

        elif intent == "roadmap":
            return "Graduation Roadmap"

        # 👇 fallback للـ AI
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
Generate a short smart chat title (max 4 words).

Rules:
- Based on user question
- Same language
- No symbols
- Clear and meaningful
"""
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.3
            )

            title = res.choices[0].message.content.strip()

            return " ".join(title.split()[:4])

        except Exception as e:
            print("TITLE ERROR:", e)
            return question[:25]
# ================= HELPERS =================
    def _extract_names(self, courses):
        names = []

        for c in courses:
            if isinstance(c, dict):
                names.append(
                    c.get("courseName") or c.get("name") or c.get("title") or "Unknown"
                )
            elif isinstance(c, list):
                names.append(str(c))
            else:
                names.append(str(c))

        return names

# ================= COURSE GRAPH =================
    COURSE_GRAPH = [
    # 🔹 First Year
    {"name": "Intro to Computer Science", "prereq": []},
    {"name": "Computer Programming", "prereq": ["Intro to Computer Science"]},
    {"name": "Discrete Mathematics", "prereq": []},
    {"name": "Linear Algebra", "prereq": []},
    {"name": "Electronics", "prereq": []},
    {"name": "Physics", "prereq": []},

    # 🔹 Second Year
    {"name": "Data Structures", "prereq": ["Computer Programming"]},
    {"name": "Logic Design", "prereq": ["Electronics"]},
    {"name": "Object Oriented Programming", "prereq": ["Computer Programming"]},
    {"name": "File Processing", "prereq": ["Computer Programming"]},
    {"name": "Assembly Language", "prereq": ["Logic Design"]},
    {"name": "Statistics and Probability", "prereq": ["Discrete Mathematics"]},

    # 🔹 Third Year
    {"name": "Algorithms", "prereq": ["Data Structures"]},
    {"name": "Databases", "prereq": ["Data Structures"]},
    {"name": "Operating Systems", "prereq": ["Data Structures"]},
    {"name": "Computer Graphics", "prereq": ["Data Structures"]},
    {"name": "Software Engineering", "prereq": ["Data Structures"]},
    {"name": "Artificial Intelligence", "prereq": ["Algorithms"]},
    {"name": "Neural Networks", "prereq": ["Artificial Intelligence"]},

    # 🔹 Fourth Year
    {"name": "Machine Learning", "prereq": ["Artificial Intelligence"]},
    {"name": "Cloud Computing", "prereq": ["Operating Systems"]},
    {"name": "Computer Security", "prereq": ["Operating Systems"]},
    {"name": "Distributed Systems", "prereq": ["Operating Systems"]},
    {"name": "Parallel Processing", "prereq": ["Operating Systems"]},
    {"name": "Data Warehousing", "prereq": ["Databases"]},
    {"name": "Internet of Things", "prereq": ["Computer Networks"]},

    # 🔹 Supporting Courses
    {"name": "Computer Networks", "prereq": ["Data Structures"]},
    {"name": "Web Programming", "prereq": ["Data Structures"]},
    {"name": "Systems Analysis and Design", "prereq": ["Data Structures"]},

    # 🔹 Graduation
    {"name": "Senior Project 1", "prereq": []},
    {"name": "Graduation Project 2", "prereq": ["Senior Project 1"]},
]

    # ================= COURSE DIFFICULTY =================
    COURSE_DIFFICULTY = {
    "Intro to Computer Science": "easy",
    "Computer Programming": "medium",
    "Discrete Mathematics": "medium",
    "Linear Algebra": "medium",
    "Electronics": "medium",
    "Physics": "medium",

    "Data Structures": "hard",
    "Logic Design": "medium",
    "Object Oriented Programming": "medium",
    "File Processing": "easy",
    "Assembly Language": "hard",
    "Statistics and Probability": "medium",

    "Algorithms": "hard",
    "Databases": "medium",
    "Operating Systems": "hard",
    "Computer Graphics": "medium",
    "Software Engineering": "medium",
    "Artificial Intelligence": "hard",
    "Neural Networks": "hard",

    "Machine Learning": "hard",
    "Cloud Computing": "medium",
    "Computer Security": "medium",
    "Distributed Systems": "hard",
    "Parallel Processing": "hard",
    "Data Warehousing": "medium",
    "Internet of Things": "medium",

    "Computer Networks": "medium",
    "Web Programming": "easy",
    "Systems Analysis and Design": "medium",

    "Senior Project 1": "medium",
    "Graduation Project 2": "hard",
}
# ================= SMART RECOMMEND =================
    def _recommend_smart(self):
        completed = self._extract_names(self._get_previous_courses())
        gpa = self._get_gpa()

        available = []
        for course in self.COURSE_GRAPH:
            if course["name"] in completed:
                continue

            if all(p in completed for p in course["prereq"]):
                available.append(course["name"])

        # 🔥 إضافة فلترة حسب الصعوبة (من غير حذف الكود القديم)
        filtered = []

        for c in available:
            difficulty = self.COURSE_DIFFICULTY.get(c, "medium")

            if gpa:
                if gpa < 2:
                    if difficulty == "easy":
                        filtered.append(c)

                elif gpa < 3:
                    if difficulty != "hard":
                        filtered.append(c)

                else:
                    filtered.append(c)
            else:
                filtered.append(c)

        # 🔥 نفس اللوجيك القديم (بس على filtered)
        if gpa:
            if gpa < 2:
                return filtered[:2]
            elif gpa < 3:
                return filtered[:3]
            else:
                return filtered[:5]

        return filtered
# ================= AVAILABLE COURSES =================
    def _get_available_courses(self, completed):
        available = []

        for course in self.COURSE_GRAPH:
            if course["name"] in completed:
                continue

            if all(p in completed for p in course["prereq"]):
                available.append(course["name"])

        return available
    
# ================= FILTER BY GPA =================
    def _filter_by_gpa(self, courses, gpa):

        if not gpa:
            return courses

        if gpa >= 3.2:
            return courses  # 🔥 كله متاح

        elif gpa >= 2.5:
            return [c for c in courses if self.COURSE_DIFFICULTY.get(c) != "hard"]

        else:
            return [c for c in courses if self.COURSE_DIFFICULTY.get(c) == "easy"]

# ================= ROADMAP =================
    def _generate_roadmap(self):
        completed = self._extract_names(self._get_previous_courses())
        roadmap = []
        temp_done = completed.copy()

        for _ in range(8):
            term = []

            for c in self.COURSE_GRAPH:
                if c["name"] in temp_done:
                    continue

                if all(p in temp_done for p in c["prereq"]):
                    term.append(c["name"])

            if not term:
                break

            roadmap.append(term[:5])
            temp_done.extend(term)

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

Use:
- GPA
- current courses
- previous courses

Be smart, specific, and short.

Memory:
{memory}
"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return res.choices[0].message.content.strip()

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

# ================= MAIN =================
    def answer(self, question, student_id=None):

        lang = self._detect_lang(question)
        intent = self._detect_intent(question)

# ===== GPA =====
        if intent == "gpa":
            gpa = self._get_gpa()
            answer = self._format_gpa(gpa)

# ===== CURRENT =====
        elif intent == "current":
            names = self._extract_names(self._get_current_courses())
            answer = "\n".join(names) if names else "No courses"

# ===== PREVIOUS =====
        elif intent == "previous":
            names = self._extract_names(self._get_previous_courses())
            answer = "\n".join(names) if names else "No previous"

# ===== STUDY PLAN =====
        elif intent == "study_plan":
            gpa = self._get_gpa()
            current = self._extract_names(self._get_current_courses())

            prompt = f"""
Student GPA: {gpa}
Current courses: {current}

Give smart study plan.
"""
            answer = self._ask_ai(prompt, student_id)

# ===== RECOMMEND =====
        elif intent == "recommend":
            rec = self._recommend_smart()
            answer = "\n".join(rec) if rec else "No available courses"

# ===== ROADMAP =====
        elif intent == "roadmap":
            roadmap = self._generate_roadmap()

            txt = ""
            for i, term in enumerate(roadmap, 1):
                txt += f"Term {i}:\n" + "\n".join(term) + "\n\n"

            answer = txt

# ===== FAQ =====
        else:
            faq_answer, score = self._faq(question, lang)

            if score > self.threshold:
                answer = faq_answer
            else:
                answer = self._ask_ai(question, student_id)

        self._save_history(student_id, question, answer)

        return {"answer": answer}