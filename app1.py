from flask import Flask, request, jsonify
from faqbot import FAQBot
import os
import uuid
import datetime
import jwt

app = Flask(__name__)
bot = FAQBot("data.csv")

# =====================================
# STORAGE
# =====================================
sessions = {}


# =====================================
# GET STUDENT ID FROM TOKEN
# =====================================
def get_student_id_from_token(token):
    return 1


# =====================================
# CREATE SESSION
# =====================================
@app.route("/chat/session", methods=["POST"])
def create_session():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authorization missing"}), 401

        token = auth_header.replace("Bearer ", "")
        bot.token = token

        student_id = get_student_id_from_token(token)
        if not student_id:
            return jsonify({"error": "Invalid token"}), 401

        session_id = str(uuid.uuid4())

        if student_id not in sessions:
            sessions[student_id] = {}

        sessions[student_id][session_id] = {
            "name": "New Chat",
            "messages": [],
            "updated_at": datetime.datetime.now(),
            "last_message": ""
        }

        return jsonify({"session_id": session_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# CHAT
# =====================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        question = data.get("question")
        session_id = data.get("session_id")

        if not question:
            return jsonify({"error": "Question required"}), 400

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authorization missing"}), 401

        token = auth_header.replace("Bearer ", "")
        bot.token = token

        student_id = get_student_id_from_token(token)
        if not student_id:
            return jsonify({"error": "Invalid token"}), 401

        if student_id not in sessions or session_id not in sessions[student_id]:
            return jsonify({"error": "Invalid session"}), 400

        result = bot.answer(question, student_id)

        # حفظ الرسالة
        sessions[student_id][session_id]["messages"].append({
            "q": question,
            "a": result["answer"]
        })

        # 🔥 AUTO CHAT TITLE
        sessions[student_id][session_id]["name"] == "New Chat":
        clean_title = question.strip().capitalize()
        sessions[student_id][session_id]["name"] = clean_title[:30]

        # تحديث الوقت
        sessions[student_id][session_id]["updated_at"] = datetime.datetime.now()

        # حفظ آخر رسالة
        sessions[student_id][session_id]["last_message"] = question

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =====================================
# LIST SESSIONS
# =====================================
@app.route("/chat/sessions", methods=["GET"])
def list_sessions():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authorization missing"}), 401

        token = auth_header.replace("Bearer ", "")
        bot.token = token

        student_id = get_student_id_from_token(token)
        if not student_id:
            return jsonify({"error": "Invalid token"}), 401

        user_sessions = sessions.get(student_id, {})

        # ترتيب حسب آخر استخدام
        sorted_sessions = sorted(
            user_sessions.items(),
            key=lambda x: x[1].get("updated_at"),
            reverse=True
        )

        result = []

        for sid, data in sorted_sessions:
            result.append({
                "session_id": sid,
                "name": data["name"],
                "last_message": data.get("last_message", "")
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# DELETE SESSION
# =====================================
@app.route("/chat/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authorization missing"}), 401

        token = auth_header.replace("Bearer ", "")
        bot.token = token

        student_id = get_student_id_from_token(token)
        if not student_id:
            return jsonify({"error": "Invalid token"}), 401

        if student_id in sessions and session_id in sessions[student_id]:
            del sessions[student_id][session_id]
            return jsonify({"message": "Session deleted"})

        return jsonify({"error": "Session not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# RENAME SESSION
# =====================================
@app.route("/chat/session/<session_id>", methods=["PUT"])
def rename_session(session_id):
    try:
        data = request.get_json()
        new_name = data.get("name")

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authorization missing"}), 401

        token = auth_header.replace("Bearer ", "")
        bot.token = token

        student_id = get_student_id_from_token(token)
        if not student_id:
            return jsonify({"error": "Invalid token"}), 401

        if student_id in sessions and session_id in sessions[student_id]:
            sessions[student_id][session_id]["name"] = new_name
            return jsonify({"message": "Session renamed"})

        return jsonify({"error": "Session not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# HEALTH
# =====================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Guide Bot is working 🚀"
    })


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)