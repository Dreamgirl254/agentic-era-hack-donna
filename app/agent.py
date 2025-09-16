import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from vertexai.preview import generative_models as genai  # Gemini
from langgraph.graph import StateGraph, END

import google.auth
from google.adk.agents import Agent

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# -----------------------------
# Storage Helpers
# -----------------------------
TASKS_FILE = "focusflow_tasks.json"

def load_tasks():
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "suggested": [], "last_completed": None, "streak": 0, "low_count": 0}

def save_tasks(data):
    with open(TASKS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Gemini Helper
# -----------------------------
def rephrase_with_gemini(base_suggestion: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Rephrase this productivity suggestion in an upbeat, motivational way,
    keeping it short and friendly:
    "{base_suggestion}"
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# Agent Logic
# -----------------------------
class FocusFlowAgent:
    def __init__(self):
        self.tasks = load_tasks()

    def suggest_task(self, energy: str) -> str:
        """Suggest task block based on energy, with Gemini rephrasing + multimodal tips."""
        multimodal_tip = ""
        if energy == "low":
            suggestion = "Do a quick 10–15 min task like clearing emails or tidying your desk."
            self.tasks["low_count"] += 1
            if self.tasks["low_count"] >= 3:  # LangGraph branch: too many lows
                multimodal_tip = "🌬️ Try a 2-min breathing exercise to recharge."
        elif energy == "medium":
            suggestion = "Do a 25–30 min focus block, like writing a draft or coding a feature."
            self.tasks["low_count"] = 0
            multimodal_tip = "🎵 Put on your favorite playlist to keep the flow."
        elif energy == "high":
            suggestion = "Go for a 45–60 min deep work sprint on your hardest task."
            self.tasks["low_count"] = 0
            multimodal_tip = "🤸 Take a quick stretch before diving in — prime your body for focus."
        else:
            return "⚡ Tell me your energy level (low, medium, high) to get a tailored suggestion."

        # Pass through Gemini for rephrasing
        motivational = rephrase_with_gemini(suggestion)

        # Save suggestion history
        record = {"energy": energy, "task": motivational, "tip": multimodal_tip}
        self.tasks["suggested"].append(record)
        save_tasks(self.tasks)

        return f"{motivational}\n{multimodal_tip}"

    def mark_completed(self, task: str) -> str:
        """Mark a suggested task as completed and update streak."""
        today = datetime.today().date()
        last_done = self.tasks.get("last_completed")

        if last_done:
            last_done_date = datetime.strptime(last_done, "%Y-%m-%d").date()
            if today - last_done_date == timedelta(days=1):
                self.tasks["streak"] += 1
            elif today > last_done_date:
                self.tasks["streak"] = 1  # reset streak
        else:
            self.tasks["streak"] = 1

        self.tasks["last_completed"] = str(today)
        self.tasks["completed"].append(task)
        save_tasks(self.tasks)

        return f"✅ Task marked complete: {task}\n🔥 Flow Streak: {self.tasks['streak']} days!"

    def get_summary(self) -> str:
        """Summarize user’s progress."""
        completed = len(self.tasks["completed"])
        suggested = len(self.tasks["suggested"])
        streak = self.tasks.get("streak", 0)
        return (
            f"You’ve completed {completed}/{suggested} tasks. "
            f"🔥 Current Flow Streak: {streak} days. Keep it up!"
        )
