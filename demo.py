"""
Gradio demo for SmartTrip Agent.
"""
import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


def plan_fn(destination: str, origin: str, start: str, end: str, budget: int, preferences: str, travelers: int):
    import uuid
    sid = str(uuid.uuid4())
    try:
        resp = httpx.post(f"{API_BASE}/plan", json={
            "destination": destination,
            "origin": origin,
            "start_date": start,
            "end_date": end,
            "budget": budget,
            "preferences": preferences,
            "travelers": travelers,
            "session_id": sid,
        }, timeout=120)
        data = resp.json()
        steps = data.get("steps", 0)
        tools = data.get("tools_called", [])
        tools_str = ", ".join(tools) if tools else "none"
        return f"{data['itinerary']}\n\n---\n*{steps} ReAct steps, tools: {tools_str}*"
    except Exception as e:
        return f"[Error: {e}]"


with gr.Blocks(title="SmartTrip Agent") as demo:
    gr.Markdown("## SmartTrip AI Travel Planner\n*Powered by LangGraph ReAct Agent*")

    with gr.Row():
        with gr.Column(scale=2):
            destination = gr.Textbox(label="Destination", placeholder="e.g. Chengdu, Tokyo, Paris")
            origin = gr.Textbox(label="Departure City", placeholder="e.g. Beijing, Shanghai")
            with gr.Row():
                start_date = gr.Textbox(label="Start Date", placeholder="2026-07-01")
                end_date = gr.Textbox(label="End Date", placeholder="2026-07-05")
            budget = gr.Number(label="Budget (CNY)", value=5000, minimum=0)
            preferences = gr.Textbox(label="Preferences", placeholder="e.g. nature, food, budget-friendly, luxury")
            travelers = gr.Number(label="Travelers", value=1, minimum=1, maximum=20)
            plan_btn = gr.Button("Plan My Trip", variant="primary")

        with gr.Column(scale=3):
            output = gr.Markdown("Your itinerary will appear here...", label="Itinerary")

    plan_btn.click(
        plan_fn,
        [destination, origin, start_date, end_date, budget, preferences, travelers],
        output,
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
