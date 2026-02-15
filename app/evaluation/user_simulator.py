import time
import json
from utils.llm_agent import start_conversation
from utils.parameter_store import load_parameters, overwrite_parameters

# app/evaluation/user_simulator.py
SIM_USERS = {
    "easy": [
        (
            {
                "city": "Tunis", "user_type": "residential", "start_date": "2025-10-24", "end_date": "2025-10-25",
                "appliances": ["Water Heater"], "appliance_settings": [{"name": "Water Heater", "start_time": "00:00", "end_time": "23:59", "can_shift": True}],
                "Tmin": 20.0, "Tmax": 25.0, "max_power": 5.0, "do_not_disturb": [], "solar_pv_capacity": 5.0
            },
            [
                "I live in Tunis, residential user, set dates to 2025-10-24 to 2025-10-25.",
                "I have a water heater that can run anytime.",
                "Set comfort temp to 20-25°C, max power 5 kW, solar capacity 5 kW."
            ]
        ),
        # Add9 more 
    ],
    "medium": [
        (
            {
                "city": "Tunis", "user_type": "industrial", "start_date": "2025-10-24", "end_date": "2025-10-25",
                "appliances": ["Water Heater", "Dishwasher"], "appliance_settings": [
                    {"name": "Water Heater", "start_time": "00:00", "end_time": "23:59", "can_shift": True},
                    {"name": "Dishwasher", "start_time": "18:00", "end_time": "22:00", "can_shift": False}
                ],
                "Tmin": 20.0, "Tmax": 25.0, "max_power": 50.0, "do_not_disturb": [], "solar_pv_capacity": 15.0
            },
            [
                "I’m in Tunis, industrial facility, dates 2025-10-24 to 2025-10-25.",
                "I use a water heater anytime and a dishwasher in the evening.",
                "Comfort range 20-25°C, max power 50 kW, solar capacity 15 kW."
            ]
        ),
        # Add more 
    ],
    "hard": [
        (
            {
                "city": "Tunis", "user_type": "residential", "start_date": "2025-10-24", "end_date": "2025-10-25",
                "appliances": ["Water Heater", "Heating"], "appliance_settings": [
                    {"name": "Water Heater", "start_time": "", "end_time": "", "can_shift": True},
                    {"name": "Heating", "start_time": "00:00", "end_time": "23:59", "can_shift": False}
                ],
                "Tmin": 18.0, "Tmax": 22.0, "max_power": 10.0, "do_not_disturb": ["22:00-07:00"], "solar_pv_capacity": 10.0
            },
            [
                "I’m in Tunis, it’s a house, set it for October 24, 2025.",
                "I have a water heater, not sure when it runs, and heating all day.",
                "Keep it comfy, maybe 18-22°C, don’t disturb at night, solar panels."
            ]
        ),
        # Add  more cases 
    ]
}

def run_trial(difficulty, ground_truth, responses):
    # Mock conversation: Feed responses sequentially
    chat_history = []
    start_time = time.time()
    num_questions = 0
    for resp in responses:
        reply, saved = start_conversation(resp, chat_history)
        chat_history.append({"role": "user", "content": resp})
        chat_history.append({"role": "assistant", "content": reply})
        num_questions += 1  # Count assistant messages as questions
        if saved:
            break
    latency = time.time() - start_time
    final_params = load_parameters()
    exact_match = json.dumps(final_params) == json.dumps(ground_truth)  
    return exact_match, num_questions, latency

def run_harness():
    results = {}
    for diff in ["easy", "medium", "hard"]:
        accs, qs, lats = [], [], []
        for gt, resps in SIM_USERS[diff]:
            overwrite_parameters({})  # Reset
            em, q, lat = run_trial(diff, gt, resps)
            accs.append(em)
            qs.append(q)
            lats.append(lat)
        results[diff] = {
            "exact_match_acc": sum(accs) / len(accs),
            "avg_questions": sum(qs) / len(qs),
            "avg_latency": sum(lats) / len(lats)
        }
    print(json.dumps(results, indent=2))
    # Target: easy ~90% acc, 5 qs; medium ~80%, 7 qs; hard ~60%, 10 qs (ReAct-like with few steps)

if __name__ == "__main__":
    run_harness()