import time
import json
from utils.llm_agent import start_conversation
from utils.parameter_store import load_parameters, overwrite_parameters
from evaluation.user_simulator import SIM_USERS

def run_trial(difficulty, ground_truth, responses):
    chat_history = []
    start_time = time.time()
    num_questions = 0
    for resp in responses:
        reply, saved = start_conversation(resp, chat_history)
        chat_history.append({"role": "user", "content": resp})
        chat_history.append({"role": "assistant", "content": reply})
        num_questions += 1
        if saved:
            break
    latency = time.time() - start_time
    final_params = load_parameters()
    exact_match = json.dumps(final_params, sort_keys=True) == json.dumps(ground_truth, sort_keys=True)
    return exact_match, num_questions, latency

def run_harness():
    results = {}
    for diff in ["easy", "medium", "hard"]:
        accs, qs, lats = [], [], []
        for gt, resps in SIM_USERS[diff]:
            overwrite_parameters({})  # Clear params for each trial
            em, q, lat = run_trial(diff, gt, resps)
            accs.append(em)
            qs.append(q)
            lats.append(lat)
        results[diff] = {
            "exact_match_acc": sum(accs) / len(accs),
            "avg_questions": sum(qs) / len(qs),
            "avg_latency": sum(lats) / len(lats),
            "num_trials": len(accs)
        }
    return results