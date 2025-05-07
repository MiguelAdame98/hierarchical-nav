# pcfg_spatial_planner.py
# A self-contained demo of PCFG-based hierarchical spatial planning using NLTK

import random
from nltk import PCFG
from nltk.parse.generate import generate

# ------------------------
# Placeholder for MemoryGraph interaction
# ------------------------
class MockMemoryGraph:
    def __init__(self):
        self.experiences = {
            17: {'links': [19, 5]},
            19: {'links': [18]},
            5: {'links': [4]},
            4: {'links': [3]},
            3: {'links': []},
            18: {'links': []},
        }
        self.current_exp = 17

    def get_current_exp_id(self):
        return self.current_exp

    def get_exps_organised_by_distance_from_exp(self, exp_id=None):
        if exp_id is None:
            exp_id = self.current_exp
        # For simplicity, return a flat dict with dummy distances
        return {
            18: 2,
            3: 3
        }

    def get_all_exps_in_memory(self, wt_links=True):
        return self.experiences

# ------------------------
# Grammar Generator from MemoryGraph
# ------------------------
def build_exploration_grammar(memory_graph):
    current_exp = memory_graph.get_current_exp_id()
    distances = memory_graph.get_exps_organised_by_distance_from_exp(current_exp)

    # Assign probabilities inversely proportional to distance
    total = sum(1.0 / (dist + 1e-5) for dist in distances.values())
    rules = []
    for target, dist in distances.items():
        prob = (1.0 / (dist + 1e-5)) / total
        rules.append(f"NAVPLAN -> GOTO_{target} [{prob:.4f}]")

    # Terminal expansions
    terminal_rules = []
    for src, data in memory_graph.get_all_exps_in_memory().items():
        for dst in data['links']:
            terminal_rules.append(f"STEP_{src}_{dst} -> 'step({src},{dst})' [1.0]")

    # Compose the grammar
    grammar_string = [
        "EXPLORE -> NAVPLAN [1.0]"
    ] + rules + [
        f"GOTO_{target} -> MOVESEQ_{current_exp}_{target} [1.0]" for target in distances
    ] + [
        f"MOVESEQ_{current_exp}_{18} -> STEP_{current_exp}_19 STEP_19_18 [1.0]",
        f"MOVESEQ_{current_exp}_{3} -> STEP_{current_exp}_5 STEP_5_4 STEP_4_3 [1.0]"
    ] + terminal_rules

    return PCFG.fromstring("\n".join(grammar_string))

# ------------------------
# Sample and Display Plans
# ------------------------
def sample_plan(grammar, n=3):
    print("\nSampled Exploration Plans:")
    for plan in generate(grammar, n=n, start=grammar.start()):
        print("  ->", plan)

# ------------------------
# Run a Simple Demo
# ------------------------
if __name__ == "__main__":
    print("Creating PCFG from MemoryGraph...")
    memory_graph = MockMemoryGraph()
    grammar = build_exploration_grammar(memory_graph)

    print("Productions:")
    for prod in grammar.productions():
        print(" ", prod)

    sample_plan(grammar, n=5)  # Generate some plans
