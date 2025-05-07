from nltk import PCFG
from nltk.parse.generate import generate
import random
rule_grammar = PCFG.fromstring(""" 

    EXPLORE -> NAVPLAN [1.0000]
    NAVPLAN -> GOTO_9 [0.0000]
    NAVPLAN -> GOTO_8 [0.0000]
    NAVPLAN -> GOTO_10 [0.0000]
    NAVPLAN -> GOTO_14 [0.0000]
    NAVPLAN -> GOTO_11 [0.0000]
    NAVPLAN -> GOTO_7 [0.0000]
    NAVPLAN -> GOTO_13 [0.0000]
    NAVPLAN -> GOTO_15 [0.0000]
    NAVPLAN -> GOTO_2 [0.0000]
    NAVPLAN -> GOTO_5 [0.0000]
    NAVPLAN -> GOTO_0 [1.0000]
    NAVPLAN -> GOTO_6 [0.0000]
    NAVPLAN -> GOTO_3 [0.0000]
    NAVPLAN -> GOTO_12 [0.0000]
    NAVPLAN -> GOTO_4 [0.0000]
    NAVPLAN -> GOTO_1 [0.0000]
    GOTO_9 -> MOVESEQ_9_9 [1.0000]
    GOTO_8 -> MOVESEQ_9_8 [1.0000]
    GOTO_10 -> MOVESEQ_9_10 [1.0000]
    GOTO_14 -> MOVESEQ_9_14 [1.0000]
    GOTO_11 -> MOVESEQ_9_11 [1.0000]
    GOTO_7 -> MOVESEQ_9_7 [1.0000]
    GOTO_13 -> MOVESEQ_9_13 [1.0000]
    GOTO_15 -> MOVESEQ_9_15 [1.0000]
    GOTO_2 -> MOVESEQ_9_2 [1.0000]
    GOTO_5 -> MOVESEQ_9_5 [1.0000]
    GOTO_0 -> MOVESEQ_9_0 [1.0000]
    GOTO_6 -> MOVESEQ_9_6 [1.0000]
    GOTO_3 -> MOVESEQ_9_3 [1.0000]
    GOTO_12 -> MOVESEQ_9_12 [1.0000]
    GOTO_4 -> MOVESEQ_9_4 [1.0000]
    GOTO_1 -> MOVESEQ_9_1 [1.0000]
    MOVESEQ_9_9 -> STEP_9_9 [1.0000]
    MOVESEQ_9_8 -> STEP_9_8 [1.0000]
    MOVESEQ_9_10 -> STEP_9_10 [1.0000]
    MOVESEQ_9_14 -> STEP_9_14 [1.0000]
    MOVESEQ_9_11 -> STEP_9_11 [1.0000]
    MOVESEQ_9_7 -> STEP_9_7 [1.0000]
    MOVESEQ_9_13 -> STEP_9_13 [1.0000]
    MOVESEQ_9_15 -> STEP_9_15 [1.0000]
    MOVESEQ_9_2 -> STEP_9_2 [1.0000]
    MOVESEQ_9_5 -> STEP_9_5 [1.0000]
    MOVESEQ_9_0 -> STEP_9_0 [1.0000]
    MOVESEQ_9_6 -> STEP_9_6 [1.0000]
    MOVESEQ_9_3 -> STEP_9_3 [1.0000]
    MOVESEQ_9_12 -> STEP_9_12 [1.0000]
    MOVESEQ_9_4 -> STEP_9_4 [1.0000]
    MOVESEQ_9_1 -> STEP_9_1 [1.0000]
    MOVESEQ_9_18 -> STEP_9_19 STEP_19_18 [1.0000]
    STEP_9_5 -> 'step(9,5)' [1.0000]
    STEP_19_18 -> 'step(19,18)' [1.0000]
    STEP_9_14 -> 'step(9,14)' [1.0000]
    STEP_4_3 -> 'step(4,3)' [1.0000]
    STEP_9_12 -> 'step(9,12)' [1.0000]
    STEP_10_11 -> 'step(10,11)' [1.0000]
    STEP_5_6 -> 'step(5,6)' [1.0000]
    STEP_13_11 -> 'step(13,11)' [1.0000]
    STEP_9_8 -> 'step(9,8)' [1.0000]
    STEP_9_7 -> 'step(9,7)' [1.0000]
    STEP_9_0 -> 'step(9,0)' [1.0000]
    STEP_6_5 -> 'step(6,5)' [1.0000]
    STEP_9_11 -> 'step(9,11)' [1.0000]
    STEP_9_13 -> 'step(9,13)' [1.0000]
    STEP_3_5 -> 'step(3,5)' [1.0000]
    STEP_9_2 -> 'step(9,2)' [1.0000]
    STEP_2_3 -> 'step(2,3)' [1.0000]
    STEP_7_8 -> 'step(7,8)' [1.0000]
    STEP_15_14 -> 'step(15,14)' [1.0000]
    STEP_1_0 -> 'step(1,0)' [1.0000]
    STEP_9_1 -> 'step(9,1)' [1.0000]
    STEP_6_7 -> 'step(6,7)' [1.0000]
    STEP_9_6 -> 'step(9,6)' [1.0000]
    STEP_0_2 -> 'step(0,2)' [1.0000]
    STEP_14_13 -> 'step(14,13)' [1.0000]
    STEP_14_15 -> 'step(14,15)' [1.0000]
    STEP_8_7 -> 'step(8,7)' [1.0000]
    STEP_7_6 -> 'step(7,6)' [1.0000]
    STEP_3_4 -> 'step(3,4)' [1.0000]
    STEP_12_11 -> 'step(12,11)' [1.0000]
    STEP_9_15 -> 'step(9,15)' [1.0000]
    STEP_10_9 -> 'step(10,9)' [1.0000]
    STEP_9_3 -> 'step(9,3)' [1.0000]
    STEP_9_4 -> 'step(9,4)' [1.0000]
    STEP_0_1 -> 'step(0,1)' [1.0000]
    STEP_11_13 -> 'step(11,13)' [1.0000]
    STEP_9_19 -> 'step(9,19)' [1.0000]
    STEP_8_9 -> 'step(8,9)' [1.0000]
    STEP_2_0 -> 'step(2,0)' [1.0000]
    STEP_3_2 -> 'step(3,2)' [1.0000]
    STEP_9_9 -> 'step(9,9)' [1.0000]
    STEP_13_14 -> 'step(13,14)' [1.0000]
    STEP_9_10 -> 'step(9,10)' [1.0000]
    STEP_11_12 -> 'step(11,12)' [1.0000]
    STEP_11_10 -> 'step(11,10)' [1.0000]
    STEP_5_3 -> 'step(5,3)' [1.0000]
""")
def sample_plans(grammar, n=50):
    return list(generate(grammar, n=n, depth=100))

plans = sample_plans(rule_grammar)
print("Sampled plans:")
if not plans:
    print("  (No plans generated)")
else:
    for i, plan in enumerate(plans):
        print(f"  Plan {i+1}: {' '.join(plan)}")

def mutate_plan(plan):
    if not plan:
        return plan
    mutated = plan[:]
    swap_map = {"step(0,1)": "step(1,2)", "step(1,2)": "step(0,1)"}
    idx = random.randint(0, len(plan) - 1)
    if mutated[idx] in swap_map:
        mutated[idx] = swap_map[mutated[idx]]
    return mutated

print("\nMutated versions:")
for i, plan in enumerate(plans):
    mplan = mutate_plan(plan)
    print(f"  Plan {i+1} mutated: {' '.join(mplan)}")
