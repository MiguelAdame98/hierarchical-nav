from nltk import PCFG, Nonterminal
from nltk.parse.generate import generate
import random
grammar = PCFG.fromstring("""
  EXPLORE -> NAVPLAN [1.0]

  # now a softmax over 10 choices
  NAVPLAN -> GOTO_0 [0.25]
  NAVPLAN -> GOTO_1 [0.15]
  NAVPLAN -> GOTO_2 [0.15]
  NAVPLAN -> GOTO_3 [0.10]
  NAVPLAN -> GOTO_4 [0.10]
  NAVPLAN -> GOTO_5 [0.05]
  NAVPLAN -> GOTO_6 [0.05]
  NAVPLAN -> GOTO_7 [0.05]
  NAVPLAN -> GOTO_8 [0.05]
  NAVPLAN -> GOTO_9 [0.05]

  GOTO_0 -> MOVESEQ_9_0 [1.0]
  GOTO_1 -> MOVESEQ_9_1 [1.0]
  GOTO_2 -> MOVESEQ_9_2 [1.0]
  GOTO_3 -> MOVESEQ_9_3 [1.0]
  GOTO_4 -> MOVESEQ_9_4 [1.0]
  GOTO_5 -> MOVESEQ_9_5 [1.0]
  GOTO_6 -> MOVESEQ_9_6 [1.0]
  GOTO_7 -> MOVESEQ_9_7 [1.0]
  GOTO_8 -> MOVESEQ_9_8 [1.0]
  GOTO_9 -> MOVESEQ_9_9 [1.0]

  MOVESEQ_9_0 -> HOPSEQ_9_0 [1.0]
  MOVESEQ_9_1 -> HOPSEQ_9_1 [1.0]
  MOVESEQ_9_2 -> HOPSEQ_9_2 [1.0]
  MOVESEQ_9_3 -> HOPSEQ_9_3 [1.0]
  MOVESEQ_9_4 -> HOPSEQ_9_4 [1.0]
  MOVESEQ_9_5 -> HOPSEQ_9_5 [1.0]
  MOVESEQ_9_6 -> HOPSEQ_9_6 [1.0]
  MOVESEQ_9_7 -> HOPSEQ_9_7 [1.0]
  MOVESEQ_9_8 -> HOPSEQ_9_8 [1.0]
  MOVESEQ_9_9 -> HOPSEQ_9_9 [1.0]

  HOPSEQ_9_0 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_3 STEP_3_2 STEP_2_1 STEP_1_0 [1.0]
  HOPSEQ_9_1 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_3 STEP_3_2 STEP_2_1 [1.0]
  HOPSEQ_9_2 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_3 STEP_3_2 [1.0]
  HOPSEQ_9_3 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_3 [1.0]
  HOPSEQ_9_4 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 [1.0]
  HOPSEQ_9_5 -> STEP_9_9 STEP_9_8 STEP_8_5 [1.0]
  HOPSEQ_9_6 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_6 [1.0]
  HOPSEQ_9_7 -> STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_7 [1.0]
  HOPSEQ_9_8 -> STEP_9_9 STEP_9_8 [1.0]
  HOPSEQ_9_9 -> STEP_9_9 [1.0]

  STEP_9_9 -> 'right' 'right' 'forward' 'forward' 'forward' 'forward' 'forward' 'left' [1.0]
  STEP_9_8 -> 'right' 'right' 'forward' 'forward' 'forward' 'forward' 'left' 'left' [1.0]
  STEP_8_5 -> 'right' 'right' 'forward' 'forward' 'forward' 'forward' 'forward' 'left' 'left' [1.0]
  STEP_5_4 -> 'forward' 'forward' 'forward' 'right' 'forward' 'left' 'forward' 'left' 'left' [1.0]
  STEP_4_3 -> 'right' 'right' 'forward' 'forward' 'forward' 'forward' 'forward' 'forward' 'forward' 'left' 'forward' 'left' [1.0]
  STEP_3_2 -> 'right' 'forward' 'right' 'forward' 'forward' 'forward' 'left' 'forward' 'forward' 'left' 'left' [1.0]
  STEP_2_1 -> 'right' 'right' 'forward' 'forward' 'forward' 'forward' 'forward' 'forward' 'forward' 'forward' 'left' 'left' [1.0]
  STEP_1_0 -> 'left' 'left' 'forward' 'forward' 'forward' 'forward' 'forward' [1.0]
""")
'''def sample_plans(grammar, n=50):
    return list(generate(grammar, n=n, depth=100))

plans = sample_plans(grammar)
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
    print(f"  Plan {i+1} mutated: {' '.join(mplan)}")'''


from collections import deque


from collections import deque, namedtuple
import heapq
import math

def beam_enumerate(grammar, beam_width=5, max_steps=1000):
    """
    Enumerate up to `beam_width` complete derivations,
    filtering out zero-probability productions.
    Returns a list of (joint_prob, terminal_list).
    """
    # each beam entry is (–log_prob, derivation_list)
    beam = [(0.0, [grammar.start()])]
    completed = []

    for _ in range(max_steps):
        if not beam or len(completed) >= beam_width:
            break
        next_beam = []

        # Expand each partial derivation in the beam
        for score, deriv in beam:
            # If fully expanded, collect as complete
            if all(not isinstance(sym, Nonterminal) for sym in deriv):
                completed.append((math.exp(-score), deriv))
                continue

            # find leftmost nonterminal
            idx, nt = next((i,s) for i,s in enumerate(deriv)
                           if isinstance(s, Nonterminal))

            # expand only positive-prob productions
            for prod in grammar.productions(lhs=nt):
                p = prod.prob()
                if p <= 0.0:
                    continue
                new_score = score - math.log(p)
                new_der   = deriv[:idx] + list(prod.rhs()) + deriv[idx+1:]
                next_beam.append((new_score, new_der))

        # keep top beam_width by score
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    # if not enough completed, drain any fully-expanded from the beam
    for score, deriv in beam:
        if all(not isinstance(sym, Nonterminal) for sym in deriv):
            completed.append((math.exp(-score), deriv))
    return completed[:beam_width]
# 3) Sample one plan from those top-K
def sample_from_beam(candidates):
    probs, plans = zip(*candidates)

    total   = sum(probs)
    weights = [p/total for p in probs]
    choice  = random.choices(plans, weights=weights, k=1)[0]
    print("Sampled plan:", " ".join(choice))
    return random.choices(plans, weights)[0]


# 2) Pure PCFG sampler (for proposals)
def sample_from_pcfg(grammar, symbol=None):
    from nltk import Nonterminal
    if symbol is None:
        symbol = grammar.start()
    if not isinstance(symbol, Nonterminal):
        return [symbol]
    prods = grammar.productions(lhs=symbol)
    probs = [p.prob() for p in prods]
    chosen = random.choices(prods, weights=probs)[0]
    result = []
    for sym in chosen.rhs():
        result += sample_from_pcfg(grammar, sym)
    return result

# 3) Compute plan probability under grammar
def plan_prob(plan):
    # For a full tree, the joint prob is ∏ rule_probs, but reconstructing
    # that tree is complex. Here, since each unique plan maps to exactly
    # one derivation, we approximate by counting rules used.
    # *In your case all rules are prob=1, so every plan has equal mass.*
    return 1.0

# 4) Single-site subtree re-sampling proposal
def propose(current):
    # pick a random nonterminal position in the derivation tree;
    # here, we cheat and re-sample the *entire* plan
    return sample_from_pcfg(grammar)

# 5) MCMC loop
def mcmc_plans(grammar, burn_in=50, sample_gap=10, n_samples=5):
    current = sample_from_pcfg(grammar)
    samples = []
    # burn-in
    for _ in range(burn_in):
        prop = propose(current)
        # MH acceptance (here trivially always accept since plan_prob is constant)
        current = prop
    # then collect
    for _ in range(n_samples):
        for __ in range(sample_gap):
            prop = propose(current)
            if random.random() < plan_prob(prop)/plan_prob(current):
                current = prop
        samples.append(current)
    return samples
# 2) Temperature-aware choice
def tempered_choice(prods, T=1.0):
    ps = [p.prob() for p in prods]
    # raise to 1/T
    adj = [p**(1.0/T) for p in ps]
    tot = sum(adj)
    weights = [a/tot for a in adj]
    return random.choices(prods, weights=weights, k=1)[0]

# 3) Top-down sampler with temperature
def sample_with_temperature(grammar, symbol=None, T=1.0, max_depth=50):
    from nltk import Nonterminal
    if symbol is None:
        symbol = grammar.start()
    if not isinstance(symbol, Nonterminal) or max_depth<=0:
        return [symbol] if not isinstance(symbol, Nonterminal) else []
    prods = grammar.productions(lhs=symbol)
    chosen = tempered_choice(prods, T)
    result = []
    for sym in chosen.rhs():
        result += sample_with_temperature(grammar, sym, T, max_depth-1)
    return result
# 4) Run an example
if __name__ == "__main__":
    candidates = beam_enumerate(grammar, beam_width=3)
    print("Top-3 candidates (prob, plan):")
    for p, plan in candidates:
        print(f"  {p:.4f} -> {' '.join(plan)}")

    print("\nSampled plan:", ' '.join(sample_from_beam(candidates)))

    plans = mcmc_plans(grammar, burn_in=100, sample_gap=20, n_samples=5)
    for i, p in enumerate(plans,1):
        print(f"Sample {i}: {' '.join(p)}")

    for T in [0.5, 1.0, 2.0]:
        print(f"\n--- Temperature T={T} ---")
        for i in range(3):
            plan = sample_with_temperature(grammar, T=T)
            print(f"Plan {i+1}:", ' '.join(plan))


    