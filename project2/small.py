import numpy as np
import pandas as pd

"""
Use posterior sampling.
https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
"""

S = 100
A = 4

def initialize():
    # create a matrix for each (s, a) pair
    return np.full((S, A, S), 1)

def update_counts(sars_, transition_counts, k_max):
    for _ in k_max:
        rollouts = []
        for s in range(S):
            for a in range(A):
                s_ = np.random.choice(100, 1, p=transition_counts[s,a])
                rollouts.append(s, a, s_)
        
        # check if (s, a, s') exists in sars_, if not then ignore the update
        for (s, a, s_) in rollouts:
            try:
                reward = sars_.loc[(sars_['s'] == s+1) & (sars_['a'] == a+1) & (sars_['sp'] == s_+1)]['r']
            except:
                continue
            
            # update distribution (maybe add a penalty to the other s_'s?)
            transition_counts[s,a,s_] += reward
    return transition_counts


def output_policy(transition_counts, outfile):
    with open(outfile, 'w') as f:
        for s in range(S):
            # this time, take the greedy action in expectation
            max_EU, max_a = 0, 0
            for a in range(A):
                EU = 0
                for s_ in range(S):
                    try:
                        reward = sars_.loc[(sars_['s'] == s+1) & (sars_['a'] == a+1) & (sars_['sp'] == s_+1)]['r']
                    except:
                        continue
                    p = transition_counts[s,a,s_] / np.sum(transition_counts[s,a])
                    EU += p * reward
                if EU > max_EU:
                    max_EU = EU
                    max_a = a
            
            # write to file
            f.write(str(max_a) + '\n')
    

def compute(infile, outfile):
    transition_counts = initialize()
    sars_ = pd.read_csv(infile)
    transition_counts = update_counts(sars_, transition_counts, k_max)
    output_policy(transition_counts, outfile)


def main():
    inputfilename = 'data/small.csv'
    outputfilename = 'small.policy'
    compute(inputfilename, outputfilename)

if __name__ == "__main__":
    main()