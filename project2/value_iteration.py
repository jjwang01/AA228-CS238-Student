import numpy as np
import pandas as pd
import sys

"""
Use a full update.
"""

S, A, gamma = None, None, None

def initialize(sars_):
    N = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for idx, row in sars_.iterrows():
        s, a, r, s_ = row['s'], row['a'], row['r'], row['sp']
        N[s-1,a-1,s_-1] += 1
        R[s-1,a-1] += r
    
    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            n = np.sum(N[s,a])
            R[s,a] = R[s,a].astype(float) / n
            for s_ in range(S):
                T[s,a,s_] = N[s,a,s_].astype(float) / n
    
    return N, T, R
            

def lookahead(N, T, R, k_max):
    U = [(0,0) for _ in range(S)]
    Up = [(0,0) for _ in range(S)]
    for _ in k_max:
        U, Up = Up, U
        for s in range(S):
            results = []
            for a in range(A):
                results.append((a, R[s,a] + gamma * sum([T[s,a,s_]*Up[s_][1] for s_ in range(S)])))
            U[s] = max(results, key=lambda x : x[1])
    return U


def output_policy(U, outfile):
    with open(outfile, "w") as f:
        for s in range(S):
            f.write(str(U[s][0]+1) + "\n")


def compute(infile, outfile, k_max):
    sars_ = pd.read_csv(infile)
    N, T, R = initialize(sars_)
    U = lookahead(N, T, R, k_max)
    output_policy(U, outfile)


def main():
    global S, A, gamma
    if len(sys.argv) != 3:
        print("Usage should be: value_iteration.py <file_type> <k_max>")
    
    if sys.argv[1] == 'small':
        inputfilename = 'data/small.csv'
        outputfilename = 'small.policy'
        S = 100
        A = 4
        gamma = 0.95
    elif sys.argv[1] == 'medium':
        inputfilename = 'data/medium.csv'
        outputfilename = 'medium.policy'
        S = 50000
        A = 7
        gamma = 1
    elif sys.argv[1] == 'large':
        inputfilename = 'data/large.csv'
        outputfilename = 'large.policy'
        S = 312020
        A = 9
        gamma = 0.95
    else:
        print("No specified file type")
    
    k_max = int(sys.argv[2])
    compute(inputfilename, outputfilename, k_max)


if __name__ == "__main__":
    main()