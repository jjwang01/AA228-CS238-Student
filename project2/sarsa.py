import numpy as np
import pandas as pd
import sys

S, A, gamma, alpha, lambda_ = None, None, None, None, None

def initialize():
    Q = np.zeros((S,A))
    N = np.zeros((S,A))
    return Q, N


def update_model(sars_, Q, N):
    l = None
    for idx, row in sars_.iterrows():
        s, a, r, s_ = row['s']-1, row['a']-1, row['r']-1, row['sp']-1
        if l != None:
            N[l[0], l[1]] += 1
            delta = r + gamma * Q[s,a] - Q[l[0], l[1]]
            for s__ in range(S):
                for a__ in range(A):
                    Q[s__][a__] += alpha * delta * N[s__,a__]
                    N[s__][a__] *= gamma * lambda_
        else:
            N[:,:] = 0
        l = (s, a, r)
    return Q, N


def output_policy(Q, outfile):
    pi = np.argmax(Q, axis=-1)
    with open(file_name, "w") as f:
        for i, p in enumerate(P):
            f.write(str(p+1))
            if i != P.size - 1:
                f.write("\n")


def compute(infile, outfile, k_max):
    Q, N = initialize()
    sars_ = pd.read_csv(infile)
    for _ in range(k_max):
        Q_new, N_new = update_model(sars_, Q, N)
        if np.sum(Q_new - Q) <= 1e-1:
            break
        Q, N = Q_new, N_new
    output_policy(Q, outfile)


def main():
    global S, A, gamma, alpha, lambda_
    if len(sys.argv) != 5:
        print("Usage should be: sarsa.py <file_type> <alpha> <lambda_> <k_max>")
    
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

    alpha = float(sys.argv[2])
    lambda_ = float(sys.argv[3])
    k_max = int(sys.argv[4])
    compute(inputfilename, outputfilename, k_max)

if __name__ == "__main__":
    main()
