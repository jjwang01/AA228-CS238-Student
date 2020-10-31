import numpy as np
import pandas as pd
import sys

S, A, gamma, alpha = None, None, None, None

def initialize():
    Q = np.zeros((S,A))
    return Q

def update_model(Q, s, a, r, s_):
    prev_Q = Q.copy()
    Q[s,a] += alpha * (r + gamma * np.max(Q[s_,:]) - Q[s,a])
    print(np.sum(np.square(Q-prev_Q)))
    return Q

def output_policy(Q, outfile):
    pi = np.argmax(Q, axis=-1)
    with open(outfile, "w") as f:
        for p in pi:
            f.write(str(p+1)+"\n")

def compute(infile, outfile, k_max):
    sars_ = pd.read_csv(infile)
    Q = initialize()
    for k in range(k_max):
        for idx, row in sars_.iterrows():
            s, a, r, s_ = row['s']-1, row['a']-1, row['r']-1, row['sp']-1
            Q = update_model(Q, s, a, r, s_)
    output_policy(Q, outfile)

def main():
    global S, A, gamma, alpha
    if len(sys.argv) != 4:
        print("Usage should be: qlearning.py <file_type> <alpha> <k_max>")
    
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
    k_max = int(sys.argv[3])
    compute(inputfilename, outputfilename, k_max)

if __name__ == "__main__":
    main()