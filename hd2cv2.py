from time import time
import pyldpc
import numpy as np
from scipy.sparse import coo_matrix


def hd2cv2(HD, M):
    (b, c) = HD.shape
    (r, n) = (b * M, c * M)
    col_weights = np.zeros(n, dtype=np.int)
    row_weights = np.zeros(r, dtype=np.int)
    col_max_weight = max(col[col >= 0].sum() for col in HD.T)
    row_max_weight = max(row[row >= 0].sum() for row in HD)
    rows_in_col = np.zeros((n, col_max_weight))
    cols_in_row = np.zeros((r, row_max_weight))
    cols_idx_in_row = np.zeros((r, row_max_weight))
    H = np.zeros((r, n))

    for i in range(b):
        for j in range(c):
            if HD[i, j] >= 0:
                for h in range(M):
                    row_idx = i * M + h
                    col_idx = j * M + (HD[i, j] + h) % M
                    rows_in_col[col_idx][col_weights[col_idx]] = row_idx
                    cols_in_row[row_idx][row_weights[row_idx]] = col_idx
                    cols_idx_in_row[row_idx][row_weights[row_idx]] = col_weights[col_idx]
                    row_weights[row_idx] += 1
                    col_weights[col_idx] += 1
                    H[row_idx, col_idx] = 1

    return H

base98 = np.array([
    [0, 8, 3, 5],
    [0, 4, 6, 7],
    [0, 0, 0, 0]
])

base100 = np.array([
    [0, 2, 5, 6],
    [0, 4, 1, 3],
    [0, 0, 0, 0]
])

M = 9
H98 = hd2cv2(base98, M) # Sucessful decoding rate for snr = 4 and max_iter = 15 is 99.79%
#Sucessful decoding rate for snr = 4 and max_iter = 50 is 99.92%
H100 = hd2cv2(base100, M) # Sucessful decoding rate for snr = 4 and max_iter = 15 is 99.84%
#Sucessful decoding rate for snr = 4 and max_iter = 50 is 99.9%

base564 = np.array([
    [0, 8, 20, 11, 14, 9, 6],
    [0, 5, 7, 13, 4, 12, 15],
    [0, 0, 0, 0, 0, 0, 0]
])

base522 = np.array([
    [0, 1, 15, 2, 19, 7, 12],
    [0, 5, 9, 18, 8, 20, 3],
    [0, 0, 0, 0, 0, 0, 0]
])

M = 21
H564 = hd2cv2(base564, M)
#Sucessful decoding rate for snr = 4 and max_iter = 50 is 98.5%
#Sucessful decoding rate for snr = 4 and max_iter = 15 is 97.9%
H522 = hd2cv2(base522, M)
#Sucessful decoding rate for snr = 4 and max_iter = 50 is 99.5%
#Sucessful decoding rate for snr = 4 and max_iter = 15 is 98.1%

H = H522
k, n = H.shape
#tG = pyldpc.CodingMatrix(H)
print("k,n = ",(k,n))

print_matrix = True
with open("h522.mtx", "w") as f:
    f.write("coo_matrix\n")
    f.write(str(k) + " " + str(n) + "\n")
    cooH = coo_matrix(H)
    f.write(str(len(cooH.row)) + "\n")
    for i, j in zip(cooH.row, cooH.col):
        f.write(str(i) + " " + str(j) + "\n")

"""
sample_size = 1000
snr = 4
max_iter = 15
if 1:
    matches=[]
    t = time()
    x_sent = np.full((n,), 0)
    y_sent = pow(-1, x_sent)
    sigma = 10 ** (-snr / 20)
    for i in range(sample_size):
        e = np.random.normal(0, sigma, size=n)
        y = y_sent + e
        x_decoded = pyldpc.Decoding_logBP(H, y, snr, max_iter)
        matches.append((x_sent == x_decoded).all())
    t = time()-t
    print("Sucessful decoding rate for snr = {} and max_iter = {} is {}%".format(snr,max_iter,sum(matches)*100/sample_size))
    print("\nDecoding time of {} messages in seconds:".format(sample_size),t)
"""

