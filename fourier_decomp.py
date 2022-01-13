# Step 1 convert to Fourier modes 
# Step 2 collect each fourier mode in time
# Step 3 apply a DFT and see what happens!?


A = np.zeros((19,19,2*19*19))


c = 0
for i in range(19):
    for j in range(19):
        for m in range(19):
            for n in range(19):
                A[m, n, c] = np.sin((i+1)*m + (j+1)*n*2*np.pi/20)
                A[m, n, c+361] = np.cos((i+1)*m + (j+1)*n*2*np.pi/20)
        c+=1


A_reshape = A.reshape(361, 2*19*19)


pinv = np.linalg.pinv(A_reshape, rcond=1e-6)
b = test_img.flatten()

x = np.matmul(pinv, b)
approx = np.matmul(A_reshape, x)
