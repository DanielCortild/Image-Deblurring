# Image Deblurring based on the ROF Model using a Primal-Dual and a Three Operator Splitting Method

The aim of this notebook is to deblur images according to the Model suggested by Rudin, Osher and Fatemi. We focus on a Primal-Dual method suggested by Chomboll and Pock (2010) and on a Three Operator Splitting Method suggested by Davis and Yin (2017).


```python
# Import Libraries
from PIL import Image, ImageOps
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
```

## Importing the Image
Note that a black-and-white image is basically just considered to be a 2-dimensional array of integers between $0$ (Representing black) and $255$ (Representing white). Although also possible, we shall not rescale the values to be between $0$ and $1$, for purely notational simplicity later on, but analogous reasoning would hold if we do so.

Here we import a small cute picture of a cat, and resize it to a "reasonable" size, although the algorithms will work (Although slower) for any size. We also implement a class to show images, with more functionality than required at the moment, but which will come in handy later.


```python
# Load the image, Resize it, Grayscale it, and convert it to an array of pixels
img = Image.open('cat.jpeg').resize((256, 256))
X_ref = np.asarray(ImageOps.grayscale(img)).astype('float32') 

# Class to plot images
class Plot:
    def __init__(self, items, title, dpi=300):
        self.i = -1
        self.j = 0
        self.h = (items + 3) // 4
        self.w = 4
        if self.h == 1:
            self.i = -1
            self.j = 0
        else:
            self.i = 0
            self.j = -1
        self.fig, self.axs = plt.subplots(self.h, self.w, figsize=self.__get_fig_dims(), dpi=dpi)
        self.get_next = self.__get_get_next()
        self.fig.suptitle(title, fontsize=18)
        
    def plot(self, img, title=None):
        ax = self.get_next()
        if title:
            ax.title.set_text(title)
        ax.imshow(img, cmap=cm.Greys_r, vmin=0, vmax=255)
        ax.set_axis_off()

    def __get_fig_dims(self):
        if self.h == 1:
            return (10, 3)
        elif self.h == 2:
            return (14, 10)
        elif self.h == 3:
            return (12,10)
        elif self.h == 4:
            return (8, 10)
        elif self.h == 5:
            return (12,15)
        else:
            return (10, 10)
        
    def __get_get_next(self):
        if self.h == 1 and self.w == 1:
            __get_next = lambda: plt
        elif self.h == 1:
            def __get_next():
                self.i += 1
                if self.i == self.w:
                    raise Exception("Plot is not large enough, please adapt Width or add Height")
                return self.axs[self.i]
        elif self.h > 1:
            def __get_next():
                self.j += 1
                if self.j == self.w:
                    self.j = 0
                    self.i += 1
                if self.i == self.h:
                    raise Exception("Plot is not large enough, please adapt Height and Width")
                return self.axs[self.i, self.j]
        else:
            __get_next = lambda: None
        return __get_next
            
    def plot_blank(self):
        self.plot(np.zeros((256, 256)) + 255)
        
    def show(self):
        plt.tight_layout()
        while (self.i != self.h - 1 and self.j != self.w - 1 and self.h != 1) or \
                (self.i != self.w - 1 and self.h == 1):
            self.plot_blank()
        plt.show()

# Show the image
P = Plot(1, "Images")
P.plot_blank()
P.plot(X_ref, "Reference Image")
P.show()
```


    
![png](outputs/output_3_0.png)
    


## Blurring Process
The original image $X_{ref}$ has been blurred into the image $X_{blur}=R(X_{ref})+n$, where $R$ is a blur operator and $n$ is Gaussian white noise with a zero-mean. The blur operator $R$ is considered a black box for the purpose of this notebook, but it is worth noting that it is a convoltuion hence it is a linear operator. 


```python
# Black Box, Not Important how it works
def blackbox():
    s = 4
    size_filter = size = 9
    PSF = np.array([[np.exp(-0.5*((i-4)/s)**2 - 0.5*((j-4)/s)**2) for j in range(size)] for i in range(size)])
    PSF /= np.sum(PSF)
    def dctshift(PSF, center):
        m, n = PSF.shape
        i, j = center
        l = min(i, m-i-1, j, n-j-1)
        PP = PSF[i-l:i+l+1, j-l:j+l+1]
        Z1 = np.diag(np.ones(l+1), k=l  )
        Z2 = np.diag(np.ones(l)  , k=l+1)
        PP = Z1 @ PP @ Z1.T + Z1 @ PP @ Z2.T + Z2 @ PP @ Z1.T + Z2 @ PP @ Z2.T
        Ps = np.zeros_like(PSF)
        Ps[0:2*l+1, 0:2*l+1] = PP
        return Ps
    dct2 = lambda a: dct(dct(a.T, norm='ortho').T, norm='ortho')
    idct2 = lambda a:  idct(idct(a.T,norm='ortho').T,norm='ortho')    
    Pbig = np.zeros_like(X_ref)
    Pbig[0:size_filter , 0:size_filter] = PSF
    e1 = np.zeros_like(X_ref)
    e1[0][0] = 1
    S = np.divide( dct2(dctshift(Pbig, (4,4))), dct2(e1) )
    R = R_T = lambda X, S_matrix=S: idct2( np.multiply(S_matrix, dct2(X)) )
    return R, R_T

R, R_T = blackbox()

# Blur Image
np.random.seed(10)
n = np.random.normal(0, 0.5, size=X_ref.shape)
X_blur = R(X_ref) + n

# Show the image
P = Plot(1, "Images")
P.plot_blank()
P.plot(X_ref, "Reference Image")
P.plot(X_blur, "Blurred Image")
P.show()
```


    
![png](outputs/output_5_0.png)
    


## Primal-Dual Method

We write the problem as a Total Variation problem, using the $1$-norm of the discrete graident as penalty function according to the ROF Model. Notice that instead of having the $\lambda$ before the first term, we could have a factor $r$ in front of the second term, representing its inverse. This would be analogous of course.
$$\min_{X\in \mathbb{R}^{256\times 256}}\left\{\frac\lambda2 \|R(X)-X_{blur}\|^2_2+\|\text{grad}(X)\|_1\right\}$$
Where $R$ is the blur operator, $X_{blur}$ is the blurred (and noisy) image, and $\text{grad}(X)$ is the discrete gradient of $X$ (Which is a linear operator). Of course all matrix-norms are vector induced. 

Now we denote $f(X)=\frac\lambda2\|R(X)-X_{blur}\|_2^2$ and $g(X)=\|X\|_1$, both being lower-semicontinuous and convex, and $L=\text{grad}$, to write the above problem as
$$\min_{X\in \mathbb{R}^{256\times 256}}\left\{f(X)+g(LX)\right\}$$

We may apply the Algorithm suggested by Chambolle and Pock to get
$$\begin{cases}
X_{k+1} &= (I+\tau \partial f)^{-1}(X_k-\tau L^*Y_k) \\
Y_{k+1} &= (I+\sigma \partial g^*)^{-1}(Y_k-\sigma L(2X_{k+1}-X_k)) \\
\end{cases}$$
Where $L^*$ is the adjoint operator of $L$, implemented right below. We also introduced variables $\sigma$ and $\tau$, which verify $\sigma\tau \|L\|^2\le 1$, to ensure convergence. It can be proven (Although not done here), that $\|L\|^2\le 8$. 


```python
# A Parameter to be set and Used later
norm_L = np.sqrt(8)

# The Discrete Gradient Linear Operator
# grad: R^{256x256} -> R^{2x256x256}
def grad(X):
    G = np.zeros_like([X, X])
    G[0, :, :-1] = X[:, 1:] - X[:, :-1] # Horizontal Direction
    G[1, :-1, :] = X[1:, :] - X[:-1, :] # Vertical Direction
    return G

# The Adjoint of the Discrete Gradient Linear Operator, the Discrete Divergence
# grad_T: R^{2x256x256} -> R^{256x256}
def grad_T(Y):
    G_T = np.zeros_like(Y[0])

    G_T[:, :-1] += Y[0, :, :-1] # Corresponds to c[0]
    G_T[:-1, :] += Y[1, :-1, :] # Corresponds to c[1]
    G_T[:, 1:] -= Y[0, :, :-1] # Corresponds to c[0]
    G_T[1:, :] -= Y[1, :-1, :] # Corresponds to c[1]

    return G_T
```

We of course also require the proximal operators of $f$ and of $g^*$, which are the following (The computations are not made here)
$$(I+\tau \partial f)^{-1}(X)=(I+R^TR)^{-1}\left(X + \tau\cdot \lambda\cdot R^T(X_{blur})\right)$$
$$(I+\sigma \partial g^*)^{-1}(Y)^{(k)}_{i,j}=\frac{Y^{(k)}_{i,j}}{\max\left(1, \sqrt{(Y^{(1)}_{i,j})^2 + (Y^{(2)}_{i,j})^2}\right)}$$
Where $i,j\in \{0, \ldots, 255\}$ and $k\in \{0,1\}$

Now everything is set to apply the algorithm. 


```python
def run_PD(X_blur, lambdas, measured_its, tau, sigma):
    """
    Runs the Primal Dual Method to solve the ROF problem 
          min_{X in R^{256x256}} lambda/2 * |R(X)-X_blur|_2^2 + |grad(X)|_1
    Using the Chambolle-Pock Algorithm
    
    Parameters:
        X_blur:         Blurry Image
        lambdas:        List of tested values of lambda
        measured_its:   List of tested values of number of iterations, in increasing order
        tau:            Value of tau
        sigma:          Value of sigma
    Returns:
        X_PD:           List of dimension len(lambdas) x len(measured_its) x 256 x 256
                        containing the deblurred images for each lambda and it
    """
    
    # Proximal of g^*
    def prox_gstar(Y):
        max_y = np.maximum(1.0, np.sqrt((Y ** 2).sum(-1)))
        return Y / max_y[..., None]

    # Proximal of f
    def prox_f(X):
        IRR = lambda X: X + tau * lamb * R_T(R(X.reshape((256, 256)))).flatten()
        LinOpIRR = sp.sparse.linalg.LinearOperator((65536, 65536), matvec=IRR, rmatvec=IRR)
        RHS = (X + tau * lamb * R_T(X_blur)).flatten()
        return sp.sparse.linalg.cg(LinOpIRR, RHS)[0].reshape((256, 256))
    
    # Set Variables X_PD
    X_PD = np.zeros((len(lambdas), len(measured_its), 256, 256))
    
    # Loop over every value of lambda
    for lamb_i, lamb in enumerate(lambdas):

        # Initial Values
        X0 = X_blur
        Y0 = grad(X0)

        # Update the values
        def update(X0, Y0):
            Xnew = prox_f(X0 - tau * grad_T(Y0))
            Ynew = prox_gstar(Y0 + sigma * grad(2 * Xnew - X0))
            return Xnew, Ynew

        # Run the iterations
        for i in tqdm(range(measured_its[-1])):
            X0, Y0 = update(X0, Y0)
            
            # Stores the image if it is supposed to be a measured iteration
            if i+1 in measured_its:
                X_PD[lamb_i, measured_its.index(i+1)] = X0
                
    return X_PD
```

Now we can run the algorithm. We run the iterations for $\lambda\in \{4,8,16, 32\}$, with up to $1000$ iterations. We shall then select the best value of $\lambda$ and iterate even more afterwards.


```python
# Some parameters
lambdas = [2, 4, 8, 16, 32, 64, 128, 256] # Different values of lambda
tau = 0.01
sigma = 1 / (tau * norm_L ** 2)
measured_its = [1000, 5000] # It takes around 1 second to compute 30 iterations

X_PD = run_PD(X_blur, lambdas, measured_its, tau, sigma)
```

    100%|██████████| 5000/5000 [02:18<00:00, 36.20it/s]
    100%|██████████| 5000/5000 [02:37<00:00, 31.68it/s]
    100%|██████████| 5000/5000 [02:36<00:00, 31.98it/s]
    100%|██████████| 5000/5000 [02:58<00:00, 28.04it/s]
    100%|██████████| 5000/5000 [02:58<00:00, 28.07it/s]
    100%|██████████| 5000/5000 [03:17<00:00, 25.35it/s]
    100%|██████████| 5000/5000 [04:00<00:00, 20.77it/s]
    100%|██████████| 5000/5000 [04:24<00:00, 18.93it/s]


We observe that the iterations take a lot longer to run for larger values of $\lambda$, so we will not increase $\lambda$ although we theoretically could.

We can then plot the results obtained above. We shall plot the image with a 1200 DPI, meaning it might take an additional 30 seconds to plot.


```python
P = Plot(4 + len(lambdas)*len(measured_its), "Denoising by Primal-Dual Method", dpi=1200)

P.plot_blank()
P.plot(X_ref, "Reference Image")
P.plot(X_blur, "Blurred Image")
P.plot_blank()

for lamb_i, it_i in np.ndindex(len(lambdas), len(measured_its)):
    P.plot(X_PD[lamb_i, it_i, :, :], f"$\lambda$={lambdas[lamb_i]}, it={measured_its[it_i]}")
    
P.show()
```


    
![png](outputs/output_14_0.png)
    


We could of course run this experiment for more values of $\lambda$ and even more iterations, which would probably yield an even better result, and the reader is invited to do so if times permits it.

## Three Operator Splitting

Consider the minimization problem

$$\min_{X\in \mathbb{R}^{N\times N}}{\frac{\lambda}{2}\lVert R(X)-X_{blur}\rVert^2 + \lVert \text{grad}(X) \rVert_1 }$$

where $R$ is the blurring operator, $X_{blur}$ is the blurred image and $\text{grad}$ is a block diagonal matrix of discrete gradients. To deal with the regularization term, we introduce a variable $Y$ to obtain


$$\min_{X,Y \in \mathbb{R}^{N\times N}\times \mathbb{R}^{2\times N\times N}}{\frac{\lambda}{2}\lVert R(X)-X_{blur}\rVert^2 + \lVert Y \rVert_1 + \iota_{\{\text{grad}(X)\}}(Y)}$$

where $\iota_{\{\text{grad}(X)\}}$ is an indicator function. Instead of minimizing over two variables $X$ and $Y$, we combine them into a new variable in $\mathbb{R}^{ N\times N}\times \mathbb{R}^{2\times N\times N}\cong \mathbb{R}^{3\times  N\times N}$ and adapt the three operators accordingly. In particular, we have


$$\min_{Z\in \mathbb{R}^{3\times N\times N}}{\frac{\lambda}{2}\lVert R'(Z)-Z_{blur}\rVert^2 + n(Z) + I_{\text{grad}}(Z)}$$

where for $Z = \begin{pmatrix}Z_1 \\ Z_2 \end{pmatrix} \in \mathbb{R}^{3\times N\times N}$ we define

$$
\begin{aligned}
&R'(Z) = \begin{pmatrix}R(Z_1)\\ 0\end{pmatrix}, \ \ \
Z_{blur} = \begin{pmatrix}X_{blur}\\ 0 \end{pmatrix}\\
&n(Z) =\lVert Z_2 \rVert_1, \ \ \ 
I_{\text{grad}}(Z) = \begin{cases}0 & \text{ if }DZ_1 = Z_2\\ \infty  & \text{ else} \end{cases}
\end{aligned}
$$


Define $f(Z) = n(Z), g(Z) = I_{\text{grad}}(Z)$ and $h(Z) =\frac{\lambda}{2}\lVert R'(Z)-Z_{blur}\rVert^2$ 

The proximal operator of $f$ is given by

$$\text{prox}_{\tau f}(Z) = \begin{pmatrix}Z_1\\ shrink(Z_2,\tau)\end{pmatrix}$$

Moreover, the proximal operator of $g$ is given by

$$\text{prox}_{\tau g}(Z) = \begin{pmatrix}Z_1 \\ \text{grad}(Z_1) \end{pmatrix}$$

Lastly, since the blurring operator is self-adjoint, the gradient of $h$ is given by

$$\nabla h(Z) = \frac\lambda2\begin{pmatrix}(R')^T(R'(Z_1) - Z_{blur})\\ 0\end{pmatrix}$$

Now, we can apply the Davis-Yin Three Operator Splitting algorithm: 

Initialize $C_0$ and iterate

$$\begin{cases}
A_k = \text{prox}_{\tau f}C_k\\
B_k = \text{prox}_{\tau g}(2A_k-C_k - \tau \nabla h(A_k))\\
C_{k+1} = C_k + B_k - A_k
\end{cases}$$


```python
def run_TOS(X_blur, lambdas, measured_its, tau, sigma):
    """
    Runs the Three Operator Splitting Method to solve the ROF problem 
          min_{X,Y in R^{256x256}xR^{2x256x256}} lambda/2 * |R(X)-X_blur|_2^2 + |Y|_1 + i_{DX}(Y)
    Using the Davis-Yin Algorithm
    
    Parameters:
        X_blur:         Blurry Image
        lambdas:        List of tested values of lambda
        measured_its:   List of tested values of number of iterations, in increasing order
        tau:            Value of tau
        sigma:          Value of sigma
    Returns:
        X_TOS:          List of dimension len(lambdas) x len(measured_its) x 256 x 256
                        containing the deblurred images for each lambda and it
    """
    
    # Compute the Proximals and Gradients Required
    def prox_f(Z, tau):
        Y = Z[1:, :, :]
        Z[1:, :, :] = np.sign(Y) * np.maximum(abs(Y)-tau, 0)
        return Z

    def prox_g(Z):
        X = Z[0, :, :]
        Z[1:, :, :] = grad(X)
        return Z

    def grad_h(Z, lamb):
        X = Z[0, :, :]
        Z[0, :, :] = lamb/2 * R_T(R(X) - X_blur)
        return Z
    
    # Set Variables X_TOS
    X_TOS = np.zeros((len(lambdas), len(measured_its), 256, 256))
    
    # Loop over every value of lambda
    for lamb_i, lamb in enumerate(lambdas):

        # Initial Values
        C0 = np.array([X_blur, grad(X_blur)[0], grad(X_blur)[1]])

        # Update the values
        def update(C0):
            Anew = prox_f(C0, tau)
            Bnew = prox_g(2 * Anew - C0 - tau * grad_h(Anew, lamb))
            Cnew = C0 + Bnew - Anew
            return Cnew

        # Run the iterations
        for i in tqdm(range(measured_its[-1])):
            C0 = update(C0)
            
            # Stores the image if it is supposed to be a measured iteration
            if i+1 in measured_its:
                X_TOS[lamb_i, measured_its.index(i+1)] = C0[0]
                
    return X_TOS
```

Now we may run the algorithm. As before, we run the iterations for $\lambda\in \{4,8,16,32,64,128,256\}$, with up to $5000$ iterations. We shall then select the best value of $\lambda$ and iterate even more afterwards.


```python
# Some parameters
lambdas =  [2, 4, 8, 16, 32, 64, 128, 256] # Different values of lambda
tau = 0.01
sigma = 1 / (tau * norm_L ** 2)
measured_its = [2500, 10000] # It takes around 1 second to compute 230 iterations

X_TOS = run_TOS(X_blur, lambdas, measured_its, tau, sigma)
```

    100%|██████████| 10000/10000 [01:03<00:00, 156.57it/s]
    100%|██████████| 10000/10000 [01:07<00:00, 147.56it/s]
    100%|██████████| 10000/10000 [01:01<00:00, 163.65it/s]
    100%|██████████| 10000/10000 [01:04<00:00, 156.06it/s]
    100%|██████████| 10000/10000 [01:09<00:00, 143.20it/s]
    100%|██████████| 10000/10000 [01:10<00:00, 142.01it/s]
    100%|██████████| 10000/10000 [01:15<00:00, 131.66it/s]
    100%|██████████| 10000/10000 [01:19<00:00, 125.32it/s]


It is interesting to notice how much faster this method is compared to the previous one. 

And we may obviously plot the solution. We plot it with a $1200$ DPI, meaning the image will take about half a minute to plot.


```python
P = Plot(4 + len(lambdas)*len(measured_its), "Denoising by Three Operator Splitting Method", dpi=1200)

P.plot_blank()
P.plot(X_ref, "Reference Image")
P.plot(X_blur, "Blurred Image")
P.plot_blank()

for lamb_i, it_i in np.ndindex(len(lambdas), len(measured_its)):
    P.plot(X_TOS[lamb_i, it_i, :, :], f"$\lambda$={lambdas[lamb_i]}, it={measured_its[it_i]}")
    
P.show()
```


    
![png](outputs/output_23_0.png)
    


Again, with more computational power and more time, one could perfect these images even more. 

## Conclusions

For both the explored methods we see (As predicated) that the more iterations the more clear of a result. We also observe that as $\lambda$ inscreases, the more noise there is present on the deblurred image. This was of course also expected due to the definition of the problem. 

To conclude the notebook, we may plot the best version of both methods, as follows. Of course, they could be improved by testing other values throughout the note book.


```python
P = Plot(4, "Denoising by Primal-Dual and by Three Operator Splitting", dpi=1200)

P.plot(X_ref, "Reference Image")
P.plot(X_blur, "Blurred Image")
P.plot(X_PD[5, 1], "Deblurred by PM")
P.plot(X_TOS[6, 1], "Deblurred by TOS")

P.show()
```


    
![png](outputs/output_26_0.png)
