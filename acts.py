import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm_notebook
import seaborn as sns
import imageio
import time
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

#data processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

#create a custom color map
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])

np.random.seed(0)

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
print(data.shape, labels.shape)

plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
plt.savefig("temp.jpg",dpi=1000)
#plt.show()

plt.style.use("ggplot")

labels_orig = labels
labels = np.mod(labels_orig, 2)

X_train, X_val, Y_train, Y_val = train_test_split(data, labels_orig, stratify=labels_orig, random_state=0)
print(X_train.shape, X_val.shape, labels_orig.shape)

enc = OneHotEncoder()
# 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
y_OH_train = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()
y_OH_val = enc.fit_transform(np.expand_dims(Y_val,1)).toarray()
print(y_OH_train.shape, y_OH_val.shape)

class FFNetwork:

    def __init__(self, init_method = 'random', activation_function = 'sigmoid', leaky_slope = 0.1):

        self.params={}
        self.params_h = []
        self.num_layers=2
        self.layer_sizes = [2, 2, 4]
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope

        np.random.seed(0)

        if init_method == "random":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])
                self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

        elif init_method == "he":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(2/self.layer_sizes[i-1])
                self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

        elif init_method == "xavier":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)]=np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(1/self.layer_sizes[i-1])
                self.params["B"+str(i)]=np.random.randn(1,self.layer_sizes[i])

        elif init_method == "zeros":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)]=np.zeros((self.layer_sizes[i-1],self.layer_sizes[i]))
                self.params["B"+str(i)]=np.zeros((1,self.layer_sizes[i]))

        self.gradients={}
        self.update_params={}
        self.prev_update_params={}
        for i in range(1,self.num_layers+1):
            self.update_params["v_w"+str(i)]=0
            self.update_params["v_b"+str(i)]=0
            self.update_params["m_b"+str(i)]=0
            self.update_params["m_w"+str(i)]=0
            self.prev_update_params["v_w"+str(i)]=0
            self.prev_update_params["v_b"+str(i)]=0

    def forward_activation(self, X):
        if self.activation_function == "sigmoid":
            return 1.0/(1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0,X)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_slope*X,X)

    def grad_activation(self, X):
        if self.activation_function == "sigmoid":
            return X*(1-X)
        elif self.activation_function == "tanh":
            return (1-np.square(X))
        elif self.activation_function == "relu":
            return 1.0*(X>0)
        elif self.activation_function == "leaky_relu":
            d=np.zeros_like(X)
            d[X<=0]=self.leaky_slope
            d[X>0]=1
            return d

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1,1)

    def forward_pass(self, X, params = None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + params["B1"] # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1) # (N, 2)
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"] # (N, 2) * (2, 4) -> (N, 4)
        self.H2 = self.softmax(self.A2) # (N, 4)
        return self.H2

    def grad(self, X, Y, params = None):
        if params is None:
            params = self.params

        self.forward_pass(X, params)
        m = X.shape[0]
        self.gradients["dA2"] = self.H2 - Y # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"]) # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1) # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T) # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"]) # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1) # (N, 2) -> (1, 2)

    def fit(self, X, Y, epochs=1, algo= "GD", display_loss=False,
            eta=1, mini_batch_size=100, eps=1e-8,
            beta=0.9, beta1=0.9, beta2=0.9, gamma=0.9 ):

        if display_loss:
            loss = {}
            Y_pred = self.predict(X)
            loss[0] = log_loss(np.argmax(Y, axis=1), Y_pred)

        for num_epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            m = X.shape[0]

            if algo == "GD":
                self.grad(X, Y)
                for i in range(1,self.num_layers+1):
                    self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/m)
                    self.params["B"+str(i)] -= eta * (self.gradients["dB"+str(i)]/m)

            elif algo == "MiniBatch":
                for k in range(0,m,mini_batch_size):
                    self.grad(X[k:k+mini_batch_size], Y[k:k+mini_batch_size])
                    for i in range(1,self.num_layers+1):
                        self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/mini_batch_size)
                        self.params["B"+str(i)] -= eta * (self.gradients["dB"+str(i)]/mini_batch_size)

            elif algo == "Momentum":
                self.grad(X, Y)
                for i in range(1,self.num_layers+1):
                    self.update_params["v_w"+str(i)] = gamma *self.update_params["v_w"+str(i)] + eta * (self.gradients["dW"+str(i)]/m)
                    self.update_params["v_b"+str(i)] = gamma *self.update_params["v_b"+str(i)] + eta * (self.gradients["dB"+str(i)]/m)
                    self.params["W"+str(i)] -= self.update_params["v_w"+str(i)]
                    self.params["B"+str(i)] -= self.update_params["v_b"+str(i)]

            elif algo == "NAG":
                temp_params = {}
                for i in range(1,self.num_layers+1):
                    self.update_params["v_w"+str(i)]=gamma*self.prev_update_params["v_w"+str(i)]
                    self.update_params["v_b"+str(i)]=gamma*self.prev_update_params["v_b"+str(i)]
                    temp_params["W"+str(i)]=self.params["W"+str(i)]-self.update_params["v_w"+str(i)]
                    temp_params["B"+str(i)]=self.params["B"+str(i)]-self.update_params["v_b"+str(i)]
                self.grad(X,Y,temp_params)
                for i in range(1,self.num_layers+1):
                    self.update_params["v_w"+str(i)] = gamma *self.update_params["v_w"+str(i)] + eta * (self.gradients["dW"+str(i)]/m)
                    self.update_params["v_b"+str(i)] = gamma *self.update_params["v_b"+str(i)] + eta * (self.gradients["dB"+str(i)]/m)
                    self.params["W"+str(i)] -= eta * (self.update_params["v_w"+str(i)])
                    self.params["B"+str(i)] -= eta * (self.update_params["v_b"+str(i)])
                self.prev_update_params=self.update_params

            elif algo == "AdaGrad":
                self.grad(X, Y)
                for i in range(1,self.num_layers+1):
                    self.update_params["v_w"+str(i)] += (self.gradients["dW"+str(i)]/m)**2
                    self.update_params["v_b"+str(i)] += (self.gradients["dB"+str(i)]/m)**2
                    self.params["W"+str(i)] -= (eta/(np.sqrt(self.update_params["v_w"+str(i)])+eps)) * (self.gradients["dW"+str(i)]/m)
                    self.params["B"+str(i)] -= (eta/(np.sqrt(self.update_params["v_b"+str(i)])+eps)) * (self.gradients["dB"+str(i)]/m)

            elif algo == "RMSProp":
                self.grad(X, Y)
                for i in range(1,self.num_layers+1):
                    self.update_params["v_w"+str(i)] = beta*self.update_params["v_w"+str(i)] +(1-beta)*((self.gradients["dW"+str(i)]/m)**2)
                    self.update_params["v_b"+str(i)] = beta*self.update_params["v_b"+str(i)] +(1-beta)*((self.gradients["dB"+str(i)]/m)**2)
                    self.params["W"+str(i)] -= (eta/(np.sqrt(self.update_params["v_w"+str(i)]+eps)))*(self.gradients["dW"+str(i)]/m)
                    self.params["B"+str(i)] -= (eta/(np.sqrt(self.update_params["v_b"+str(i)]+eps)))*(self.gradients["dB"+str(i)]/m)

            elif algo == "Adam":
                self.grad(X, Y)
                num_updates=0
                for i in range(1,self.num_layers+1):
                    num_updates+=1
                    self.update_params["m_w"+str(i)]=beta1*self.update_params["m_w"+str(i)]+(1-beta1)*(self.gradients["dW"+str(i)]/m)
                    self.update_params["v_w"+str(i)]=beta2*self.update_params["v_w"+str(i)]+(1-beta2)*((self.gradients["dW"+str(i)]/m)**2)
                    m_w_hat=self.update_params["m_w"+str(i)]/(1-np.power(beta1,num_updates))
                    v_w_hat=self.update_params["v_w"+str(i)]/(1-np.power(beta2,num_updates))
                    self.params["W"+str(i)] -=(eta/np.sqrt(v_w_hat+eps))*m_w_hat

                    self.update_params["m_b"+str(i)]=beta1*self.update_params["m_b"+str(i)]+(1-beta1)*(self.gradients["dB"+str(i)]/m)
                    self.update_params["v_b"+str(i)]=beta2*self.update_params["v_b"+str(i)]+(1-beta2)*((self.gradients["dB"+str(i)]/m)**2)
                    m_b_hat=self.update_params["m_b"+str(i)]/(1-np.power(beta1,num_updates))
                    v_b_hat=self.update_params["v_b"+str(i)]/(1-np.power(beta2,num_updates))
                    self.params["B"+str(i)] -=(eta/np.sqrt(v_b_hat+eps))*m_b_hat

            if display_loss:
                Y_pred = self.predict(X)
                loss[num_epoch+1] = log_loss(np.argmax(Y, axis=1), Y_pred)
                self.params_h.append(np.concatenate((self.params['W1'].ravel(), self.params['W2'].ravel(), self.params['B1'].ravel(), self.params['B2'].ravel())))

        if display_loss:
            plt.plot(loss.values(), '-o', markersize=5)
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()


    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()

def post_process(scatter_plot=False, gradient_plot=True, plot_scale=0.1):
    Y_pred_train = model.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train,1)
    Y_pred_val = model.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val,1)
    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)
    print("Training accuracy", round(accuracy_train, 4))
    print("Validation accuracy", round(accuracy_val, 4))

    if scatter_plot:
        plt.scatter(X_train[:,0], X_train[:,1], c=Y_pred_train, cmap=my_cmap, s=15*(np.abs(np.sign(Y_pred_train-Y_train))+.1))
        plt.show()

    if gradient_plot:
        h = np.asarray(model.params_h)
        h_diff = (h[0:-1, :] - h[1:, :])
        for i in range(18):
            plt.subplot(6, 3, i+1)
            plt.plot(h_diff[:, i], '-')
            plt.ylim((-plot_scale, plot_scale))
            plt.yticks([])
            plt.xticks([])
        plt.show()

model = FFNetwork(init_method='xavier', activation_function='sigmoid')
model.fit(X_train, y_OH_train, epochs=10, eta=1, algo="GD", display_loss=True)
post_process()

for init_method in ['zeros', 'random', 'xavier', 'he']:
    for activation_function in ['sigmoid']:
        print(init_method, activation_function)
        model = FFNetwork(init_method=init_method, activation_function=activation_function)
        model.fit(X_train, y_OH_train, epochs=50, eta=1, algo="GD", display_loss=True)
        post_process(plot_scale=0.05)
        print('\n--\n')

for init_method in ['zeros', 'random', 'xavier', 'he']:
    for activation_function in ['tanh']:
        print(init_method, activation_function)
        model = FFNetwork(init_method=init_method, activation_function=activation_function)
        model.fit(X_train, y_OH_train, epochs=100, eta=0.5, algo="NAG", display_loss=True)
        post_process(plot_scale=0.05)
        print('\n--\n')

for init_method in ['zeros', 'random', 'xavier', 'he']:
    for activation_function in ['relu']:
        print(init_method, activation_function)
        model = FFNetwork(init_method=init_method, activation_function=activation_function)
        model.fit(X_train, y_OH_train, epochs=50, eta=0.25, algo="GD", display_loss=True)
        post_process(plot_scale=0.05)
        print('\n--\n')

for init_method in ['zeros', 'random', 'xavier', 'he']:
    for activation_function in ['leaky_relu']:
        print(init_method, activation_function)
        model = FFNetwork(init_method=init_method, activation_function=activation_function, leaky_slope=0.1)
        model.fit(X_train, y_OH_train, epochs=50, eta=0.5, algo="GD", display_loss=True)
        post_process(plot_scale=0.05)
        print('\n--\n')

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        # input[input < 0] /= 4
        return input
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[input < 0][0 <= grad_input] = 0
        # grad_input[(input < 0) * (0 <= grad_input)] = 0
        # grad_input[input < 0] /= 4
        # grad_input[input < 0] /= torch.abs(input)
        return grad_input

    # custom activation
class SOLeakyReLU(nn.Module):
    def __init__(self):
        super(SOLeakyReLU, self).__init__()

    def forward(self, x):
        return MyReLU.apply(x)