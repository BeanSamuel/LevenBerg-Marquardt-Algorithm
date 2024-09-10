import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.mu = 1e-3
        self.mu_max = 1e10
        self.mu_min = 1e-8
        self.mu_decay = 0.1
        self.tolerance = 1e-4
        self.max_iters = 100

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def neural_network(self, x, W2, b2, W3, b3):
        hidden_layer = self.sigmoid(np.dot(W2, x) + b2)
        output = np.dot(W3, hidden_layer) + b3
        return output

    def forward(self, x, params):
        W2, b2, W3, b3 = params[:10], params[10], params[11:21], params[21]
        return self.neural_network(x, W2, b2, W3, b3)

    def loss(self, params, x, y):
        predictions = np.array([self.forward(x_, params) for x_ in x])
        return predictions - y

    def jacobian(self, params, x):
        """計算雅可比矩陣"""
        W2, b2, W3, b3 = params[:10], params[10], params[11:21], params[21]
        J = np.zeros((len(x), len(params)))  # 初始化雅可比矩陣

        for i in range(len(x)):
            xi = x[i]
            # 計算隱藏層輸出
            hidden_layer = self.sigmoid(np.dot(W2, xi) + b2)
            
            # 對於每個參數計算偏導數
            # W2 的偏導數
            for j in range(10):
                J[i, j] = W3[j] * hidden_layer[j] * (1 - hidden_layer[j]) * xi  # 每個 W2 的偏導數

            # b2 的偏導數
            for j in range(10):
                J[i, 10] += W3[j] * hidden_layer[j] * (1 - hidden_layer[j])

            # W3 的偏導數
            for j in range(10):
                J[i, 11+j] = hidden_layer[j]  # 每個 W3 的偏導數
            
            # b3 的偏導數
            J[i, 21] = 1  # 偏導數是常數

        return J

    def train(self, x_train, y_train, initial_params):
        params = initial_params
        for iteration in range(self.max_iters):

            residuals = self.loss(params, x_train, y_train)
            J = self.jacobian(params, x_train)

            H = np.dot(J.T, J)  # Hessian approximation
            gradient = np.dot(J.T, residuals)

            I = np.eye(len(params))
            update = np.linalg.solve(H + self.mu * I, -gradient)

            new_params = params + update

            new_residuals = self.loss(new_params, x_train, y_train)
            new_cost = np.sum(new_residuals ** 2)

            old_cost = np.sum(residuals ** 2)

            if new_cost < old_cost:
                params = new_params
                self.mu = max(self.mu * self.mu_decay, self.mu_min)
                print(f"Iteration {iteration}, Cost: {new_cost}")
                if np.abs(old_cost - new_cost) < self.tolerance:
                    print("Convergence reached.")
                    break
            else:
                self.mu = min(self.mu * 10, self.mu_max)
            
        return params

    def generate_training_data(self, m=100):
        x = np.linspace(0, 1, m)
        y = np.sin(2 * np.pi * x)
        return x, y

model = Model()

x_train, y_train = model.generate_training_data()

np.random.seed(42)
initial_params = np.random.randn(22)  # 10 W2, 1 b2, 10 W3, 1 b3

optimal_params = model.train(x_train, y_train, initial_params)

x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)
y_pred = np.array([model.forward(xi, optimal_params) for xi in x_test])

plt.plot(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, label="Prediction", linestyle='dashed')
plt.legend()
plt.show()

