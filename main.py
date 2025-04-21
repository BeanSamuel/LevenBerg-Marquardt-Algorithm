import numpy as np
import matplotlib.pyplot as plt

class LevenbergMarquardtNN:
    """
    Levenberg-Marquardt 神經網路訓練類別
    用於非線性最小二乘問題的參數優化
    """
    def __init__(self, hidden_size=10):
        self.mu = 1e-3           # 初始阻尼參數 (damping factor)
        self.mu_max = 1e10       # 最大阻尼參數
        self.mu_min = 1e-8       # 最小阻尼參數
        self.mu_decay = 0.1      # 阻尼參數衰減率
        self.tolerance = 1e-4    # 收斂閥值
        self.max_iters = 100     # 最大迭代次數
        self.hidden_size = hidden_size  # 隱藏層神經元數量

    def sigmoid(self, x):
        """Sigmoid 激活函數"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def unpack_params(self, params):
        """
        將參數向量拆解為 W2, b2, W3, b3
        params 長度必須為 hidden_size + 1 + hidden_size + 1
        """
        W2 = params[:self.hidden_size]
        b2 = params[self.hidden_size]
        start = self.hidden_size + 1
        W3 = params[start:start+self.hidden_size]
        b3 = params[-1]
        return W2, b2, W3, b3

    def forward_one(self, x, params):
        """
        計算單一輸入 x 的網路輸出
        x: 標量輸入
        params: 參數向量
        """
        W2, b2, W3, b3 = self.unpack_params(params)
        z2 = W2 * x + b2        # shape=(hidden_size,)
        a2 = self.sigmoid(z2)   # shape=(hidden_size,)
        output = np.dot(W3, a2) + b3  # shape=標量
        return output
    
    def compute_residuals(self, params, x, y):
        """
        計算所有訓練資料的殘差向量
        r = y_pred - y_true
        """
        y_pred = np.array([self.forward_one(xi, params) for xi in x])
        return y_pred - y

    def compute_jacobian(self, params, x):
        """
        計算雅可比矩陣 J (大小 len(x) x len(params))
        J[i,j] = ∂f(params, x[i]) / ∂params[j]
        """
        W2, b2, W3, b3 = self.unpack_params(params)
        m = len(x)
        n_params = len(params)
        J = np.zeros((m, n_params))
        for i, xi in enumerate(x):
            z2 = W2 * xi + b2
            a2 = self.sigmoid(z2)
            da2_dz2 = a2 * (1 - a2)
            J[i, :self.hidden_size] = W3 * da2_dz2 * xi
            J[i, self.hidden_size] = np.sum(W3 * da2_dz2)
            start = self.hidden_size + 1
            J[i, start:start+self.hidden_size] = a2
            J[i, -1] = 1.0
        return J
    
    def train(self, x, y, initial_params):
        """
        使用 Levenberg-Marquardt 演算法訓練參數
        x: 輸入資料 (array)
        y: 真實輸出 (array)
        initial_params: 初始參數向量
        """
        params = initial_params.copy()
        for iteration in range(self.max_iters):
            r = self.compute_residuals(params, x, y)
            J = self.compute_jacobian(params, x) 
            H = J.T @ J
            g = J.T @ r
            I = np.eye(len(params))
            delta = np.linalg.solve(H + self.mu * I, -g)
            new_params = params + delta
            new_r = self.compute_residuals(new_params, x, y)
            cost = np.sum(r**2)
            new_cost = np.sum(new_r**2)
            if new_cost < cost:
                params = new_params
                self.mu = max(self.mu * self.mu_decay, self.mu_min)
                print(f"Iteration {iteration}, Cost: {new_cost:.6f}, mu: {self.mu:.2e}")
                if abs(cost - new_cost) < self.tolerance:
                    print("Convergence reached.")
                    break
            else:
                self.mu = min(self.mu * 10, self.mu_max)
        return params

    def generate_data(self, m=100):
        """
        生成訓練資料: x ∈ [0,1], y = sin(2πx)
        """
        x = np.linspace(0, 1, m)
        y = np.sin(2 * np.pi * x)
        return x, y

if __name__ == "__main__":
    np.random.seed(42)
    model = LevenbergMarquardtNN(hidden_size=10)
    x_train, y_train = model.generate_data(m=100)
    n_params = model.hidden_size + 1 + model.hidden_size + 1
    initial_params = np.random.randn(n_params)
    # 開始訓練
    optimal_params = model.train(x_train, y_train, initial_params)
    # 測試與視覺化
    x_test, y_test = x_train, y_train
    y_pred = np.array([model.forward_one(xi, optimal_params) for xi in x_test])
    plt.plot(x_test, y_test, label="Actual")
    plt.plot(x_test, y_pred, linestyle='dashed', label="Prediction")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Levenberg-Marquardt 神經網路擬合")
    plt.show()