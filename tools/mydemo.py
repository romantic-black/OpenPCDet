import torch

if __name__ == '__main__':
    # 前向传播
    import torch

    # 前向传播
    A = torch.randn(3, 4, requires_grad=True)
    B = torch.randn(4, 2, requires_grad=True)
    Y = torch.mm(A, B)

    # 假设损失函数 L 是 Y 的所有元素的和
    L = Y.sum()

    # 反向传播
    L.backward()

    # 现在，A.grad 和 B.grad 存储了关于损失 L 的梯度。
    # 这些梯度是自动计算的，你也可以手动计算它们以进行验证。

    # 手动计算关于损失 L 的梯度
    dL_dY = torch.ones_like(Y)  # 因为 L 是 Y 的所有元素的和，所以 dL/dY 是全 1 矩阵
    dL_dA = torch.mm(dL_dY, B.t())
    dL_dB = torch.mm(A.t(), dL_dY)

    print("Automatically computed gradients: ")
    print("dL/dA: ", A.grad)
    print("dL/dB: ", B.grad)

    print("Manually computed gradients: ")
    print("dL/dA: ", dL_dA)
    print("dL/dB: ", dL_dB)



