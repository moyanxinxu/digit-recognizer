import torch
from iter import load_data
from net import myNet
from tqdm import tqdm


lr = 0.9
batch_size = 256
num_epoches = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

net = myNet()
train_iter, test_iter = load_data("./data/train.csv", batch_size)


net = net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss(reduction="none")

train_loss = []
test_loss = []


for epoch in tqdm(range(num_epoches)):
    net.train()
    # 训练部分
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.mean().backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        loss_sum = 0
        # 训练集误差
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            loss_sum += l.sum().item()
        if (epoch + 1) % 2 == 0:
            train_loss.append(str(loss_sum / len(train_iter.dataset)))
        loss_sum = 0
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            loss_sum += l.sum().item()
        if (epoch + 1) % 2 == 0:
            test_loss.append(str(loss_sum / len(test_iter.dataset)))

torch.save(net.state_dict(), "./models/temp.pth")

with open("./info/loss.csv", "w", encoding="utf-8") as f:
    f.write(",".join(train_loss))
    f.write("\n")
    f.write(",".join(test_loss))
