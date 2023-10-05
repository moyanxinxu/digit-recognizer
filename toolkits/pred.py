import pandas as pd
import torch

from net import myNet

preds = []
model = myNet()
model.load_state_dict(torch.load("./models/temp.pth"))
df = pd.read_csv("./data/test.csv")

data = torch.tensor(df.to_numpy(), dtype=torch.float32)
data = data.reshape((-1, 1, 28, 28))

for img in data:
    img = img.unsqueeze(dim=0)
    y_hat = model(img)
    pred = torch.argmax(y_hat, dim=1)
    preds.append(pred.item())

pd.DataFrame({"ImageId": range(1, 28001), "Label": preds}).to_csv(
    "./info/answer.csv", index=False
)
