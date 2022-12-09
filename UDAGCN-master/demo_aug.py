from torch_geometric.datasets import CoraFull
import grafog.transforms as T

node_aug = T.Compose([
    T.NodeDrop(p=0.45),
    T.NodeMixUp(lamb=0.5, classes=7),
])
#edge_aug = T.Compose([
#T.EdgeDrop(0 = 0.15),
#T.EdgeFeatureMasking(),
#])

data = CoraFull('./dataset')
data = data[0]
for epoch in range(10):  # begin training loop
    new_data = node_aug(data)  # apply the node augmentation(s)
    new_data = edge_aug(new_data)  # apply the edge augmentation(s)

    x, y = new_data.x, new_data.y
    print(sum(data.train_mask-new_data.train_mask))
    break
