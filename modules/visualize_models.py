from constants import *
import onnx

class CustomVertexCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomVertexCNN, self).__init__()
        
        # Using 1D convolutions with a kernel size of 4
        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(9216, 2048)  # Adjust the input features according to your feature map size before this layer
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding a channel dimension, assuming x has shape [batch_size, num_features]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool2(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Sigmoid activation for multi-label classification
        return x
    
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Initial large kernel convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Additional convolutional layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Adjusted final pooling to correct size reduction
        self.finalpool = nn.MaxPool2d(kernel_size=3, stride=3)  # Adjust to get from 25x25 to 7x7
        
        # Fully connected layers
        self.fc1 = nn.Linear(51200, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool2(x)
        x = self.finalpool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model instantiation
model1 = CustomVertexCNN(num_classes=NUM_TAGS)

input_names = ['Vertices']
output_names = ['Scene Context Tag Prediction']

X1 = torch.rand(5, 80)
torch.onnx.export(model1, X1, "model1.onnx", input_names=input_names, output_names=output_names)

model2 = CustomCNN(num_classes=NUM_TAGS)

input_names = ['Image']
output_names = ['Scene Context Tag Prediction']

X2 = torch.rand(5, 3, 500, 500)
torch.onnx.export(model2, X2, "model2.onnx", input_names=input_names, output_names=output_names)