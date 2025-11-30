import sys
import os
from pathlib import Path
import sys
from accelerate import Accelerator

parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))
sys.path.append(os.getcwd())
import unittest
import torch
from torch.utils.data import DataLoader,random_split,TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

from src.experiment_helpers.loop_decorator import optimization_loop

def get_mnist_data(batch_size:int):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # --- 1. Load full training dataset ---
    full_train = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    # --- 2. Split into train + val ---
    train_size = int(0.8 * len(full_train))   # 80% train
    val_size   = int(0.2 * len(full_train))   # 20% val

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    # --- 3. Load official test set ---
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    # --- 4. DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,val_loader,test_loader

def get_mnist_model():
    model= torch.nn.Sequential(*[
            torch.nn.Conv2d(1,1,4,2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(13*13,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,10)
        ])
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)
        elif isinstance(m,torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)
            
    #model.apply(init_weights)
    return model

def get_regression_data(batch_size):
    N = 1000      # number of samples
    D_in = 5      # input features
    D_out = 1     # target dimension

    torch.manual_seed(42)

    X = torch.randn(N, D_in)
    true_w = torch.tensor([[2.0], [-1.0], [0.5], [3.0], [-2.0]])
    true_b = 1.0

    y = X @ true_w + true_b + 0.1*torch.randn(N, D_out)

    # --- 2. Wrap in TensorDataset ---
    dataset = TensorDataset(X, y)

    # --- 3. Split into train/val/test ---
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,val_loader,test_loader

def get_regression_model():
    return torch.nn.Linear(5,1)
    

class TestDecorator(unittest.TestCase):
    
    def setUp(self):
        self.accelerator=Accelerator(gradient_accumulation_steps=4)
        
    
    def test_minimal_stub(self):
        
        train_loader=[1 for _ in range(2)]
        epochs=3
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=0)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_minimal_stub_all_loaders(self):
        train_loader=[1 for _ in range(2)]
        epochs=5
        val_interval=2
        val_loader=[1 for _ in range(2)]
        test_loader=[1 for _ in range(2)]
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_minimal_stub_all_loaders_save(self):
        train_loader=[1 for _ in range(2)]
        epochs=5
        val_interval=2
        val_loader=[1 for _ in range(2)]
        test_loader=[1 for _ in range(2)]
        
        save_dict={
            "epochs":0
        }
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_mnist(self):
        train_loader,val_loader,test_loader=get_mnist_data(2)
        
        save_dict={
            "epochs":0
        }
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            
        epochs=10
        val_interval=2
        limit=2
        
        model=get_mnist_model()
        
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
            
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save,
                           #model_list=[model]
                           )
        def stub(batch,train:bool,
                 #model_list:list
                 ):
            #model=model_list[0]
            images,labels=batch
            
            predicted=model(images)
            
            loss=torch.nn.CrossEntropyLoss()(predicted,labels)
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            return loss.cpu().detach().numpy() #,[model]
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertTrue((prior_value==trained_state_dict[key]).all())
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
            
        new_model=get_mnist_model()
        new_model.load_state_dict(torch.load(save_path))
        new_state_dict={key:value.cpu().detach().numpy() for key,value in new_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==trained_state_dict[key]).all())
        self.accelerator.print("\n")
        
    def test_mnist_accumulate(self):
        train_loader,val_loader,test_loader=get_mnist_data(1)
        
        save_dict={
            "epochs":0
        }
            
        epochs=5
        val_interval=2
        limit=10
        
        model=get_mnist_model()
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
            
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            images,labels=batch
            
            
            if train:
                with self.accelerator.accumulate():
                    predicted=model(images)
            
                    loss=torch.nn.CrossEntropyLoss()(predicted,labels)
                
                
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                predicted=model(images)
            
                loss=torch.nn.CrossEntropyLoss()(predicted,labels)
                
            return loss.cpu().detach().numpy()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertTrue((prior_value==trained_state_dict[key]).all())
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
        new_model=get_mnist_model()
        new_model.load_state_dict(torch.load(save_path))
        new_state_dict={key:value.cpu().detach().numpy() for key,value in new_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==trained_state_dict[key]).all())
        self.accelerator.print("\n")
        
    def test_regression(self):
        train_loader,val_loader,test_loader=get_regression_data(2)
        save_dict={
            "epochs":0
        }
            
        epochs=5
        val_interval=2
        limit=10
        
        model=get_regression_model()
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            x,y=batch
            
            predicted=model(x)
            loss=F.mse_loss(predicted.float(),y.float())
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            return loss.cpu().detach().numpy()
        
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
        
        
        
        
        
            

        
        
        
        
if __name__=="__main__":
    unittest.main()