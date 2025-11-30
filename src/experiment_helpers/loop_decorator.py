
# Source - https://stackoverflow.com/a/5929165
# Posted by t.dubrownik, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-29, License - CC BY-SA 4.0
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
from typing import Callable


def optimization_loop(accelerator:Accelerator,
                      train_loader:DataLoader,
                      epochs:int,
                      val_interval:int,
                      limit:int=-1,
                      val_loader:DataLoader=None,
                      test_loader:DataLoader=None,
                      save_function:Callable=None,
                      #model_list:list=[]
                      ):
    def decorator(function):
        def wrapper(*args, **kwargs):
            #nonlocal model_list
            for e in range(epochs):
                loss_buffer=[]
                start=time.time()
                for b,batch in enumerate(train_loader):
                    if b==limit:
                        break
                    loss=function(batch,True)
                    loss_buffer.append(loss)
                end=time.time()
                accelerator.print(f"\t epoch {e} elapsed {end-start}")

                accelerator.log({
                        "loss_mean":np.mean(loss_buffer),
                        "loss_std":np.std(loss_buffer),
                    })
                if save_function is not None:
                    save_function()
                if val_loader is not None and  e % val_interval==1:
                    with torch.no_grad():
                        val_loss_buffer=[]
                        start=time.time()
                        for b,batch in enumerate(val_loader):
                            if b==limit:
                                break
                            loss=function(batch,False)
                            val_loss_buffer.append(loss)
                        end=time.time()
                        accelerator.print(f"\t val epoch {e} elapsed {end-start}")

                        accelerator.log({
                                "val_loss_mean":np.mean(val_loss_buffer),
                                "val_loss_std":np.std(val_loss_buffer),
                            })
            with torch.no_grad():
                if test_loader is not None:
                    test_loss_buffer=[]
                    start=time.time()
                    for b,batch in enumerate(test_loader):
                        if b==limit:
                            break
                        loss=function(batch,False)
                        test_loss_buffer.append(loss)
                    end=time.time()
                    accelerator.print(f"\t test epoch elapsed {end-start}")

                    accelerator.log({
                            "test_loss_mean":np.mean(test_loss_buffer),
                            "test_loss_std":np.std(test_loss_buffer),
                        })

            
        return wrapper
    return decorator
