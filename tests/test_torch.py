# author: Lukas Rauch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch

def run(): 
    print("CUDA is available:     {} " .format(torch.cuda.is_available()))
    print("Number available GPUs: {}" .format(torch.cuda.device_count()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nUsing device:       {}" .format(device))

    if "cuda" in device.type:
        print("Active device ID:   {}" .format(torch.cuda.current_device()))
        print("Active device Name: {}" .format(torch.cuda.get_device_name(0)))

        # #Additional Info when using cuda
        # if device.type == 'cuda':
        print('Memory Usage:')
        print('  Allocated: ', round(torch.cuda.memory_allocated(0)/1024**3,2), 'GB')
        print('  Reserved:  ', round(torch.cuda.memory_reserved(0)/1024**3,2), 'GB')




if __name__ == "__main__":
    print(" =========> TEST GPUs <=========") 
    run()
    