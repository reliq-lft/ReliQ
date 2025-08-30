"""
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: setup/gpu.py
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import subprocess
import tools

class GraphicsProcessingUnit(object):
    def __init__(self, device: str):
        info: dict[str, any] = tools.lspci_info(device)
        if 'NVIDIA' in device:
            self.vendor: str = 'NVIDIA'
            self.product: str = ' '.join(device.split()[2:])
        elif 'AMD' in device:
            self.vendor: str = 'AMD'
            self.product: str = ' '.join(device.split()[2:])
        elif 'Intel' in device:
            self.vendor: str = 'Intel'
            self.product: str = device.split(':')[-1].strip()
        self.subsystem: str = info['Subsystem']
        self.driver: str = info['Kernel driver in use']
        self.modules: str = info['Kernel modules']
        self.control: list[str] = info['Control'].split()
        self.status: list[str] = info['Status'].split()

class GraphicsProcessingUnits(object):
    def __init__(self): 
        self._gpus: list[GraphicsProcessingUnit] = []
        pci_devices = tools.lspci()
        for device in pci_devices:
            if not 'VGA' in device: continue
            self._gpus.append(GraphicsProcessingUnit(device))

    @property
    def GPUs(self) -> list[GraphicsProcessingUnit]: return self._gpus

    @property
    def has_nvidia(self) -> str: 
        has_nvidia_gpu = any(gpu.vendor == 'NVIDIA' for gpu in self.GPUs)
        return tools.bool_to_argbool(has_nvidia_gpu)

    @property
    def has_amd(self) -> str: 
        has_amd_gpu = any(gpu.vendor == 'AMD' for gpu in self.GPUs)
        return tools.bool_to_argbool(has_amd_gpu)
        
    @property
    def has_intel(self) -> str:
        has_intel_gpu = any(gpu.vendor == 'Intel' for gpu in self.GPUs)
        return tools.bool_to_argbool(has_intel_gpu)

    def __getitem__(self, idx) -> GraphicsProcessingUnit: return self._gpus[idx]

            
