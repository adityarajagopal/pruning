import sys

class Hook(object):
    def __init__(self, model):
        self.modules = model._modules
        self.layers = self.modules['module']._modules
        self.count = 0

    def register_hooks(self):
        count = 0
        for k,v in self.layers.items():
            count += 1
            v.register_forward_hook(self.compute_gops)
        print(count)
    
    def compute_gops(self, module, input, output):
        print('Conv' in str(module))
        # for name, param in module.named_modules():
        #     print(name)
        return None


