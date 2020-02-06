import torch.nn as nn

def residual_dependencies(model, residual_instances):
#{{{
    def check_inst(instance): 
        check = []
        for m in instance.modules():
            check += [any(isinstance(m,inst) for inst in residual_instances['instance'])]
        return any(check)
    
    linkedConvs = []
    for n,m in model.module.named_children(): 
        if not check_inst(m):
            if isinstance(m,nn.Conv2d):
                linkedConvs.append(('basic',"module.{}".format(n)))
        else:
            for _n,_m in m.named_modules():
                if any(isinstance(_m,inst) for inst in residual_instances['instance']):
                    linkedConvs.append(('residual',"module.{}.{}".format(n,_n)))
    deps = []
    currDep = []
    for n,m in model.named_modules(): 
        if any(isinstance(m,inst) for inst in residual_instances['instance']):
            idx = residual_instances['instance'].index(type(m))
            convs = residual_instances['conv'][idx]
            ds = residual_instances['downsample'][idx]
            hasDS = any(dsName in m._modules.keys() for dsName in ds)
            curr = "{}.{}".format(n,convs[0]) 
            
            if not hasDS: 
                if len(currDep) != 0: 
                    currDep.append(curr)
                else: 
                    bType,name = zip(*linkedConvs)
                    idx = name.index(n)
                    if bType[idx-1] == 'basic':
                        prev = name[idx-1]
                    else:
                        prev = "{}.{}".format(name[idx-1],convs[0])
            
                    currDep = [prev,curr]
            else: 
                deps.append(currDep)
                currDep = []
    
    if currDep is not []: 
        deps.append(currDep)

    return deps
#}}}
