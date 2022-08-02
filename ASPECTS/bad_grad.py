# Adapted from: https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d

from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            # print("None")
            return True
        grad_output = grad_output.data
        cond1 = grad_output.ne(grad_output).any()
        cond2 = grad_output.gt(1e6).any()
        # print("cond1", cond1)
        # print("cond2", cond2)
        return cond1 or cond2

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                # print(fn_dict)
                # print()
                for gi in fn_dict[fn]:
                    if is_bad_grad(gi):
                        fillcolor = "red"
                        break
                if fillcolor == "red":
                    print( str(type(fn).__name__) )
                # print()
                # if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                #     fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

if __name__ == '__main__':
    # dot -Tps tmp.dot -o tmp.ps
    import os
    
    
    # from aspects_mil_loader import ASPECTSMILLoaderBebug
    # from aspects_mil import create_model
    # dirname = "../../../data/gravo"
    # # dirname = "/media/avcstorage/gravo/"
    # loader = ASPECTSMILLoaderBebug("ncct_radiomic_features.csv", "all", 
    #         normalize = False, dirname = dirname)
    # model = create_model()
    # for param in model.parameters():
    #     print(param)
    # LOSS  = torch.nn.MSELoss()
    # for x, y in loader.get_set("train"):
    #     pred = model(x)
    #     loss = LOSS(pred, y)
    #     get_dot = register_hooks(loss)
    #     loss.backward()
    #     dot = get_dot()
    #     dot.save('tmp.dot')
    #     os.system("dot -Tps tmp.dot -o tmp.ps")
    #     break
    
    # class InstanceClassifier(torch.nn.Module):
    #     def __init__(self, bias = False):
    #         super().__init__()
    #         self.model = torch.nn.Sequential(
    #             torch.nn.Linear(1,1, bias = bias),
    #             torch.nn.Sigmoid(),
    #             torch.nn.Linear(1,1, bias = bias)
    #         )
    #         self.model = torch.nn.Linear(1,1, bias = bias)
    #     def __call__(self, x):
    #         x = self.model(x)
    #         x = torch.sign(x)
    #         x = torch.relu(x)
    #         return x
    # class Model(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.instance_classifer = InstanceClassifier()
    #     def __call__(self, x):
    #         x = self.instance_classifer(x)
    #         print(x)
    #         # x = torch.tensor([3], dtype = torch.float, requires_grad = True) - x.sum()
    #         # x = torch.tensor([3], dtype = torch.float, requires_grad = True) - x[0] - x[1] - x[2]
    #         # x = 3 - x.sum()
    #         x = 3 - x
    #         return x
    # LOSS = torch.nn.MSELoss()
    # model = Model()
    # x = torch.Tensor([[0,0,0], [0,0,1], [1,1,1], [1,0,1], [0,1,1], [0,1,0], [1,1,0]])
    # y = torch.Tensor([3,2,0,1,1,2,1]).view(-1,1)
    # # x = x[0].reshape(3,1)
    # x = x[0,0].reshape(1,1)
    # # print(x.shape)
    # # exit(0)
    # pred = model(x)
    # # .unsqueeze(dim = 0)
    # loss = LOSS(pred, y[0][0])
    # get_dot = register_hooks(loss)
    # loss.backward()
    # dot = get_dot()
    # dot.save('tmp.dot')
    # os.system("dot -Tps tmp.dot -o tmp.ps")
    
    # x = Variable(torch.randn(10, 10), requires_grad=True)
    # y = Variable(torch.randn(10, 10), requires_grad=True)
    # z = x / (y * 100)
    # z = z.sum() * 2
    # get_dot = register_hooks(z)
    # z.backward()
    # dot = get_dot()
    # dot.save('simple_example.dot')
    # os.system("dot -Tps simple_example.dot -o simple_example.ps")
    
    x = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
    y = torch.Tensor([0, 1, 1, 0]).view(-1,1)
    bias = False
    model = torch.nn.Sequential(
        torch.nn.Linear(2,2, bias = bias),
        torch.nn.Sigmoid(),
        torch.nn.Linear(2,1, bias = bias),
    )
    LOSS = torch.nn.MSELoss()
    pred = model(x[0])
    loss = LOSS(pred, y[0])
    get_dot = register_hooks(loss)
    loss.backward()
    dot = get_dot()
    dot.save('tmp.dot')
    os.system("dot -Tps tmp.dot -o tmp.ps")
    
