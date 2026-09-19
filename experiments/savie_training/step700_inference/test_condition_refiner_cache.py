import json
from types import SimpleNamespace
import torch
from savie_condition_refiner_cache import refined_condition,clear_request


class Projection:
    def __init__(self):self.calls=0
    def __call__(self,x):self.calls+=1;return x*.7,None


class Refiner:
    def __init__(self):self.calls=0
    def __call__(self,x,*,cu_seqlens,max_seqlen):
        self.calls+=1
        return x.sin()+float(max_seqlen)*.01+cu_seqlens.float().sum()*.001


@torch.inference_mode()
def test():
    m=SimpleNamespace(condition_proj=Projection(),token_refiner=Refiner())
    x=torch.randn(19,32);cu=torch.tensor([0,19],dtype=torch.int32)
    reference=lambda x,c,maxlen:((x*.7).sin()+float(maxlen)*.01+c.float().sum()*.001)
    for _ in range(8):torch.testing.assert_close(refined_condition(m,x,cu,19),reference(x,cu,19),rtol=0,atol=0)
    assert m.condition_proj.calls==m.token_refiner.calls==1
    x.add_(.1)  # Inference-mode tensors have no version counter.
    torch.testing.assert_close(refined_condition(m,x,cu,19),reference(x,cu,19),rtol=0,atol=0)
    assert m.token_refiner.calls==2
    cu[-1]=18
    torch.testing.assert_close(refined_condition(m,x,cu,19),reference(x,cu,19),rtol=0,atol=0)
    assert m.token_refiner.calls==3
    torch.testing.assert_close(refined_condition(m,x,cu,20),reference(x,cu,20),rtol=0,atol=0)
    assert m.token_refiner.calls==4
    clear_request(m)
    refined_condition(m,x,cu,20)
    assert m.token_refiner.calls==5
    print(json.dumps(dict(passed=True,same_request_eight_calls_reduce_to_one=True,
                         input_mutation_invalidates=True,layout_change_invalidates=True,
                         new_request_invalidates=True,scope='CPU cache protocol; not GPU speed or full-model validation')))


if __name__=='__main__':test()
