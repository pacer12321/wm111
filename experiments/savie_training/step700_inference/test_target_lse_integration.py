"""Reuse the existing full grouped-attention/Flex comparison for the LSE candidate."""
from pathlib import Path
import sys
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
sys.path.insert(0,str(root/'targetlse_loader'))
from savie_target_lse_candidate import install
install(None)
source=(root/'speedops_candidate/test_target_flash_integration.py').read_text()
source=source.replace("ROOT/'targetflash_loader'","ROOT/'targetlse_loader'")
source=source.replace("ROOT/'targetflash_candidate/prepared.json'","ROOT/'targetlse_candidate/prepared.json'")
source=source.replace('target_flash_gpu_cases=cases','target_lse_gpu_cases=cases')
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
