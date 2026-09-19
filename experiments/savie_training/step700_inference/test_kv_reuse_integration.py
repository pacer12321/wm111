"""Full grouped attention versus Flex; only isolated candidate gate is updated."""
from pathlib import Path
import sys
root = Path('/cache/zhonghao/h3/savie_step1000_eval')
sys.path.insert(0, str(root / 'kvreuse_loader'))
from savie_target_kv_reuse import install
install(None)
source = (root / 'speedops_candidate/test_target_flash_integration.py').read_text()
source = source.replace("ROOT/'targetflash_loader'", "ROOT/'kvreuse_loader'")
source = source.replace("ROOT/'targetflash_candidate/prepared.json'", "ROOT/'kvreuse_candidate/prepared.json'")
source = source.replace('target_flash_gpu_cases=cases', 'kvreuse_gpu_cases=cases')
exec(compile(source, str(__file__), 'exec'), {'__name__': '__main__', '__file__': str(__file__)})
