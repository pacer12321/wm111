"""CPU regression for Ref2VA Qwen visual-prefix AdaLN tags."""
import importlib.util
import os
from pathlib import Path
import unittest
import torch

if os.environ.get('SAVIE_BATCH_MODULE_FILE'):
    spec = importlib.util.spec_from_file_location('audited_ref2va_batch', os.environ['SAVIE_BATCH_MODULE_FILE'])
    batch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_module)
else:
    from src.training import ref2va_batch as batch_module

class PromptTagTests(unittest.TestCase):
    def sample(self, tags=None):
        if tags is None:
            tags = torch.tensor([1, 0, 0, 1, 0, 1, 1])
        return dict(source_video_latents=torch.zeros(24, 3, 4, 4),
                    target_video_latents=torch.ones(24, 3, 4, 4),
                    target_audio_latents=torch.zeros(2, 32, 3),
                    prompt_embeds=torch.zeros(7, 5120, dtype=torch.bfloat16),
                    text_token_tags=tags)

    def pack(self, sample, step=0):
        return batch_module.pack_ref2va_batch(
            sample, 'cpu', torch.Generator().manual_seed(4101),
            torch.Generator().manual_seed(4102), step_index=step)

    def test_mixed_prefix_preserved_for_all_eight_timesteps(self):
        sample = self.sample()
        for step in range(8):
            with self.subTest(step=step):
                b = self.pack(sample, step); inp = b['inputs']; pos = inp['text_indices']
                self.assertTrue(torch.equal(inp['token_tags'][pos], sample['text_token_tags']))
                expected = 3 * inp['timestep_indices'][pos] + sample['text_token_tags']
                actual = (3 * inp['timestep_indices'] + inp['token_tags'])[pos]
                self.assertTrue(torch.equal(actual, expected))
                layout = b['layout']
                self.assertTrue(bool((inp['token_tags'][layout.source_start:layout.source_end] == 0).all()))
                self.assertTrue(bool((inp['token_tags'][layout.target_start:layout.target_end] == 0).all()))
                self.assertTrue(bool((inp['token_tags'][inp['audio_indices']] == 2).all()))

    def test_fixed_silent_input_is_zero_at_all_dmd8_steps(self):
        s = self.sample(); s['audio_input_policy'] = 'fixed-silent'
        for step in range(8):
            b = self.pack(s, step)
            self.assertEqual(int(torch.count_nonzero(b['inputs']['audio_hidden_states'])), 0)
            self.assertEqual(tuple(b['inputs']['audio_hidden_states'].shape), (1, 6, 32))

    def test_audio_policy_does_not_change_video_noise_or_layout(self):
        s = self.sample(); a = self.pack(s, 4)
        s['audio_input_policy'] = 'fixed-silent'; b = self.pack(s, 4)
        self.assertEqual(a['layout'], b['layout'])
        for key in a['inputs']:
            if key == 'audio_hidden_states': continue
            if isinstance(a['inputs'][key], torch.Tensor):
                self.assertTrue(torch.equal(a['inputs'][key], b['inputs'][key]), key)
        self.assertTrue(torch.equal(a['target_video_rows'], b['target_video_rows']))

    def test_unknown_audio_policy_is_rejected(self):
        s = self.sample(); s['audio_input_policy'] = 'mistyped'
        with self.assertRaisesRegex(ValueError, 'audio input policy'):
            self.pack(s)

    def test_tag_change_does_not_change_other_inputs(self):
        a = self.pack(self.sample(), 5)
        b = self.pack(self.sample(torch.ones(7, dtype=torch.long)), 5)
        for key in a['inputs']:
            if key == 'token_tags': continue
            av, bv = a['inputs'][key], b['inputs'][key]
            if isinstance(av, torch.Tensor): self.assertTrue(torch.equal(av, bv), key)
            else: self.assertEqual(av, bv, key)
        self.assertEqual(a['layout'], b['layout'])
        for key in a['noise_bundle']:
            self.assertTrue(torch.equal(a['noise_bundle'][key], b['noise_bundle'][key]))

    def test_all_text_and_all_visual_prefixes(self):
        for value in (0, 1):
            with self.subTest(value=value):
                s = self.sample(torch.full((7,), value, dtype=torch.long))
                b = self.pack(s)['inputs']
                self.assertTrue(torch.equal(b['token_tags'][b['text_indices']], s['text_token_tags']))

    def test_reject_bad_tag_length(self):
        with self.assertRaisesRegex(ValueError, 'token count'):
            self.pack(self.sample(torch.ones(6, dtype=torch.long)))

    def test_reject_bad_tag_shape(self):
        with self.assertRaisesRegex(ValueError, '1-D'):
            self.pack(self.sample(torch.ones(1, 7, dtype=torch.long)))

    def test_reject_noninteger_tags(self):
        for dtype in (torch.float32, torch.bool):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, 'integer dtype'):
                self.pack(self.sample(torch.ones(7, dtype=dtype)))

    def test_reject_unknown_modalities(self):
        for value in (-1, 2, 3):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'VIDEO_TAG'):
                self.pack(self.sample(torch.full((7,), value)))

    def test_reject_bad_prompt_shape(self):
        s = self.sample(); s['prompt_embeds'] = s['prompt_embeds'][None]
        with self.assertRaisesRegex(ValueError, 'prompt_embeds'):
            self.pack(s)

    def test_real_prompt_caches_preserve_every_tag(self):
        root = Path('/temp/zhonghao/savie_stream/samples')
        if not (root/'prompt_000000.pt').exists():
            self.skipTest('real training cache not available')
        for index in (0, 1, 25, 100):
            with self.subTest(index=index):
                prompt = torch.load(root/f'prompt_{index:06d}.pt', map_location='cpu', weights_only=True)
                s = self.sample(); s.update(prompt)
                b = self.pack(s)['inputs']; tags = b['token_tags'][b['text_indices']]
                self.assertTrue(torch.equal(tags, prompt['text_token_tags']))
                self.assertGreater(int((tags == 0).sum()), 0)
                print(f'REAL_PROMPT_TAGS_PASS sample={index} visual={int((tags==0).sum())} text={int((tags==1).sum())} mismatches=0', flush=True)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
