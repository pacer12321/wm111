"""CPU full-input parity against the actual serving packer, all 8 DMD8 steps.

This is NOT a full DiT numerical forward/backward or quality certificate.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

from shared_clip_contract import (audio_latent_count, merge_encoded_pair, file_sha)
from video_only_audio_policy import install_fixed_silent_audio_inputs
from savie_mask_arithmetic import make_arithmetic_mask
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--samples",type=Path,required=True)
    p.add_argument("--batch-module",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--count",type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location("new_batch",args.batch_module)
    batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
    install_fixed_silent_audio_inputs(MiniMaxH3DenoiseBranch)
    records=[]
    for i in range(args.count):
        video=torch.load(args.samples/f"video_{i:06d}.pt",map_location="cpu",weights_only=True)
        prompt=torch.load(args.samples/f"prompt_{i:06d}.pt",map_location="cpu",weights_only=True)
        sample=merge_encoded_pair(video,prompt)
        audio_t=audio_latent_count(sample["frame_count"],sample["video_fps"])
        sample["target_audio_latents"]=torch.zeros(2,32,audio_t,dtype=torch.bfloat16)
        sample["audio_input_policy"]="fixed-silent"
        latent_t,h,w=sample["expected_latent_shape"]
        l=sample["text_token_tags"].numel()
        packed=minimax_h3_packed_sequence_ref2va_blocks(text_len=l,latent_t=latent_t,
            latent_h=h,latent_w=w,audio_t=audio_t,
            ref_blocks=[dict(kind="video",ref_audio_t=0,latent_t=latent_t,latent_h=h,latent_w=w)])
        tags=packed["token_tags"].clone();tags[packed["text_pos"]]=sample["text_token_tags"]
        branch=MiniMaxH3DenoiseBranch(packed=packed,text_embeddings=sample["prompt_embeds"],
            token_tags=tags,device=torch.device("cpu"))
        used=l+2*latent_t*(h//2)*(w//2)+2*audio_t
        padded=branch.seq_len
        # Serving pads to its Ulysses alignment; training is unpadded. Prove that
        # every valid query excludes every padding key before comparing prefixes.
        layout_kwargs=dict(num_frames=latent_t,used_len=used,tokens_per_frame=(h//2)*(w//2),
            text_start=0,text_len=l)
        source_layout=SimpleNamespace(**layout_kwargs,video_start=l,video_end=l+latent_t*(h//2)*(w//2))
        target_layout=SimpleNamespace(**layout_kwargs,video_start=source_layout.video_end+2*audio_t,video_end=used)
        phys=torch.arange(padded);per_rank=padded//2
        perm=(phys%per_rank)*2+phys//per_rank
        mask=make_arithmetic_mask(target_layout,source_layout,padded,perm)
        valid_q=torch.arange(used);pad_k=torch.arange(used,padded)
        qphys=(valid_q%2)*per_rank+valid_q//2
        kphys=(pad_k%2)*per_rank+pad_k//2
        if pad_k.numel():
            assert not bool(mask(None,None,qphys[:,None],kphys[None,:]).any()),"padding leaks into real queries"
        tv=1-torch.tensor(minimax_h3_time_shift_sigmas(num_steps=9,shift_scale=12)[:-1])
        ta=1-torch.tensor(minimax_h3_time_shift_sigmas(num_steps=9,shift_scale=3)[:-1])
        for step in range(8):
            b=batch.pack_ref2va_batch(sample,"cpu",torch.Generator().manual_seed(4101),
                torch.Generator().manual_seed(4102),step_index=step)
            x=b["inputs"]
            # Deliberately nonzero incoming serving audio: the explicit video-only
            # adapter must zero it at EVERY step, not just initialize it once.
            out=branch.forward_kwargs(video_rows=x["hidden_states"][0],
                audio_rows=torch.full_like(x["audio_hidden_states"][0],123),
                t_video=float(tv[step]),t_audio=float(ta[step]),
                imgvid_cond_timestep=.999,audio_ref_cond_timestep=1.)
            comparisons={
                "video_input":(x["hidden_states"],out["x"][:,branch.img_pos_dev]),
                "audio_input":(x["audio_hidden_states"],out["audio_x"][:,branch.audio_pos_dev]),
                "prompt":(x["encoder_hidden_states"][0],out["prompt_embeds"]),
                "position_ids":(x["position_ids"],out["img_position_ids"][0,:used]),
                "tags":(x["token_tags"],out["token_tags"][:used]),
                "timesteps":(x["timestep"],out["unique_timesteps"]),
                "timestep_indices":(x["timestep_indices"],out["inverse_indices"][:used]),
                "video_indices":(x["video_indices"],out["img_pos_info"]["position_ids"]),
                "audio_indices":(x["audio_indices"],out["audio_pos_info"]["position_ids"]),
                "text_indices":(x["text_indices"],out["text_pos_info"]["position_ids"]),
            }
            for name,(trained,served) in comparisons.items():
                assert trained.shape==served.shape,(name,trained.shape,served.shape)
                assert torch.equal(trained,served),(i,step,name,float((trained-served).abs().max()))
            records.append(dict(sample_id=sample["sample_id"],step=step,passed=True,
                checked=list(comparisons),audio_rows=2*audio_t,latent_frames=latent_t))
        print(json.dumps(dict(sample_id=sample["sample_id"],all_eight_steps_passed=True,
            audio_rows=2*audio_t,latent_frames=latent_t)),flush=True)
    receipt=dict(passed=True,test="train_infer_full_inputs",audio_input_policy="fixed-silent",
        records=records,batch_sha256=file_sha(args.batch_module),
        note="candidate input parity, not full model/output or compiled backward acceptance")
    args.output.write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
