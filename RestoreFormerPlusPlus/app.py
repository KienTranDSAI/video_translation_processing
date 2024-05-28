import os, sys
import argparse
import cv2
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from glob import glob
import cv2
import numpy as np

from RestoreFormer import RestoreFormer

if not os.path.exists('experiments/pretrained_models'):
    os.makedirs('experiments/pretrained_models')
realesr_model_path = 'experiments/pretrained_models/RealESRGAN_x4plus.pth'
if not os.path.exists(realesr_model_path):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O experiments/pretrained_models/RealESRGAN_x4plus.pth")

if not os.path.exists('experiments/RestoreFormer/'):
    os.makedirs('experiments/RestoreFormer/')
restoreformer_model_path = 'experiments/RestoreFormer/last.ckpt'
if not os.path.exists(restoreformer_model_path):
    os.system("wget https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt -O experiments/RestoreFormer/last.ckpt")

if not os.path.exists('experiments/RestoreFormerPlusPlus/'):
    os.makedirs('experiments/RestoreFormerPlusPlus/')
restoreformerplusplus_model_path = 'experiments/RestoreFormerPlusPlus/last.ckpt'
if not os.path.exists(restoreformerplusplus_model_path):
    os.system("wget https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt -O experiments/RestoreFormerPlusPlus/last.ckpt")

# background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=realesr_model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

os.makedirs('output', exist_ok=True)


# def inference(img, version, scale, weight):
def inference(img, version, aligned, scale):
    # weight /= 100
    print(img, version, scale)
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('too large size')
            return None, None
        
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'RestoreFormer':
            face_enhancer = RestoreFormer(
            model_path=restoreformer_model_path, upscale=2, arch='RestoreFormer', bg_upsampler=upsampler)
        elif version == 'RestoreFormer++':
            face_enhancer = RestoreFormer(
            model_path=restoreformerplusplus_model_path, upscale=2, arch='RestoreFormer++', bg_upsampler=upsampler)

        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            has_aligned = True if aligned == 'aligned' else False
            _, restored_aligned, restored_img = face_enhancer.enhance(img, has_aligned=has_aligned, only_center_face=False, paste_back=True)
            if has_aligned:
                output = restored_aligned[0]
            else:
                output = restored_img
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out.{extension}'
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None


# title = "RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Paris"
# important_links=r'''
# <div align='center'>
# [![paper_RestroeForemer++](https://img.shields.io/badge/TPAMI-Restorformer%2B%2B-green
# )](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.pdf)
# &nbsp; 
# [![paere_RestroeForemer](https://img.shields.io/badge/CVPR22-Restorformer-green)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.pdf)
# &nbsp;
# [![code_RestroeForemer++](https://img.shields.io/badge/GitHub-RestoreFormer%2B%2B-red

# )](https://github.com/wzhouxiff/RestoreFormerPlusPlus)
# &nbsp; 
# [![code_RestroeForemer](https://img.shields.io/badge/GitHub-RestoreFormer-red)](https://github.com/wzhouxiff/RestoreFormer)
# &nbsp;
# [![demo](https://img.shields.io/badge/Demo-Gradio-orange
# )](https://gradio.app/hub/wzhouxiff/RestoreFormerPlusPlus)
# </div>
# '''
# description = r"""
# <div align='center'>
# <a target='_blank' href='https://arxiv.org/pdf/2308.07228.pdf' style='float: left'>
# <img src='https://img.shields.io/badge/TPAMI-RestorFormer%2B%2B-green' alt='paper_RestroeForemer++'>
# </a>
# &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
# <a target='_blank' href='https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.pdf' style='float: left'>
# <img src='https://img.shields.io/badge/CVPR22-RestorFormer-green' alt='paere_RestroeForemer' >
# </a>
# &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
# <a target='_blank' href='https://github.com/wzhouxiff/RestoreFormerPlusPlus' style='float: left'>
# <img src='https://img.shields.io/badge/GitHub-RestoreFormer%2B%2B-red' alt='code_RestroeForemer++'>
# </a>
# &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
# <a target='_blank' href='https://github.com/wzhouxiff/RestoreFormer' style='float: left'>
# <img src='https://img.shields.io/badge/GitHub-RestoreFormer-red' alt='code_RestroeForemer' >
# </a>
# &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
# <a target='_blank' href='https://huggingface.co/spaces/wzhouxiff/RestoreFormerPlusPlus' style='float: left' >
# <img src='https://img.shields.io/badge/Demo-Gradio-orange' alt='demo' >
# </a>
# &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
# </div>
# <br>
# Gradio demo for <a href='https://github.com/wzhouxiff/RestoreFormerPlusPlus' target='_blank'><b>RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Paris</b></a>.
# <br>
# It is used to restore your Old Photos.
# <br>
# To use it, simply upload your image.<br>
# """

# article = r"""
# If the proposed algorithm is helpful, please help to ⭐ the GitHub Repositories: <a href='https://github.com/wzhouxiff/RestoreFormer' target='_blank'>RestoreFormer</a> and
# <a href='https://github.com/wzhouxiff/RestoreFormerPlusPlus' target='_blank'>RestoreFormer++</a>. Thanks! 
# [![GitHub Stars](https://img.shields.io/github/stars/wzhouxiff%2FRestoreFormer
# )](https://github.com/wzhouxiff/RestoreFormer)
# [![GitHub Stars](https://img.shields.io/github/stars/wzhouxiff%2FRestoreFormerPlusPlus
# )](https://github.com/wzhouxiff/RestoreFormerPlusPlus)

# ---

# 📝 **Citation**
# <br>
# If our work is useful for your research, please consider citing:
# ```bibtex
# @article{wang2023restoreformer++,
#     title={RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Paris},
#     author={Wang, Zhouxia and Zhang, Jiawei and Chen, Tianshui and Wang, Wenping and Luo, Ping},
#     booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)},
#     year={2023}
# }
# @article{wang2022restoreformer,
#     title={RestoreFormer: High-Quality Blind Face Restoration from Undegraded Key-Value Pairs},
#     author={Wang, Zhouxia and Zhang, Jiawei and Chen, Runjian and Wang, Wenping and Luo, Ping},
#     booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#     year={2022}
# }
# ```

# If you have any question, please email 📧 `wzhoux@connect.hku.hk`.
# """

# css=r"""

# """

# demo = gr.Interface(
#     inference, [
#         gr.Image(type="filepath", label="Input"),
#         gr.Radio(['RestoreFormer', 'RestoreFormer++'], type="value", value='RestoreFormer++', label='version'),
#         gr.Radio(['aligned', 'unaligned'], type="value", value='unaligned', label='Image Alignment'),
#         gr.Number(label="Rescaling factor", value=2),
#     ], [
#         gr.Image(type="numpy", label="Output (The whole image)"),
#         gr.File(label="Download the output image")
#     ],
#     title=title,
#     description=description,
#     article=article,
#     )

# demo.queue(max_size=20).launch(share="True")
# inference("/home/kientran/Code/Work/Overlayed video/RestoreFormerPlusPlus/simon.png", 'RestoreFormer', 'no', 2)
def video_inference(img, version, aligned, scale):
    # weight /= 100
    # print(img, version, scale)
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        # img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('too large size')
            return None, None
        
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'RestoreFormer':
            face_enhancer = RestoreFormer(
            model_path=restoreformer_model_path, upscale=2, arch='RestoreFormer', bg_upsampler=upsampler)
        elif version == 'RestoreFormer++':
            face_enhancer = RestoreFormer(
            model_path=restoreformerplusplus_model_path, upscale=2, arch='RestoreFormer++', bg_upsampler=upsampler)

        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            has_aligned = True if aligned == 'aligned' else False
            _, restored_aligned, restored_img = face_enhancer.enhance(img, has_aligned=has_aligned, only_center_face=False, paste_back=True)
            if has_aligned:
                output = restored_aligned[0]
            else:
                output = restored_img
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out.{extension}'
        # cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None


# if __name__ == "__main__":
#     cap = cv2.VideoCapture("/home/kientran/Code/Work/Overlayed video/Blend_lipsync/simon_talk.mp4")
#     while cap.isOpened():

#         ret, frame = cap.read()
    
        
#         if ret:

#             print(frame.shape)
#             out, savepath = video_inference(frame,'RestoreFormer', 'no', 2)
#             out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
#             out = cv2.resize(out, frame.shape[:2])
#             # print(out.shape)
#             # break
#             cv2.imshow("Video", out)

#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break



#     cap.release()



    
#     cv2.destroyAllWindows() 
