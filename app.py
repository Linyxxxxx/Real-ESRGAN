import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import glob
import numpy as np
import gradio as gr

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


if __name__ == "__main__":

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = "weights/RealESRGAN_x4plus.pth"
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model
    )

    def inference_block(img, n_blocks=4, pad_factor=10, outscale=4):
        w, h, c = img.shape
        w_block = w // n_blocks
        h_block = h // n_blocks
        w_pad = w_block // pad_factor
        h_pad = h_block // pad_factor

        output_full = np.ones((w * outscale, h * outscale, 3))

        for i in range(n_blocks):
            for j in range(n_blocks):
                # 切块
                w_is_first = not (w_block * i > 0)
                x1 = w_block * i if w_is_first else w_block * i - w_pad
                w_is_last = not (w_block * (i + 1) < w - 1)
                x2 = w_block * (i + 1) if w_is_last else w_block * (i + 1) + w_pad
                h_is_first = not (h_block * j > 0)
                y1 = h_block * j if h_is_first else h_block * j - h_pad
                h_is_last = not (h_block * (j + 1) < h - 1)
                y2 = h_block * (j + 1) if h_is_last else h_block * (j + 1) + h_pad

                img_block = img[x1: x2, y1: y2, :]

                # sr
                output_block, _ = upsampler.enhance(img_block, outscale=outscale)

                # merge
                if not w_is_first:
                    output_block[:w_pad*outscale, :, :] = \
                        output_block[:w_pad*outscale, :, :] * 0.5 + output_full[x1*outscale: x1*outscale+w_pad*outscale, y1*outscale: y2*outscale, :] * 0.5
                if not h_is_first:
                    output_block[:, :h_pad*outscale, :] = \
                        output_block[:, :h_pad*outscale, :] * 0.5 + output_full[x1*outscale: x2*outscale, y1*outscale: y1*outscale+h_pad*outscale, :] * 0.5

                output_full[x1*outscale: x2*outscale, y1*outscale: y2*outscale, :] = output_block

        if outscale is not None and outscale != 4:
            output_full = cv2.resize(
                output_full, (
                    int(h * outscale),
                    int(w * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output_full/255.


    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # input image
                input_image = gr.Image(label="输入图像", type='numpy')

                with gr.Tab(label='参数设置'):
                    n_blocks = gr.inputs.Slider(minimum=1, maximum=16, step=1, default=1, label="切块个数")
                    pad_factor = gr.inputs.Slider(minimum=1, maximum=20, step=1, default=10, label="融合系数")
                    outscale = gr.inputs.Slider(minimum=1, maximum=16, step=1, default=4, label="放大倍数")

                button = gr.Button("开始")
                 
            with gr.Column():
                result = gr.Image(label="超分结果", type='numpy')
             
        button.click(inference_block, inputs=[input_image, n_blocks, pad_factor, outscale], outputs=[result])

    PORT = 12358
    demo.launch(server_port=PORT)