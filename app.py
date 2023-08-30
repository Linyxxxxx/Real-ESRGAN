import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import glob
import numpy as np
import gradio as gr

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

MAX_BLOCK_SIZE = 480
MAX_RESULOTION = 20000
SAVE_PATH = "Output.jpg"

def store_img(img, outscale):
    w, h, c = img.shape
    longer = w if w > h else h
    new_nblock = longer // MAX_BLOCK_SIZE
    if new_nblock > 32: new_nblock = 32
    if new_nblock < 1: new_nblock = 1

    print("n_block", new_nblock)

    return (
        gr.update(label=f"输入图像（{w} x {h}）"),
        gr.update(value="开始", interactive=True),
        w, h,
        gr.update(label=f"超分结果（预计分辨率：{w*outscale}x{h*outscale}）"),
        gr.update(value=new_nblock),
    )

def clear_img(img):
    return (
        gr.update(label=f"输入图像"),
        gr.update(value="图像尚未上传，请稍候", interactive=False),
        gr.update(label="超分结果"),
    )

def on_outscale_slide_change(outscale, input_w, input_h):
    if input_w*outscale >= MAX_RESULOTION or input_h*outscale >= MAX_RESULOTION:
        return (
            gr.update(label=f"超分结果（预计分辨率：{input_w*outscale}x{input_h*outscale}）"),
            gr.update(value="超出最大分辨率限制，请调小放大倍数", interactive=False),
        )
    return (
        gr.update(label=f"超分结果（预计分辨率：{input_w*outscale}x{input_h*outscale}）"),
        gr.update(value="开始", interactive=True),
    )


if __name__ == "__main__":

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = "weights/RealESRGAN_x4plus.pth"
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model
    )

    def inference_block(img, n_blocks=4, outscale=4):
        pad_factor=10

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

        max_img_size = 1080
        new_w = w
        new_h = h
        if w > max_img_size:
            new_w = max_img_size
            new_h = int(h / w * new_w)
        if h > max_img_size:
            new_h = max_img_size
            new_w = int(w / h * new_h)
        output_original_size = cv2.resize(output_full, (new_h, new_w)) / 255.
        output_original_size[output_original_size > 1] = 1

        if outscale is not None and outscale != 4:
            output_full = cv2.resize(
                output_full, (
                    int(h * outscale),
                    int(w * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        out_w, out_h, c = output_full.shape

        cv2.imwrite(SAVE_PATH, output_full[:,:,::-1])

        return (
            gr.update(value=output_original_size, label=f"超分结果（{out_w} x {out_h}）过大，预览图像经过压缩"),
            gr.update(value=SAVE_PATH, visible=True),
        )


    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # <center>AI图片放大
            """
        )
        with gr.Row():
            with gr.Column():
                # input image
                input_w = gr.State(value=0)
                input_h = gr.State(value=0)
                input_image = gr.Image(label="输入图像", type='numpy')

                with gr.Tab(label='参数设置'):
                    # n_blocks = gr.inputs.Slider(minimum=1, maximum=32, step=1, default=4, label="切块个数")
                    n_blocks = gr.State(value=4)
                    # pad_factor = gr.inputs.Slider(minimum=1, maximum=20, step=1, default=10, label="融合系数")
                    outscale = gr.inputs.Slider(minimum=1, maximum=16, step=1, default=4, label="放大倍数")

                button = gr.Button("图像尚未上传，请稍候", interactive=False)
                 
            with gr.Column():
                result = gr.Image(label="超分结果", type='numpy', show_download_button=False, height=640)

                file = gr.File(SAVE_PATH, label="保存完整大小图像", visible=False)

        with gr.Row():
            gr.Markdown(
                """
                <center> <当前示例服务器同时运行多个程序，同时网络环境不稳定，处理速度不代表实际最终应用速度>
                <center> <最大分辨率受限于服务器内存，算法本身理论分辨率无限>
                """
            )


        input_image.upload(
            store_img,
            [input_image, outscale],
            [input_image, button, input_w, input_h, result, n_blocks]
        )

        input_image.clear(
            clear_img,
            [input_image],
            [input_image, button, result]
        )

        outscale.input(
            on_outscale_slide_change,
            [outscale, input_w, input_h],
            [result, button],
        )

        button.click(inference_block, inputs=[input_image, n_blocks, outscale], outputs=[result, file])

    PORT = 12360
    # demo.queue().launch(server_port=PORT)
    demo.queue().launch(server_port=PORT, share=True)