import sys
import os

from PlotNeuralNet.pycore.tikzeng import *
from PlotNeuralNet.pycore.blocks import *


def to_end_with_image(path_file, to='(3,0,0)', width=8, height=8, name="end_temp"):
    return r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """ + to + """ {\includegraphics[width=""" + str(width) + "cm" + """,height=""" + str(height) + "cm" + """]{""" + path_file + """}};
\end{tikzpicture}
\end{document}
"""


arch = [
    to_head('../PlotNeuralNet'),
    to_cor(),
    to_begin(),

    # input
    to_input('./../thesis/mmcs_sfedu_thesis/img/sky_segmentation/image_with_sky.png', to="(-1,0,0)"),

    # block-001
    to_ConvConvRelu(name='ccr_b1', offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                    height=40, depth=40),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),

    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', offset="(1,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),
    *block_2ConvPool(name='b3', botton='pool_b2', top='pool_b3', offset="(1,0,0)",
                     size=(25, 25, 4.5), opacity=0.5),
    *block_2ConvPool(name='b4', botton='pool_b3', top='pool_b4', offset="(1,0,0)",
                     size=(16, 16, 5.5), opacity=0.5),

    # Bottleneck
    # block-005
    to_ConvConvRelu(name='ccr_b5', offset="(2,0,0)", to="(pool_b4-east)",
                    width=(8, 8), height=8, depth=8),
    to_connection("pool_b4", "ccr_b5"),

    # Decoder
    *block_Unconv(name="b6", botton="ccr_b5", top='end_b6', offset="(2.1,0,0)",
                  size=(16, 16, 5.0), opacity=0.5),
    to_skip(of='ccr_b4', to='ccr_res_b6', pos=1.25),
    *block_Unconv(name="b7", botton="end_b6", top='end_b7', offset="(2.1,0,0)",
                  size=(25, 25, 4.5), opacity=0.5),
    to_skip(of='ccr_b3', to='ccr_res_b7', pos=1.25),
    *block_Unconv(name="b8", botton="end_b7", top='end_b8', offset="(2.1,0,0)",
                  size=(32, 32, 3.5), opacity=0.5),
    to_skip(of='ccr_b2', to='ccr_res_b8', pos=1.25),

    *block_Unconv(name="b9", botton="end_b8", top='end_b9', offset="(2.1,0,0)",
                  size=(40, 40, 2.5), opacity=0.5),
    to_skip(of='ccr_b1', to='ccr_res_b9', pos=1.25),

    to_end_with_image('./../thesis/mmcs_sfedu_thesis/img/sky_segmentation/image_mask.png', to="(37, 0, 0)")
]


def main():
    name_file = str(sys.argv[0]).split('.')[0]
    to_generate(arch, os.path.join("~/dev/BachelorDiploma/plotter", name_file + '.tex'))


if __name__ == '__main__':
    main()
