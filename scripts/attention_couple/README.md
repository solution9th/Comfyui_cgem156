複数プロンプトを領域分けしながら生成するノードです。UNetの計算が一回分（Cross Attentionのみ複数回）で済むため、高速に生成できます。

+ 一つ目のプロンプトはKSamplerに直接入力してください（pooled_outputはこのプロンプトを参照します。）

+ base_maskには一つ目のプロンプトの領域を指定するマスクを入れてください。

+ 二つ目以降は右クリック->add_inputで入力を適宜ふやして、cond_n, mask_nにそれぞれ入れてください。

すべてのプロンプトでマスクの値が0になっている領域がある場合エラーが起きます。

![image](https://github.com/laksjdjf/cgem156-ComfyUI/assets/22386664/b7680173-448a-42e1-8177-514f91931422)