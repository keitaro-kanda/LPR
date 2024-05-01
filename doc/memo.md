# バイナリデータの記録形式
参考：https://www.kushiro-ct.ac.jp/yanagawa/C-2015/16-0617/index.html
- char型
    - signed char（符号付き整数）型：-128 ~ 127
    - unsigned char（符号なし整数）型：0 ~ 255
- int型
    - 4byte = 32bitの情報量を持つ
    - リトルエンディアン：下位のバイトが先頭，上位のバイトが末尾に記録される
    - ビッグエンディアン：上位のバイトが先頭，下位のバイトが末尾に記録される
- double型
    - 8byte = 64bitの情報量を持ち，IEEE倍精度浮動小数点形式のビット列（２進数）として記録される

Lファイル中で明記されているデータ形式
- UnsignedByte → 符号なしchar型
- IEEE754MSBSingle
    - 単精度浮動小数点？
    - MSB (Most Significant Bit)：最上位ビット
    - MSBが先頭→ビッグエンディアン
- IEEE754LSBSingle
    - 単精度浮動小数点(4byte = 32bit)？
    - LSB：最下位ビット？
    LSBが先頭→リトルエンディアン
- UnsignedLSB2
    - LSB (Least Significant Bit)：最下位ビット，ビット列を２進数にした時に最も小さいくらいを表すビットのこと．

# ASCIIコード
参考：https://www.kishugiken.co.jp/technical/ascii%E3%82%B3%E3%83%BC%E3%83%89/
- 0x00 ~ 0xffまで，$16 \times 16 = 256$種類の文字を表現できる
- 0xは16進数であることを示す接頭字みたいなもん？