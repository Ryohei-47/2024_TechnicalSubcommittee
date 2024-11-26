## モデルに画像を認識させる
- 何枚かの写真を学習させる
- そのモデルを使用して好きな顔だけを表示させる<br>
→サンプル画像の中から好みの写真だけを表示

## 使う技術
### 仮想環境の構築
- pythonのインストールしてね、versionは3.10系
- python -m venv (仮想環境の名前)<br>
エラーが出たら↓
- 管理者権限でPowerShell開いて！
- .\venv\Scripts\activate

### メモ
- GCPの顔認識API使う（使い方はface_detect.pyとface_detect_crop.pyを参照）
- 必要なライブラリ関連
```
pip install tensorflow keras pillow
```
- ライブラリのバージョンで動かないので以下を実行する
```
pip install scipy
pip install numpy<2
pip install tensorflow==2.9
```
- 実行方法
```
モデルの作成
python data.py
```
- ディレクトリについて
```
face_imagesディレクトリ
→サンプルデータ
```
