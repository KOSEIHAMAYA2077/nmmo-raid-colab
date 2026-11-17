# Colabで最後の1人の経験量・モデル幅を比較する

入口は [last_picker_scaling.ipynb](last_picker_scaling.ipynb)。この文書を作った段階ではColabへ接続せず、インストール・プロファイル・研究用学習を実行していない。Macでの短い合成テストと既存レイドの回帰テストだけで検証する。Linux環境のインストール成否、速度、長時間学習の成績は利用者の最初の実行で確認する。

## 比較するもの

- 幅64（28,685個／人）と幅128（106,317個／人）。味方埋め込み幅16、独立128人、ボス01、元の戦闘報酬を共通にする。
- 各人が本人として2,048・8,192・32,768回判断した時点（256・1,024・4,096更新）で保存・評価。同じ系列の途中経過を比較する。
- seed 41・42・43。各更新は全員A4回・B4回。モデル初期重みの形状は幅で異なるが、同seedの6人抽選・本人・席・A/B・環境seedの日程は一致する。
- 評価は従来の未使用1,536対。学習時は評価の6人集合を抽選から除外。通常／職表示反転×確率選択／最大確率選択の4評価を保存する。

[研究の設計書](../nmmo2/raid_nmmo/COLAB_LAST_PICKER_SCALING.md)・[実行モジュール](../nmmo2/raid_nmmo/scaling_last_picker.py)

## Notebookの順番

1. 固定コミットのコードを取得。毎回同じコードを使う。
2. `/content` に独立Python 3.10.16環境を用意する。Colabのカーネル自体は置き換えない。
3. Driveを利用者がマウントし、研究の永続保存先を指定する。
4. 検算・短い回帰テストを実行する。これは成績測定用の学習ではない。
5. `DO_PROFILE=True` にしてCPU／CUDA×両幅を各1更新だけ測る。設定を戻しても測定ファイルは残る。
6. 時間・容量の見積もりを見て、幅、seed、停止点を指定して `DO_TRAIN=True` で1系列ずつ実行する。
7. 完了した保存点を集計。次の停止点へ進むときは同じrun名を使い、`STOP_UPDATE` を256→1024→4096に増やす。

最初から全6系列×4,096更新を起動しない。同じ系列を複数ランタイムから同時に書き込まない。途中で切断されたら、同じコード・幅・seed・保存先で上から環境を作り直し、同じ学習セルを再実行する。`--stop-update` は追加回数ではなく累積の更新番号。

## CPU／CUDAの選択と計測

`--device cpu`／`--device cuda` を選べる。float32・TF32無効・TorchのCPU側1スレッドを共通にする。CUDAでは方策、入力、記憶、マスク、PPOの記録・収益をGPUへ送り、NMMOと準備環境の状態遷移はCPUで続ける。転送・同期を含めた速さを測る。Torchはcu126配布を使えば同じ環境でCPUもCUDAも動かせる。CPU配布を選んだ場合、CUDAは利用できない。[実装](../nmmo2/raid_nmmo/scaling_last_picker.py)・[収集とPPOへの受け渡し](../nmmo2/raid_nmmo/train_last_picker.py)

`profile --updates 1 --devices cpu cuda` は幅64／128それぞれの1更新を測り、JSONと比較用CSVへ保存する。抽選から圧縮ログまでの実時間と、シナリオ生成、準備観測、準備方策の推論、状態更新と履歴、戦闘、PPO更新、ログ圧縮、チェックポイント保存を記録。GPUの各計時では同期を入れる。CUDAがなければその条件を未実行として記録し、CPUの代用結果をCUDA欄に入れない。

同seedの初期パラメータと試合抽選はそろえるが、CPUとCUDAのサンプリング乱数・計算精度・軌跡のビット一致は保証しない。CPU／CUDAの学習系列は別フォルダで管理し、幅の比較では同じバックエンドを使う。MacではCUDAテストをスキップし、GPUの動作・速度・同一GPUでの再開一致はColabで確認する。

NMMO戦闘は既に一致検証した `CertifiedBattle` を既定とし、再利用できることを実行時に確認した編成のみ勝敗と戦闘点を再利用する。`--battle native` でも実行できる。幅間の比較では両方を同じモードにする。[高速化の条件と過去の一致検証](../nmmo2/raid_nmmo/EQUALIZED_FAILURE_REWARDS.md)

プロファイルは毎回ゼロからの使い捨て計測。研究系列の経験には加算しない。最初の1更新はキャッシュの温まりやGPU初期化を含み得る。短い実行からの時間・ログ容量外挿なので、全評価・Driveコピーの時間は別。最初の保存点までの実測で見積もりを更新する。

## 切断と保存のルール

`/content` に書き、16更新ごと・停止点・指定保存点でDriveへコピーする。突然切断された場合、未保存の最大15更新を同じ乱数状態からやり直す。更新境界の状態は以下を含む。

- 全128人のモデル、Adam状態、次の更新番号、個人別A/B回数。
- NumPy抽選器、NumPyのグローバル乱数、Python乱数、Torch CPUおよび初期化済みCUDAの乱数状態。
- 幅・seed・報酬・日程ハッシュ、計算用ソースのハッシュ、Gitコミット、パッケージ版、計算機情報。
- 戦闘の再利用結果。個体のGRU記憶は各試合ゼロに戻す仕様で、試合途中からは再開しない。

ログは更新範囲ごとの `chunks/uXXXXXX/trials.jsonl.gz`。希望職と確定操作の全履歴、実戦闘点、学習点を保存する。`stats.json` は更新ごとの損失と集計・計測時間。チェックサム付き `manifest.json` を最後に置き、範囲の連鎖を検証する。未完の `.pending-*` は再開点に数えない。コードやパッケージ、日程が異なる場合は黙って再開しない。

保存容量を抑えるため、**自動生成した再開用state.ptだけは、最新2点・初期点・指定の3保存点を残す**。古い中間点の再開状態を整理しても、試合ログ・統計・マニフェスト・保存点の重みは残す。利用者の既存ファイルや別実験は削除しない。チェックポイントの実サイズはプロファイルに出る。全ログ容量は学習長に比例する。

古い `latest.pt` / `final.pt` からのAdamなし継続はこの系列に混ぜない。`state.pt` はこのコードで自分が作り、チェックサムを照合したものを読み込む。

## Notebookを使わない場合

リポジトリのルートで実行する。下記は利用者がColabで実行するコマンドで、今回の作成作業では実行しない。

```bash
/content/nmmo-scaling-env/bin/python -m nmmo2.raid_nmmo.scaling_last_picker profile \
  --schedule runs/2026-11-16_last_picker_frozen/schedule.jsonl \
  --out /content/drive/MyDrive/NMMO/scaling/profile.json

/content/nmmo-scaling-env/bin/python -m nmmo2.raid_nmmo.scaling_last_picker train \
  --schedule runs/2026-11-16_last_picker_frozen/schedule.jsonl \
  --width 64 --seed 41 --stop-update 256 \
  --out /content/nmmo-scaling-runs/w64_s41 \
  --durable /content/drive/MyDrive/NMMO/scaling/w64_s41
```

## 結果の読み方

各保存点の `evaluations/uXXXXXX/` に4モードの全対・全手順と `summary.json` を置く。主指標は同じ本人がAでH・BでDを確定した対の割合。A/B別討伐率、逆選択、未確定、希望職・確定職の分布、個人別正答数、中央値、1対以上／全対正答の人数も出す。

`report` は完了した保存点のみを集計し、各名簿を別行で `comparison.csv`・`comparison.json`・`RESULTS.md` にする。個体差や途中時点を独立学習seedとして数えない。候補個体の新しい6人組での再確認、2人・6人の自由配役、地形・複雑なジョブはこの実装の範囲外。

## ローカルで確認したこと

Macの既存Python 3.9.6・Torch 2.8.0環境で19件中17件通過、CUDA実機が必要な2件はスキップ。CPUでは幅64・128の両方で、連続2更新と1更新で保存→別作業フォルダへDrive相当の領域から復元→2更新目の結果が一致した。次の512シナリオ、全128人の重み、Adam状態、全乱数状態、圧縮した操作ログ、損失と時間以外の統計を照合済み。戦闘はこの再開テストだけ決定的な合成戦闘を使う。既存の実NMMO戦闘・表示介入も回帰テストした。[機械可読の検証記録](verification/last_picker_scaling_cpu.json)・[テスト一覧](verification/last_picker_scaling_cpu.txt)

ColabのPython 3.10導入、CUDAでの実動作・再開・速度比較は未確認。利用者がノートブックの検証セルで同じ受け入れテストを実行し、結果をDriveに残せる。長時間学習やColabのランタイム起動は今回行っていない。[検証用入口](verify_last_picker.py)

## インストールの根拠

確認日：2026-11-17。NMMO 2.1.2のPython要件は3.7以上3.11未満。[PyPIの版別メタデータ](https://pypi.org/project/nmmo/2.1.2/)・[上流setup.py](https://github.com/NeuralMMO/environment/blob/98b6bbd/setup.py)

uvで指定Pythonを用意し、カーネル外の仮想環境を使う。[uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)・[環境指定](https://docs.astral.sh/uv/pip/environments/)

Torch 2.8.0のCPU配布またはCUDA 12.6配布を指定。NMMOのビルド前にNumPyとCythonを入れ、既存の固定requirementsを適用する。[PyTorch公式の過去版インストール](https://pytorch.org/get-started/previous-versions/)・[セットアップコード](setup_last_picker.py)
