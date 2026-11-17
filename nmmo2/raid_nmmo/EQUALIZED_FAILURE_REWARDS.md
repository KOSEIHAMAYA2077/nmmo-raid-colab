# 失敗部分報酬の非対称を取り除く診断比較

2026-11-17。元の②の戦闘報酬条件を対照として保持し、診断条件だけで失敗したAのDとBのHの学習報酬を同額にする。目的は、Bで最初にDを希望する頻度の低下が変わるか、同じ本人がA=H/B=Dを確定する対が増えるかを別々に測ること。

## 変えるもの

事前に測ったAのDの平均報酬0.47301136363636354とBのHの0.8977272727272733の平均、0.6853693181818183を両ケースの学習報酬に使う。出発して討伐に失敗したAのDとBのHだけが対象。正解の討伐点2、本人Tの報酬、未出発0はそのまま。実際のボス戦とその成績は従来のNMMO 2実装を使い、戦闘報酬と学習報酬を各試合に両方保存する。[元の実測](../../runs/2026-11-16_last_picker_reward_audit/summary.json)

この補正はA/Bの文脈と本人の確定職に依存する診断用の加工であり、環境そのものの戦闘報酬を変更するものではない。観測にはA/Bのラベル・正解職・不足職ラベルを追加しない。フラグなしでは元の戦闘報酬を学習に使う。[補正関数と呼び出し](train_last_picker.py)

## そろえるもの

元のseed41/42/43それぞれのinitial.ptと同一の初期重みを乱数seedから生成し、テンソル一致を確認する。各人A1,024＋B1,024試合、256更新、各更新で各人A4＋B4、モデル幅64、味方埋込16、1体28,685パラメータ、準備16手、完了時職固定、共通報酬、ボス01と手書き戦闘を維持する。元の学習設定・モデル・観測・戦闘コードと評価日程のハッシュを照合する。

A/Bの試合順、6人、本人、席、味方の職配置、環境seedを元ログと全試合で照合する。行動乱数の初期seedも同じにする。学習で方策が変わるため実際の行動履歴は後の更新で異なり得る。最初の更新の履歴は元条件と全試合で一致することを確認する。元のPPO更新方法・学習率・エントロピー係数も維持する。

## 見る順序

1. 学習の序盤（更新1〜85）・中盤（86〜170）・終盤（171〜256）で、Bの最初の希望職がDの率を元条件と比較する。一度でもDを希望した率、D確定率、Dを試した後にHで確定する条件付き率も出す。[既存の解析](analyze_exploration.py)
2. 元の②と同じ未使用1,536対でA=H/B=Dの確定率を主指標として比較する。A・Bの正答率、逆選択、未確定、正常表示と表示反転、確率選択と最大確率選択を名簿別に出す。[対比較](last_picker.py)

Hへの偏りだけが減っても補完の獲得とは扱わない。BのD確定が増えてAのH確定も維持され、同一対での両条件正答が増えるかを見る。評価で記録するteam_rewardは実際の戦闘点。診断用に加工した点で攻略成績を上乗せしない。

## 記録と検証

training.jsonl.gzのrewardはPPOへ渡した値（float32）、combat_rewardは実際の戦闘報酬。training.csvはmean_rewardとmean_combat_rewardを分けて記録する。settings.jsonに補正値、報酬モード、対照のinitial.ptとsettingsのハッシュ、元報酬表のハッシュを保存する。

6条件で補正対象2ケースだけが変わり、未出発と討伐成功とTが維持されることをテストする。全学習ログで初期重み、抽選順、最初の更新の履歴、報酬変換、各人A/B同数、未使用評価集合を照合する。[監査](analyze_last_picker_training.py)・[テスト](test_train_last_picker.py)

## 再現

リポジトリのルートで新しい出力先を指定する。seed42/43も対応する元条件のフォルダをreference-runに指定する。

```sh
nmmo2/.venv/bin/python -m unittest nmmo2.raid_nmmo.test_train_last_picker nmmo2.raid_nmmo.test_last_picker nmmo2.raid_nmmo.test_ready_prep
nmmo2/.venv/bin/python -m nmmo2.raid_nmmo.train_last_picker train --seed 41 --updates 256 --pairs-per-member 4 --schedule runs/2026-11-16_last_picker_frozen/schedule.jsonl --reference-run runs/2026-11-16_last_picker_train_s41 --equalize-failure-rewards --out runs/equalized_reproduce41
nmmo2/.venv/bin/python -m nmmo2.raid_nmmo.analyze_last_picker_training runs/equalized_reproduce41
python3 nmmo2/raid_nmmo/analyze_exploration.py runs/equalized_reproduce41 --out runs/equalized_exploration_reproduce41
```

## 戦闘結果の再利用による高速化と同一性確認

`--certified-battle` は、乱数で狙われる個体のIDが変わっても、学習に渡す勝敗・総報酬が一致する状態だけを再利用する。対象は、範囲攻撃なし、ボスの単体攻撃でDPSが一撃死する条件で、全生存者が同じ満HPのDPS、MP・CD・敵視が一致し、同率ターゲットが全生存者に一致する場合。この状態では全員が同じ攻撃を行い、一撃で1人ずつ減る。どのIDが死んでも以後の総攻撃量は同じになる。先行する別の同率抽選にこの条件を満たさないものがあれば、その編成は再利用しない。[実装](certified_battle.py)

全729編成×2乱数の1,458戦で元のNMMO実行と勝敗・総報酬が完全一致した。さらに3名簿それぞれについて、先行していた再利用なしの実行と128更新時点の全重みテンソル、全131,072試合の準備履歴・戦闘成績・学習報酬、全更新の集計と損失が一致した。[直接比較](../../runs/2026-11-17_equalized_acceleration_check/native_comparison.json)・[学習の一致](../../runs/2026-11-17_equalized_acceleration_check/replay_verification.json)

採用した3名簿は同じ初期重みからそれぞれ256更新を実行したもの。先行実行の128更新へ追加学習したものではない。先行実行は一致確認後に停止し、ローカルのequalized_native_partial_s41/42/43へ保存した。検証用の128更新スナップショットと、完全なgzipにした先頭131,072試合も保存する。[切替記録](../../runs/2026-11-17_equalized_acceleration_check/replacement.json)

高速化を含む再現は上の学習コマンドに `--certified-battle` を付ける。単位時間あたりの試合数は変わるが、対照と比較する各人の経験数はA/B各1,024回のまま。対応する全ログの一致を確認した範囲と、状態の対称性による再利用条件を区別して記録している。
