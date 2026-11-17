# ② 最後の1人として、味方に合わせる選択をゼロから学習する

2026-11-16。①は既存重みの凍結評価。②では同じ幅・観測・準備ルール・報酬で新規に初期化した128人の独立IPPOを学習する。後続の2人分担は別実験。

## 報酬の事前確認

[学習に使う戦闘](ready_battle.py)で、①の全1,536対を使ってA/B×本人T/H/Dの6条件を実行した。出発は確定済みとして各職で同じボス戦を行った。合計9,216試合。

| 味方 | 本人T | 本人H | 本人D |
|---|---|---|---|
| A・1T0H4D | 討伐0%、報酬0.630682 | 討伐100%、報酬2 | 討伐0%、報酬0.473011 |
| B・1T1H3D | 討伐0%、報酬0.886364 | 討伐0%、報酬0.897727 | 討伐100%、報酬2 |

各セル1,536試合。各試合の報酬はボス最大HPに対して削った割合＋討伐したら1、未出発は0。学習にそのまま使用し、職別ボーナスは加えない。[集計](../../runs/2026-11-16_last_picker_reward_audit/summary.json)・[生記録](../../runs/2026-11-16_last_picker_reward_audit/trials.jsonl)

## バランスを各人に保証する

1更新で128人それぞれを4つのランダムな6人組で本人にする。各組でA/Bを1回ずつ経験するため、1更新は1,024試合、各人A4回・B4回。味方5人は初めから完了済み、本人だけが未配役。本人は毎回未配役・GRUゼロで始まる。味方の発言は0固定、本人の通信は今までと同じ4記号から選ぶ。本人の職を当てるラベル、H不足のラベルは入力に含めない。

名簿seed 41・42・43を別々にゼロ初期化し、各256更新。各人の判断経験は2,048試合（A1,024＋B1,024）。各名簿で総262,144試合。初期重みと最終重み、各人のA/B回数、全試合の希望職・完了・操作履歴を保存する。既存のMonte Carlo型の独立PPO更新を使い、3エポック、Adam学習率3e-4、割引1、同じエントロピー係数とする。[学習コード](train_last_picker.py)・[モデルとPPO](ready_policy.py)

評価の6人集合を最初に取り置き、学習時には抽選し直して重複を避ける。評価は①と同一の全1,536対・環境seed・行動乱数。本人全128人が全6席で2回ずつ判断する。対のA/Bでは実際の編成以外の条件をそろえる。未使用の組とは全6人の集合を指し、全ペア初対面を意味しない。

## 主評価と表示介入

主指標は同一対でA=H/B=Dを確定した割合。AのH正答率とBのD正答率、逆選択、片方以上未確定、各条件の討伐率を名簿別に報告する。未確定とは完了を押さなかった場合で、途中の正しい希望職と区別する。

学習済みの同じ重みで、A/Bの実際の編成を固定したまま、変更対象の味方1人の職表示だけをH↔Dに入れ替える。操作できる本人の観測だけを各手で変え、実際の職・確定状態・他の味方の情報・発言・ボス・行動乱数は維持する。表示介入に追随すれば、正常表示でA=H/B=Dだった同じ対がA=D/B=Hへ変わることを確認できる。確率選択が主、最大確率選択が補助。[表示介入を含む凍結評価](last_picker.py)

## 再現

リポジトリのルートから、存在しない出力先を指定する。

```sh
nmmo2/.venv/bin/python -m unittest nmmo2.raid_nmmo.test_train_last_picker nmmo2.raid_nmmo.test_last_picker nmmo2.raid_nmmo.test_ready_prep
nmmo2/.venv/bin/python -m nmmo2.raid_nmmo.train_last_picker audit --schedule runs/2026-11-16_last_picker_frozen/schedule.jsonl --out runs/reward_audit_reproduce
nmmo2/.venv/bin/python -m nmmo2.raid_nmmo.train_last_picker train --seed 41 --updates 256 --pairs-per-member 4 --schedule runs/2026-11-16_last_picker_frozen/schedule.jsonl --out runs/last_picker_train_reproduce41
```

同じ学習スクリプトをseed 42・43で実行する。training.jsonl.gzの各行に名簿・本人・席・A/B・実際の味方編成・報酬と本人の履歴を保存する。historyの各手は[操作ID、通信記号、希望職ID、完了フラグ]。-1は未配役、0/1/2はT/H/D。initial/finalの評価JSONLは①と同じ書式。final_swappedは表示介入で実際の編成はA/Bのまま。
