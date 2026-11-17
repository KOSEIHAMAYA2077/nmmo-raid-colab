# nmmo-raid-colab

Neural MMO 2.1.2 を使った 6 人レイドの「最後の 1 人」実験を、Colab で動かすためのコードです。

- `colab/last_picker_scaling.ipynb`：Colab で開く入口。既定では学習も計測も起動しません
- `colab/LAST_PICKER_SCALING.md`：使い方と保存の仕様
- `nmmo2/raid_nmmo/`：レイドの実装と学習・評価・テスト
- `nmmo2/raid_proto/`：ボスとレイドの最小の環境
- `runs/2026-11-16_last_picker_frozen/schedule.jsonl`：固定した評価の日程

研究の記録や分析は別の非公開リポジトリにあります。ここに入っているのは、実験を再現するためのコードだけです。
