# CRANE+ 物体把持：模倣学習とResidual PPO

このリポジトリには、CRANE+による立方体把持課題について、Gazebo上で収集した成功デモ、PyTorchによる模倣学習（Behavior Cloning）モデル、ROS 2での閉ループ評価ツール、Residual PPOによる補正学習、MP4録画機能が含まれています。

## ロボットと箱を初期状態に戻す

**CRANE+ Grasp**パネルで録画が実行中でないことを確認し、Mac側で次を実行します。

```bash
./tools/reset_grasp_scene.sh
```

このコマンドは、グリッパーを開き、アームの4関節をゼロ位置へ戻し、`wood_cube_5cm`をワールド内の初期位置へ戻します。録画中にリセットするとエピソードが無効になるため、`ros2 bag record`の実行中は安全のため処理を拒否します。

リセット完了後、パネルで`approach`を選択し、**START NEW EPISODE**を押します。

## 模倣学習用の遷移データを作成する

```bash
python3 tools/prepare_bc_dataset.py grasp_episode_20261101_161013_4446cd
```

変換結果は、対象エピソードの`bc_dataset/`ディレクトリへ出力されます。

- `transitions.jsonl`：計測した関節状態と、約0.2秒後のコントローラー目標値
- `summary.json`：関節順序、正規化値、区間境界、変換設定

長時間停止している区間は除外します。削除した時間区間をまたぐ目標値が作られないよう、連続していない動作区間は別々のセグメントとして扱います。

## 位置を変えた成功デモを自動収集する

GUIパネルで録画が実行中でないことを確認し、次を実行します。

```bash
./tools/run_auto_collection.sh 30
```

各エピソードでは、到達可能な狭い範囲内で箱の位置を変化させ、5関節の値と箱の位置を記録します。Gazebo上で箱を3 cm以上持ち上げた場合だけ成功と判定します。自動収集では容量の大きいrosbagを作成せず、軽量なJSONLデータを`datasets/`へコピーし、統合された`bc_collection/transitions.jsonl`へ変換します。

自動収集時の観測には`object_x`、`object_y`、`object_z`が含まれるため、箱の位置を変化させる意味があります。関節状態だけを入力する方策では、移動した箱の位置を推定できません。

現在の主要データセットは`datasets/grasp_auto_20261102_012854`です。30回中30回の把持に成功し、統合後は5,036遷移あります。計測範囲の詳細は[`datasets/README.md`](datasets/README.md)を参照してください。

## 模倣学習モデルを訓練する

データは遷移単位ではなくエピソード単位で分割します。現在の30デモは、訓練24、検証3、テスト3エピソードに分けられます。正規化に使う統計量は、訓練エピソードだけから計算します。

```bash
python3 tools/train_bc.py \
  datasets/grasp_auto_20261102_012854/bc_collection/transitions.jsonl \
  models/bc_grasp_retrained \
  --device cpu
```

このコマンドは、移植可能なTorchScript方策、正規化・データ分割メタデータ、関節別評価値、損失履歴、損失グラフを出力します。リポジトリに保存済みのモデルは`models/bc_grasp_v1/policy.ts`です。

## Gazeboで評価してMP4を録画する

`pai_ros2`コンテナとCRANE+のGazeboシミュレーションを起動した状態で、次を実行します。

```bash
./tools/run_bc_evaluation.sh 10
```

各エピソードではロボットと箱をリセットし、新しい箱位置を選び、シミュレーション時刻で0.2秒間隔に学習済み方策を実行します。計測した持上げ量が3 cm以上になった場合だけ成功と判定します。関節可動域と1.5 rad/sの速度制限も適用されます。

学習済み動作がグリッパーを閉じ始めると、その試行が終わるまで閉状態を維持します。これはすべての訓練デモと同じ動作であり、箱が一時的にずれたときに、内部状態を持たない方策がグリッパーを再び開くことを防ぎます。

実行完了時に出力先が表示されます。主な内容は次のとおりです。

- `run/summary.json`：箱位置、持上げ高さ、エラー、成功率
- `run/traces/`：各ステップの観測値と適用した動作
- `gazebo.mp4`：Gazeboデスクトップを録画したH.264動画

自動生成される評価ディレクトリはGitの管理対象外です。代表的な軽量動画と、検証済みの10エピソードの結果は`results/bc_grasp_v1/`に保存しています。

## Gazeboをブラウザで確認する

コンテナの起動中に、次のURLを開きます。

<http://localhost:6080/vnc.html?autoconnect=1&resize=scale>

評価を開始する前に、画面下部のタスクバーで**Gazebo**ウィンドウを選択してください。MP4録画はこのVNCデスクトップをそのまま記録するため、カメラの拡大率や視点変更も動画へ反映されます。

## 模倣学習の評価結果

- 未使用テストデータのMAE：0.00630 rad
- 現在値をそのまま予測するベースラインのMAE：0.03212 rad
- Gazebo評価：未使用のランダム位置で10回中9回の把持に成功
- 成功判定：30 mm以上の持上げ
- 全10試行の平均最大持上げ量：47.6 mm

正確なファイル構成と制約事項は[`results/bc_grasp_v1/README.md`](results/bc_grasp_v1/README.md)を参照してください。現在の観測では、Gazeboが直接出力する箱のXYZ座標を使用しており、カメラ画像を入力する方策にはなっていません。

## Residual PPOで補正学習する

ここではPPOをランダム動作から学習する代替方策としてではなく、模倣学習モデルの出力を小さく修正する補正方策として使用します。固定したBCモデルが5関節の基準目標値を出力し、PPOは関節角度、関節速度、箱と手先の位置差、持上げ高さ、経過時間を観測して補正を加えます。

アーム各関節の補正は最大±0.06 rad、グリッパーの補正は最大±0.10 radです。物理的な関節制限と速度制限は、PPOの出力に対しても適用されます。

コンテナ内の学習環境は次のコマンドで再構築できます。

```bash
./tools/setup_ppo_env.sh
```

Gazeboと`pai_ros2`コンテナを起動し、最初に環境の動作確認を行います。

```bash
PPO_CHECK_ONLY=1 ./tools/run_ppo_training.sh 8
```

続いて、短いスモークテストまたは長時間の学習を実行します。

```bash
./tools/run_ppo_training.sh 128
./tools/run_ppo_training.sh 10000
```

学習中のGazebo/VNC画面を録画する場合は、`PPO_RECORD_VIDEO=1`を指定します。学習全体の動画は容量が大きくなるため、短い実行での利用を推奨します。

```bash
PPO_RECORD_VIDEO=1 ./tools/run_ppo_training.sh 128
```

学習中ではなく、保存済み方策だけを録画する場合は次を実行します。

```bash
./tools/run_ppo_evaluation.sh 10
```

リセット、接近、把持、持上げを含む決定論的な10評価エピソードが、`evaluations/.../gazebo.mp4`へ記録されます。

学習結果はGit管理対象外の`rl_runs/`以下へ出力されます。完了した実行には、`ppo_residual.zip`、`summary.json`、`training.monitor.csv`、`training_steps.csv`、`evaluation_steps.csv`が含まれます。さらに、報酬曲線、報酬内訳、補正量分布、箱位置別の成功率ヒートマップ、エピソード別CSVも自動生成されます。動画を有効にした場合は`gazebo_training.mp4`も生成されます。

128ステップのスモークテストは処理全体の動作確認を目的としており、PPOがBCを改善したと判断できる学習量ではありません。新しいモデルを採用する前に、固定した複数seedで多数の評価エピソードを実行して比較する必要があります。

保存済みのスモークテスト結果は[`results/ppo_residual_smoke_v1/`](results/ppo_residual_smoke_v1/README.md)にあります。

5,056ステップの学習結果、グラフ、制約事項は[`results/ppo_residual_5k_v1/`](results/ppo_residual_5k_v1/README.md)にまとめています。最終モデルは`models/ppo_residual_5k_v1/ppo_residual.zip`です。

## GitHub

このワークスペースは`main`ブランチを使用し、次のリモートリポジトリへ接続しています。

```text
origin  https://github.com/Yukaeru/PAI-.git
```

容量が大きくなるため、生のrosbagディレクトリはGitの管理対象外です。軽量なJSONLサンプル、メタデータ、変換ツール、統合済みの模倣学習用遷移データはGitで管理します。
