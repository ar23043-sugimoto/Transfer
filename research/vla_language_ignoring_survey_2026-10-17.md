# VLA が言語指示を無視する問題：文献調査（2026-10-17）

- 依頼: yuuya（2026-10-17）「これまでの研究は一旦置いておいて、VLA の言語指示の無視に関する論文を本格的に調査してほしい」
- 調査方法: Web 検索（arXiv、CVPR/ICLR/NeurIPS/CoRL のプロシーディング、プロジェクトページ、NVIDIA / Physical Intelligence の技術資料）。各論文の本文を読んで要点を抜き出した。本文の一部しか読めなかった論文は、その旨を各項に書いた。
- 出発点: `research/vla_vs_lbm_2026-10-17.md` §5 の 6 本（OpenVLA-OFT、Knowledge Insulation、LangGap、Stable Language Guidance、Wulff & Cangelosi、IGAR）。本メモはそれを含めて約 45 本に広げた。**2026-10-22 の再調査（yuuya「他にも見逃してる論文がないかもう一度調査してほしい」）で約 25 本を追加し、§8 と §7 末尾にまとめた。** **2026-10-24 に yuuya の依頼で範囲を言語条件付き世界モデル（WM）・World-Action Model（WAM）へ広げ、約 35 本を §9 に追加した。** 2026-10-25 の新規性調査（`research/vla_language_ignoring_novelty_2026-10-25.md`）で未収録 22 本を見つけ、同メモ §5 に載せた。
- 【推測】と書いた箇所は、論文に明記がなく私が補ったもの。

---

## 0. 結論（先に）

1. **現象は実在し、2025 年秋以降に独立した複数グループが再現している。** 代表的な証拠は (a) 指示を空文字列・無意味文字列・別タスクの文に置き換えても行動がほとんど変わらない（LIBERO-Plus、LIBERO-PRO、RADAR、VLA-Arena）、(b) 画像だけ・言語だけの除去実験で、言語だけでは 0%、画像だけでも 5〜7 割解ける（LIBERO-CF）、(c) 同じ配置で指示の意味だけ変える「反事実」評価で、SOTA モデルが学習時のタスクを実行してしまう（LIBERO-CF、LangGap、ICBench、LIBERO-PRO）。
2. **原因は「データ」と「モデル・学習」の両側にあり、文献上はおおむね合意がある。**
   - データ側: 既存データ・ベンチマークは「1 つの場面に 1 つのタスク」なので、言語を読まなくても場面から行動が決まる。言語の多様性が画像・行動に比べて圧倒的に少ない（modality imbalance）。ベイズ的に書くと p(a|v,ℓ) ∝ p(ℓ|a,v)·p(a|v)/p(ℓ|v) の言語項が情報を持たず、視覚事前分布 p(a|v) に崩壊する（BayesianVLA、BayesVLA、APT、CAG）。
   - モデル・学習側: 言語トークン数が視覚トークンに比べて 1〜2 桁少ない、action-query トークンの attention が視覚のアテンションシンクに吸われる（IGAR）、ランダム初期化の action expert からの勾配が VLM を壊す（Knowledge Insulation、APT）、少量データでの微調整が尖った極小に落ちて表現が崩れる（Flatness）、VLM の視覚・言語整合が「忘却」される（ChatVLA、VLM2VLA、parameter merging 系）。
3. **対策は 4 系統に整理できる。** (A) 条件付け方式の変更（FiLM、latent action query、視覚プロンプト、階層化）、(B) 学習法（勾配の絶縁、2 段階事前学習、SAM、パラメータ結合、対比学習、反事実ペア学習）、(C) データ（同一配置・複数指示の反事実データ、LLM による言い換え拡張、VLM による再ラベル、細粒度サブタスク注釈）、(D) 推論時ガイダンス（CFG 型の CAG / RAS、attention 再配分 IGAR）。**効果が大きいのは「同一場面で指示だけ違うデータを作る」系（C）と、それを前提にした学習法（B）**で、推論時ガイダンスは追加コストなしで効くが伸びは限定的。
4. **未解決の論点**: 多タスクで意味の多様性を増やすと学習容量が足りない（LangGap: 0%→28%）、言い換え（action 表現）は比較的直せるが**物体参照の言い換え**が難しい（LIBERO-Para）、chain-of-thought で正しく推論しても手が別の物を掴む（RoboSemanticBench: 失敗の 90% が CoT 正解・把持対象誤り）、ほぼ全ての評価が LIBERO 系のシミュレーションに偏っている。

---

## 1. 現象の報告（「本当に無視しているのか」の実証）

| 論文（時期） | 対象モデル | 実験 | 主な数値 | 一言 |
|---|---|---|---|---|
| **OpenVLA-OFT** Kim et al.（2025-02, arXiv 2502.19645） | OpenVLA 微調整 | ALOHA 複数指示タスク | FiLM なしは成功率 33%（指示をランダムに選んだのと同じ水準） | 「視覚入力の spurious correlation で言語追従が崩れる」と初期に明言 |
| **Knowledge Insulation** Driess et al.（2025-05, NeurIPS 2025, arXiv 2505.23705） | π0 系 | action expert を VLM と素朴に共学習 | 図 2: 「スプーンをゴミ箱へ」を無視してゴミを掴む | 勾配干渉で VLM の言語能力が落ちる |
| **LIBERO-Plus** Fei et al.（2025-10, arXiv 2510.13626） | OpenVLA、OFT、π0、π0-FAST、Nora、WorldVLA、UniVLA、RIPT-VLA の 10 構成 | 7 軸の摂動。言語は言い換え・空文字列・対象物の差し替え | OFT は言い換えで 97.1→81.5% と小さな低下。空指示でも Object スイートはほぼ不変（著者「VA モデルのように振る舞う」）。対象物を差し替えると成功率ほぼ 0 で、元の対象を掴み続ける | 言語摂動への「頑健さ」は言語を使っていない証拠だと解釈 |
| **LIBERO-PRO** Zhou et al.（2025-10, arXiv 2510.03827） | OpenVLA、π0、π0.5 | 物体・初期状態・指示・環境・タスクの摂動 | 指示を「xxx」にしても軌道がほぼ同じ。複合指示（ボウルを持ち上げてコンロをつける）は個別には出来ても結合すると失敗 | 「記憶（memorization）」であり理解ではない |
| **VLA-Arena** Zhang et al.（PKU, 2025-12, arXiv 2512.22539） | OpenVLA、UniVLA、π0-FAST、π0、OFT、SmolVLA | WordNet 置換 W0〜W4、指示除去 | LIBERO 学習モデルは指示を消しても約 28% しか落ちない。VLA-Arena 学習なら 52〜64% 落ちる | ベンチマーク側の指示が情報を持たないことを数値化 |
| **RADAR** Chen et al.（中山大, 2026-02, arXiv 2602.10980） | π0、π0-FAST、π0.5、RDT、OpenVLA | 実機 UR5e、自動リセット。言語は LLM 言い換え | 目次に「Paradoxical Insensitivity to Language」節があるが**本文該当部は読めず**。全体では π0.5 以外ほぼ 0/32 | 実機での同種報告【節の中身は未確認】 |
| **LIBERO-CF / CAG** Fang et al.（UNC, 2026-02, arXiv 2602.17659） | OFT、π0、π0.5、X-VLA | モダリティ除去と反事実指示 | LIBERO で π0.5 は 画像+言語 96.9%、**画像のみ 51.7%、言語のみ 0.0%**。OFT は 97.1 / 69.1 / 0.9。反事実指示では OFT の faithful grounding 4.7%、biased 83.6%（学習タスクを実行）。学習タスクの物体を場面から取り除くと grounding 39→64% | 「視覚が言語を上書きする」ことを最も直接に示した論文 |
| **LangGap** Hou & Zhao（2026-02, arXiv 2603.00592） | π0.5 | 同一配置で意味だけ変える 4 次元摂動 | 標準ベンチ 95% 超のモデルが摂動下で 0% | 「ほぼ言語を無視」 |
| **ICBench / IGAR** Zhang et al.（清華・SMU・復旦, 2026-03, arXiv 2603.06001） | π0、π0.5、OFT | 場面と矛盾する指示（属性・空間関係を変える） | 矛盾指示でも Spatial/Object の成功率 90% 超、LGS（通常と矛盾の差）は π0.5 Goal V4 で 4.0 | 矛盾を無視して実行＝linguistic blindness |
| **LIBERO-Para** Kim et al.（崇実大・中央大, 2026-03, arXiv 2603.28301） | OFT、π0.5、X-VLA、VLA-Adapter、Xiaomi-Robotics-0 | LIBERO-Goal（同一初期状態）で 13 種の言い換え 4,092 文 | 成功率低下 22.8〜51.9pt（VLA-Adapter 98.2→46.3%）。失敗の 79.5〜95.5% は計画レベル（別タスクの軌道） | 無視ではなく「表層のキーワード照合」。物体参照の言い換えが特に弱い |
| **RoboSemanticBench** Yu et al.（HIT ほか, 2026-06, arXiv 2606.02277） | OFT、GO1、DexVLA、TinyVLA、PD-VLA、π0、π0.5、GR00T N1.7、QwenGR00T | 問題の答えに対応するブロックを選んで置く | 把持成功率は 90% 超なのに正解ブロック選択はランダム以下（nSG が負）。π0.5 だけわずかに正 | 「掴めるが、どれを掴むかを言語で決めていない」 |
| **Wulff & Cangelosi**（Manchester, 2026-10, arXiv 2610.10178） | π0.5、GR00T N1.7 | activation / attribution patching、CKA | 同義語・上位語・存在しない物体名にはほぼ無反応。空文字列と方向語には強く反応。GR00T は早期層（0〜10）に言語感度が集中 | 機械論的解析。「言語が効くか」は測れるが「正しく効くか」は測れないと明記 |
| **When Instructions Retrieve Trajectories** Chen et al.（2026-09, arXiv 2609.39971） | π0.5、GR00T N1.7 | 失敗ロールアウトの行動解析と内部状態の読み出し | 失敗時は「元の行動を繰り返す」か「別の学習タスクに切り替える」 | **「無視」ではなく「指示を検索キーにして学習軌道を引き出している」**（instruction-action binding）という再解釈 |

補足として、より古い報告:
- **OpenVLA**（Kim et al., 2024-06, arXiv 2406.09246）の「language grounding」評価: 場面に 2 物体を置き、指示で片方を指定。RT-1-X 1.5〜5/10、Octo 2.5〜5.5/10 に対し RT-2-X 8.5、OpenVLA 7.5〜10。Internet 事前学習のないモデルは「正しい物体を操作できない」と記述。
- **STAR-Gen**（Gao et al., Stanford/GDM, 2025-03, arXiv 2503.01238）: 汎化の分類法。「意味（言語）汎化は LLM バックボーンがあっても全モデルで弱い」。言い換え＋物体属性の複合で π0 再実装 1/10。
- **Experiences from Benchmarking VLA**（Macquarie, 2025-11, arXiv 2511.11298）: 実機 ALOHA。「スポンジを鍋へ」で Clean Dish の手順を実行する失敗を「データセットに偏った grounding」と説明。

---

## 2. 原因分析

### 2.1 データ側: 場面がタスクを決めてしまう
- **1 配置 1 タスク**: LIBERO など既存ベンチは配置ごとに 1 タスクしか用意しないので、言語を読まずに解ける（LangGap、LIBERO-CF、VLA-Arena）。LIBERO-Goal だけは同一初期状態から複数タスクなので、画像のみモデルが 9.8〜12.4% に落ちる（BayesianVLA）。RoboCasa は画像のみ 44.6% vs 言語あり 47.8% でほぼ差がない（BayesianVLA）。
- **modality imbalance**: 言語の多様性が視覚・行動の多様性より桁違いに少ない（APT、BayesVLA、LA4VLA、CAST）。LIBERO-Para は「各物体が常に 1 つの正式名で呼ばれる」ことが物体参照の言い換えに弱い理由だと指摘。
- **ベイズ的定式化**: π(a|v,ℓ) ∝ p(ℓ|a,v)·p(a|v)/p(ℓ|v)。場面から指示が予測できる（p(ℓ|v) が鋭い）と、指示と行動の条件付き相互情報量が消え、方策は p(a|v) に崩壊する（BayesianVLA、CAG、BayesVLA、APT が同じ枠組み）。
- **模倣損失の性質**: 条件付き行動分布の台が狭いと、「接地した解」と「指示をキーにした近道」がデモ上で区別できない（When Instructions Retrieve Trajectories）。模倣損失は「どの物体を選んだか」という意味的決定を露出させない（RoboSemanticBench）。

### 2.2 モデル・学習側
- **トークン数の不均衡**: 言語トークンは数十、視覚トークンは数百〜数千（CAG）。
- **アテンションシンク**: action-query トークンが視覚の顕著なトークンに偏って注意し、指示トークンが抑圧される。attention-to-text は 0.08 程度（IGAR）。
- **勾配干渉・忘却**: ランダム初期化の action expert の勾配が事前学習済み VLM を壊す（Knowledge Insulation）。ロボット微調整で VLM の視覚・言語整合が上書きされる「spurious forgetting」（ChatVLA）。VLM の VQA 能力は OpenVLA / ECoT で多くのベンチが 0 になる（VLM2VLA）。
- **損失地形**: 少量データ微調整が尖った極小（Keskar sharpness 0.012、最大固有値 0.93）に落ち、COCO 物体トークンのクラスタが崩れる（Flatness, arXiv 2606.23641）。
- **条件付け経路**: GR00T N1.7 は cross-attention を 4 層ごとにしか置かず、言語感度は早期層に集中（Wulff & Cangelosi）。GR00T N1.5 の技術資料は「VLM の凍結、アダプタ MLP の簡素化、視覚・言語埋め込みへの LayerNorm 追加」が言語追従を大きく改善したと述べる（実機 GR-1 で言語追従率 46.6→93.3%）。【推測】埋め込みのスケール差が言語を埋もれさせていた可能性。
- **離散 vs 連続行動**: 離散行動（自己回帰）型は言語と同じ経路で共学習されるので比較的守られ、連続行動の action expert は無防備（APT、KI）。ただし OpenVLA-OFT は連続化した後も FiLM なしでは崩れる。

---

## 3. 評価ベンチマーク・診断プロトコル

| 名称 | 何を測るか | 設計上の工夫 | 備考 |
|---|---|---|---|
| LIBERO-Plus（2510.13626） | 7 軸摂動の頑健性 | 言い換え・空指示・対象差し替え | 言語は「落ちないこと」が逆に問題 |
| LIBERO-PRO（2510.03827） | 記憶 vs 理解 | 意味保存言い換え、タスク置換、複合指示、無意味文字列 | 数値は定性的記述が中心 |
| LIBERO-CF（2602.17659） | 反事実指示への忠実さ | 同じ配置に学習で見ていない実行可能な指示を割り当て。faithful / biased の grounding 率と成功率 | モダリティ除去実験も同梱 |
| LangGap（2603.00592） | 言語理解の有無 | 同一配置でタスクを全通り作る。4 次元の意味摂動 | 拡張データも提供 |
| ICBench（2603.06001） | 矛盾指示を拒めるか | 場面と矛盾する属性・空間語。LGS = 通常−矛盾の成功率差 | 「矛盾指示で成功してはいけない」逆向きの指標 |
| LIBERO-Para（2603.28301） | 言い換え頑健性 | 言語学の言い換え類型に基づく 13 種。PRIDE（言い換え距離で重み付けた成功） | 成功率は頑健性を 8〜22% 過大評価 |
| VLA-Arena（2512.22539） | 構造・言語・視覚の 3 軸難易度 | WordNet 置換 W0〜W4、指示除去テスト | 170 タスク、安全性指標も |
| RADAR（2602.10980） | 実機での動的摂動 | UR5e 自動セル、LLM 言い換え | 言語節は未読 |
| RoboSemanticBench（2606.02277） | 意味的な対象選択 | 問題の答えに対応するブロックを置く。nSG で把持と選択を分離 | CoT でも解決せず |
| STAR-Gen（2503.01238） | 汎化の分類法 | 意味軸 5 種（言い換え、多物体参照、内部知識…） | 評価設計の語彙として有用 |
| SimplerEnv-Instruct（2507.17520） | ゼロショット指示理解 | 80 タスク（集約 50、状況推論 30） | InstructVLA の評価用 |
| VLABench（ICCV 2025, 2412.18194） | 言語条件付き長期タスク | 常識・推論を要する指示 | 言語無視の直接診断ではない |
| STRONG-VLA（2604.10055） | 視覚＋言語の複合摂動 | 28 種（指示の破損を含む） | 詳細は abstract のみ確認 |

設計上の教訓（複数論文の共通点）:
1. **同一場面・複数指示**にしないと言語利用を測れない。LIBERO-Goal がその最小例。
2. 成功率だけでなく **grounding 率（正しい対象に触れたか）**、**biased 率（学習タスクを実行したか）**、**矛盾指示での成功率（低いほど良い）** を併記する。
3. 言い換えは「action 表現」と「物体参照」を分けて評価する（LIBERO-Para）。

---

## 4. 対策手法

### 4.1 アーキテクチャ・条件付け方式
| 手法 | 要点 | 効果 |
|---|---|---|
| **FiLM 注入**（OpenVLA-OFT, 2502.19645） | 言語埋め込みで視覚特徴をアフィン変調 | ALOHA 複数指示で 33% → 高成功率。LIBERO では不要だった理由は不明と明記 |
| **Latent Action Queries + ベイズ分解**（BayesianVLA/LangForce, HUST ほか, 2601.15197） | 学習可能クエリ 64 個。因果マスクで「視覚だけ見るクエリ」と「視覚＋言語を見るクエリ」を並行学習し、視覚だけのクエリが指示を説明できるよう LLR 損失を付ける。推論は posterior 枝のみ | SimplerEnv 55.2→66.5%、RoboCasa 47.8→50.4%。テキスト対話能力も保持 |
| **BayesVLA**（浙江大, 2512.11218） | 視覚-行動事前分布 × 言語条件付き尤度の分解。接触前後で分けた相を導入 | 数値は未確認 |
| **APT: Action Expert Pretraining**（浙江大, CoRL 2026, 2606.12366） | 1 段目: VLM を凍結し言語をマスクして action expert を視覚-行動事前分布として事前学習。2 段目: gated fusion で言語を注入 | LIBERO-PRO タスク摂動 avg: π0.5 11 → APT 19、VLM 微調整込み 27。実機 UO/UC で π0.5 11/20 → 17/20。タスク連鎖 0%→65% |
| **InSpire**（2505.13888） | 「物体はロボットのどちら側か」を先に答えさせてから行動 | LIBERO 未見タスク 3.6→13.6%（miniVLA）。言語無視を spurious correlation の一症状として扱う |
| **Interleave-VLA**（SJTU/UNC ほか, ICLR 2026） | 画像とテキストを交互に並べた指示（物体の切り抜き画像を指示に埋め込む） | SimplerEnv 新規カテゴリ 19.3→57.3%。言語を補う形で参照の曖昧さを回避 |
| **Point-VLA**（同済・清華ほか, 2512.18933） | 1 フレーム目に bbox を描いて指示に添える。π0.5 ベース | 実機平均 32.4%（テキスト）→ 92.5% |
| **VP-VLA**（HKUST/CUHK, 2603.22003） | System 2（Qwen3-VL）が対象と目標を同定し、SAM3 で十字線と bbox を画像に重畳。補助 grounding 損失 | RoboCasa 48.8→53.8%、実機 OOD で大幅改善。「gibberish でも性能が変わらない」先行報告を動機に挙げる |
| **ProGAL-VLA**（2604.09824） | 記号的サブゴール → 3D エンティティグラフ → 整合 cross-attention の「接地検証ボトルネック」。曖昧なら聞き返す | 言語無視を 3〜4 分の 1 に、Recall@1 0.41→0.71 |
| **SemanticVLA**（CVPR 2026） | VLM が明示的な軌跡座標（trace）を文字で出し、latent action token と相互補強 | LIBERO 97.0%、SimplerEnv 65.1%、言い換えに安定（abstract のみ確認） |
| **階層化**（Hi Robot 2502.19417、π0.5 の高位サブタスク予測、π0.7） | VLM が指示とフィードバックを解釈して短い原子指示に落とし、低位方策が実行 | π0.7 は「データ偏りに反する指示（Reverse Bussing など）」で先行モデルを上回ると報告。数値は図のみで未確認 |
| **GR00T N1.5 の変更**（NVIDIA 技術資料） | VLM 凍結、アダプタ簡素化、埋め込み LayerNorm、FLARE、DreamGen データ | Language Table 52.8→93.2%、実機 GR-1 言語追従率 46.6→93.3%。N1.6 資料は「言語追従と OOD タスク汎化は依然困難」と明記 |

### 4.2 学習法
| 手法 | 要点 | 効果 |
|---|---|---|
| **Knowledge Insulation**（π0.5-KI, 2505.23705） | action expert の勾配を VLM に流さず、VLM は離散行動＋VQA で共学習 | 言語追従の回復（定性）。APT も KI を併用 |
| **Sharpness-Aware Minimization**（Flatness, CMU, 2606.23641） | 微調整を SAM に変えるだけ。データ・構造の変更なし | π0.5: LIBERO-PRO Task 26.6→42.6、LangGap 24.5→41.7、LIBERO-CF 13.2→47.8。CFG（30.2 / 29.4 / 36.3）や CAG（21.7）より大きい。ガイダンスと併用可 |
| **パラメータ結合**（RETAIN, Berkeley, 2512.08333；Bring Back Language to Actions, NeurIPS 2026 WS） | 事前学習重みと微調整重みを線形補間、または task vector を SVD 分解して成分ごとに混合係数を学習 | RETAIN は OOD 場面で約 40% 改善。言語バックボーンだけ結合しても全体結合と同等。Bring Back Language は VLM 能力と行動予測の両立を主張（数値未確認） |
| **VLM2VLA: 行動を言語として学習**（Princeton, 2509.22195） | 行動チャンクを自然言語テキストに書き換え、LoRA のみで微調整 | VQA を 85% 以上保持。スペイン語・中国語・ヒンディー語の指示や「Ash Ketchum の上の物」など意味汎化で OpenVLA / ECoT を上回る。推論 6 秒超が難点 |
| **ChatVLA**（Midea, 2502.14420） | 段階的整合学習＋MLP の MoE | 理解タスク大幅回復。言語追従の直接評価はなし |
| **VLA 共学習**（π0.5 の WD データ、STAR-Gen の VQA 共学習） | Web の画像・言語データを混ぜる | STAR-Gen: VQA 共学習は言い換え・多物体参照・内部知識を改善するが物体属性は悪化。RoboSemanticBench では RoboVQA 共学習が逆効果（10.7→8.2%） |
| **対比整合**（HULC 2204.06252、GRIF CoRL 2023、RoVLA 2605.19678） | 言語と視覚（ゴール画像や軌跡）を対比学習で整合。GRIF は同一場面の他軌跡をハードネガティブに使う（これがないと場面照合になる） | HULC: 整合損失を外すと平均長 2.64→2.29。RoVLA は言い換え一貫性制約を提案（数値未確認） |
| **Equivariant Counterfactual Training**（2609.39971） | 同じ指示で別行動が必要になる視覚的に区別可能な場面を追加し、対のデモを同じ更新で学習 | LIBERO-PRO 位置交換 36→59%、実機 UR5e 未知位置 8→88% |
| **STRONG-VLA**（2604.10055） | 摂動カリキュラム → クリーンデータで再整合の 2 段階 | LIBERO で OFT +14.5pt（既知摂動） |
| **LA4VLA**（2606.27295） | デモを原子区間に分け低レベル言語記述を対応付け、言語-行動のみの事前学習を混ぜる | シミュ最大 +17.8pt、実機最大 +45pt |

### 4.3 データ
| 手法 | 要点 | 効果 |
|---|---|---|
| **DIAL**（Google, RSS 2023, 2211.11736） | CLIP を微調整して未ラベル軌跡に別表現の指示を自動付与 | 60 の新規指示で +41% |
| **CAST**（Berkeley, 2508.13446） | VLM で**反事実**ラベル（同じ観測で別の細粒度指示）を生成 | ナビで成功率 2 倍、ディストラクタ付き操作でも同様 |
| **LangGap 拡張**（2603.00592） | 同一配置で全タスクを作る | 単タスク 0→90%、多タスク 0→28% |
| **MCSI: LLM 言い換え**（Stable Language Guidance, 2601.04052） | 各指示を K 個に言い換え、期待損失で学習 | π0 破壊的書き換え平均 52.4→71.8%、RAS 併用で 82.2% |
| **VLA-IT データセット**（InstructVLA, ICLR 2026, 2507.17520） | 650K 件（キャプション、QA、指示書き換え、文脈生成）を 1:7 で混ぜる | SimplerEnv-Instruct 23.9（OpenVLA）→46.2% |
| **細粒度サブタスク注釈**（π0.5、π0.7、GR00T N1.6 資料） | 区間ごとに「冷蔵庫の扉を開ける」などの詳細ラベル | π0.7 はこれとサブゴール画像で参照表現・逆向き指示を改善。N1.6 資料は「改善はするが頑健な汎化には至らず」 |
| **合成指示拡張で OpenVLA 微調整**（2603.16044） | LLM でキーフレームから指示 5 本 | 小規模・評価が弱い（Top-1 は低下）。参考程度 |
| **摂動データで学習**（LIBERO-Plus） | 2 万軌跡を多様条件で収集 | 視覚摂動は大幅改善。言語に特化した結果は未確認 |

### 4.4 推論時ガイダンス（学習不要）
| 手法 | 式・仕組み | 効果と注意 |
|---|---|---|
| **CAG**（2602.17659） | π = π_uncond + ω(π_cond − π_uncond)。uncond は言語を落とした同じモデル（TF）か、言語なしで別途微調整した VA モデル | π0.5 LIBERO-CF grounding 30.8→46.3%、success 13.2→21.7%（VA）。ω を上げすぎると操作成功が落ちる。学習時のランダム言語ドロップアウトは逆効果（9.5%） |
| **RAS**（2601.04052） | 言語なしスコアを「アフォーダンス事前分布」とし、残差 Δ=s(a|o,ℓ)−s(a|o,∅) を γ 倍 | π0 +12pt。単独だと元指示でわずかに低下（94.2→90.7）。「Bias Suppressor」と位置づけ |
| **IGAR**（2603.06001） | 隠れ状態スパイクでシンクを検出し、言語シンクの注意を非シンク言語トークンへ再配分（前 16 層） | 矛盾指示での成功率を下げる（π0 Goal V4 76.4→36.4%、LGS 19.4→59.4）。π0.5 には効きにくい。+7ms |
| **TAG**（2603.24584） | CFG の「言語を消す」代わりに「対象物体を画像から消す」 | LIBERO 95.2→97.9%。言語追従そのものは測っていない |
| **ACG**（KAIST, 2510.22201） | 言語を落とす CFG は指示の小変化で行動分布が揺れて不安定だと指摘し、自己注意を潰した枝を使う | 行動の滑らかさ用。CFG が言語追従以外に効かない傍証 |

---

## 5. 系譜（時系列）

- **2022**: HULC（対比整合損失が効く）、DIAL（指示の自動多様化）。言語条件付き模倣学習の時代から「言語の多様性不足」は問題だった。
- **2023**: GRIF（言語はゴール画像より接地が難しい。同一場面のハードネガティブが必須）、RT-2 / OpenVLA 評価（Internet 事前学習のないモデルは言語 grounding で落ちる）。
- **2025 前半**: OpenVLA-OFT（FiLM）、Knowledge Insulation、STAR-Gen（意味汎化は全モデル弱い）、ChatVLA、InSpire、Hi Robot。
- **2025 秋**: LIBERO-Plus、LIBERO-PRO が「空指示でも動く」を広く周知。CAST、VLM2VLA。
- **2025-12〜2026-03（集中期）**: VLA-Arena、BayesVLA、BayesianVLA、Stable Language Guidance、LIBERO-CF/CAG、RADAR、LangGap、IGAR/ICBench、LIBERO-Para、VP-VLA、TAG、InstructVLA（ICLR）、Interleave-VLA（ICLR）。
- **2026-04〜10**: STRONG-VLA、ProGAL-VLA、RoVLA、RoboSemanticBench、APT（CoRL）、Flatness/SAM、LA4VLA、SemanticVLA（CVPR）、When Instructions Retrieve Trajectories、Wulff & Cangelosi、π0.7（steerability）。

---

## 6. 論点と未解決事項

1. **「無視」か「誤った接地」か**: 初期の報告は「無視（VA モデル化）」と表現したが、2026 年後半の解析（When Instructions Retrieve Trajectories、LIBERO-Para、Wulff & Cangelosi）は「言語は使われているが、学習軌道を検索するキーとしてしか使われていない」と再解釈している。対策の設計に影響する（単に言語の重みを上げるだけでは、別の学習タスクに飛ぶだけ）。
2. **物体参照の接地が最難関**: 言い換えのうち action 表現は比較的直るが、物体名の言い換えで 19.8〜51pt 落ちる（LIBERO-Para）。CoT が正しくても把持対象を誤る（RoboSemanticBench）。視覚プロンプト系（Point-VLA、VP-VLA、Interleave-VLA）が効くのはこの経路を言語から画像へ逃がしているから【推測】。
3. **多タスクでの容量不足**: LangGap の 0→28% は、現状の VLA が「同一場面で多数の指示を区別する」ことに根本的に弱い可能性を示す。
4. **推論時ガイダンスの限界**: CAG / RAS / IGAR はいずれも 10〜20pt 程度で、SAM（表現正則化）や反事実データに及ばない。π0.5 ではほぼ効かない報告が複数（IGAR、RAS）。
5. **共学習の効果は一様でない**: VQA 共学習が物体属性を悪化させた例（STAR-Gen）や RoboVQA 共学習が逆効果だった例（RoboSemanticBench）がある。
6. **評価の偏り**: ほぼ全てが LIBERO 系シミュレーション。実機での系統的診断は RADAR、CAG の Franka 実験、APT、ECT の UR5e 程度。
7. **アーキテクチャ依存**: 言語感度の場所と大きさはモデルごとに違う（GR00T N1.7 は早期 cross-attention 層、π0.5 は分散）。ある対策が別モデルで効かないのはこのためと考えられる【推測】。

---

## 7. 出典一覧（URL）

### 現象・診断
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- Knowledge Insulating VLA Models（π0.5-KI）: https://arxiv.org/abs/2505.23705
- LIBERO-Plus: https://arxiv.org/abs/2510.13626 ／ プロジェクト: https://sylvestf.github.io/LIBERO-plus/
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- VLA-Arena: https://arxiv.org/abs/2512.22539
- RADAR: https://arxiv.org/abs/2602.10980
- When Vision Overrides Language（LIBERO-CF / CAG）: https://arxiv.org/abs/2602.17659 ／ https://vla-va.github.io/
- LangGap: https://arxiv.org/abs/2603.00592
- IGAR / ICBench（ECCV 2026 ポスター）: https://arxiv.org/abs/2603.06001 ／ https://ray-nh.github.io/igar/
- LIBERO-Para: https://arxiv.org/abs/2603.28301
- RoboSemanticBench: https://arxiv.org/abs/2606.02277
- When Instructions Retrieve Trajectories（ECT）: https://arxiv.org/abs/2609.39971
- Wulff & Cangelosi, Do VLA Models Understand Instructions?: https://arxiv.org/abs/2610.10178
- OpenVLA（language grounding 評価）: https://arxiv.org/abs/2406.09246
- STAR-Gen: https://arxiv.org/abs/2503.01238
- Experiences from Benchmarking VLA Models: https://arxiv.org/abs/2511.11298
- VLABench（ICCV 2025）: https://arxiv.org/abs/2412.18194
- STRONG-VLA: https://arxiv.org/abs/2604.10055
- 36kr 解説記事（LangForce）: https://eu.36kr.com/en/p/3807332793081348
- Medium 解説（LIBERO-Plus/PRO）: https://medium.com/@yananchen1116/how-brittle-are-the-vla-models-in-robotics-66ab85286ecf

### 対策（アーキテクチャ・学習）
- BayesianVLA / LangForce: https://arxiv.org/abs/2601.15197
- BayesVLA（Seeing to Act, Prompting to Specify）: https://arxiv.org/abs/2512.11218
- APT（CoRL 2026）: https://arxiv.org/abs/2606.12366 ／ https://xukechun.github.io/papers/APT/
- Flatness Preserves Instruction Following（SAM）: https://arxiv.org/abs/2606.23641
- Stable Language Guidance（MCSI + RAS）: https://arxiv.org/abs/2601.04052
- InSpire: https://arxiv.org/abs/2505.13888
- Interleave-VLA（ICLR 2026）: https://proceedings.iclr.cc/paper_files/paper/2026/file/444876e32cba0408c63f5295481cfe20-Paper-Conference.pdf
- Point-VLA: https://arxiv.org/abs/2512.18933
- VP-VLA: https://arxiv.org/abs/2603.22003
- ProGAL-VLA: https://arxiv.org/abs/2604.09824
- SemanticVLA（CVPR 2026）: https://openaccess.thecvf.com/content/CVPR2026/papers/Ni_SemanticVLA_Towards_Semantic_Reasoning_over_Action_Memorization_via_Synergistic_Explicit_CVPR_2026_paper.pdf
- RoVLA: https://arxiv.org/abs/2605.19678
- LA4VLA: https://arxiv.org/abs/2606.27295
- VLM2VLA（Actions as Language）: https://arxiv.org/abs/2509.22195
- ChatVLA: https://arxiv.org/abs/2502.14420
- RETAIN（Robust Finetuning via Parameter Merging）: https://arxiv.org/abs/2512.08333
- Bring Back Language to Actions（NeurIPS 2026 RoboPAD WS）: https://neurips.cc/virtual/2026/162887
- HULC: https://arxiv.org/abs/2204.06252
- GRIF: https://bair.berkeley.edu/blog/2023/10/17/grif
- Hi Robot: https://arxiv.org/abs/2502.19417
- π0.5: https://arxiv.org/abs/2504.16054
- π0.7: https://www.pi.website/download/pi07.pdf
- GR00T N1.5 技術資料: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- GR00T N1.6 技術資料: https://research.nvidia.com/labs/gear/gr00t-n1_6/

### 対策（データ）
- DIAL（RSS 2023）: https://arxiv.org/abs/2211.11736
- CAST: https://arxiv.org/abs/2508.13446
- InstructVLA / VLA-IT（ICLR 2026）: https://arxiv.org/abs/2507.17520
- Synthetic Instruction Augmentation for OpenVLA: https://arxiv.org/abs/2603.16044

### 対策（推論時）
- TAG: https://arxiv.org/abs/2603.24584
- ACG: https://arxiv.org/abs/2510.22201

### 周辺
- Do What? Reject the Impossible（偽前提指示の検出）: https://arxiv.org/abs/2508.16292
- PAIR-VLA（視覚頑健 RL 微調整、言語は対象外）: https://arxiv.org/abs/2605.13105
- Decoupling Vision, Language, and Action（DEM。言語を消すと 11.3%、言語エンコーダは凍結で十分）: https://arxiv.org/abs/2609.18374
- DILL（視覚ショートカットの分離。言語は直接扱わず）: https://arxiv.org/abs/2609.37165
- Bridging Language and Action サーベイ: https://arxiv.org/abs/2312.10807

### 読めなかった・未確認の箇所
- RADAR の「Paradoxical Insensitivity to Language」節の数値（本文該当部が取得できず）
- Bring Back Language to Actions の本文（OpenReview が取得不可。要旨のみ）
- Flatness の実機ベンチの内訳、SemanticVLA の本文、BayesVLA の数値、RoVLA の数値
- π0.7 の言語追従の数値（図のみ）

---

## 8. 再調査で追加した論文（2026-10-22）

再調査の方法: 前回と違う語彙（grounding gap、instruction reliance、paraphrase、red teaming、how-level、premise conflict、shortcut learning、verification、steering）、会議プロシーディング（CoRL 2025、ICLR/CVPR/ECCV 2026）、既出論文の被引用、運転・ナビゲーション領域、技術レポート（π0、GR00T N1、Gemini Robotics、Qwen-RobotManip）で再検索した。

### 8.0 再調査の結論
- **前回の結論（§0）を覆す論文はなかった。** ただし次の 3 点が強化・更新された。
  1. **「言語は符号化されているが制御に使われていない」という機械論的証拠が 2 本独立に出た**（Encoded but Not in Control、Not All Features Are Created Equal）。π0.5 では指示カテゴリを線形プローブで 99.3% 復号できるのに、指示を変えても行動は統計的に変わらない（ANOVA p=0.247）。指示トークンの attention 寄与は LIBERO で 18〜28%、RoboTwin では 1.7%。これは §6-1 の「無視か誤接地か」論争に対し、**「情報はある。経路が弱い」**という第三の答えを与える。
  2. **実機・運転・空間語・How-level 指示へ対象が広がった**（Spa-Bench、RoboTwin-IF、Language2Drive、ICR-Drive、SIMPLER-IN、ConflictVLA-Bench、Penn の π0 実機評価）。運転 VLA でも「成功した走行の 93.2% は、同じ場面で別の指示を与えると失敗する」という同型の報告がある。
  3. **新しい対策カテゴリ**: 推論時の「言い換え＋検証」（CoVer、Rephrase Before You Act）、反事実介入を学習に組み込む（CofactVLA）、行動トークナイザを言語と整合させる（SALT）、red teaming で脆弱な指示を自動生成する（DAERT）。一方で「言い換えの多数決」は効かないという否定的結果（Science Tokyo）も出た。

### 8.1 現象・診断の追加

| 論文（時期） | 対象 | 何をしたか | 主な数値・知見 |
|---|---|---|---|
| **Encoded but Not in Control**（HKU・上海 AI Lab ほか, 2026-10, arXiv 2610.06235） | π0.5、GR00T N1.7、FastWAM、LingBot-VA | 場面固定で指示だけ差し替え、layerwise action lens、線形プローブ、attention 寄与分解、semi-NMF | LIBERO-Object で対象を差し替えても 86.5〜98.3% が元の対象へ向かう（指示された対象へは 0.5〜13%）。実機でも元対象 74〜91% vs 指示対象 3〜6%。対象・場面の同一性はプローブでほぼ完全に復号できる。微調整後、中間層のクラスタが「対象ベース」から「場面ベース」へ再編。著者は **representation-to-control grounding gap** と命名。相関的解析であり因果は示していないと明記 |
| **Not All Features Are Created Equal**（Case Western, 2026-03, arXiv 2603.19233） | π0.5、OFT、X-VLA、SmolVLA、GR00T N1.5、ACT | 活性注入、反事実プロンプト（空・否定・誤物体）、per-token SAE、線形プローブ。39 万ロールアウト | π0.5 は空プロンプトで行動の cos 類似度 0.775 に落ちるが、層 0 の活性を注入するだけで 0.997 に戻る。**言語が効くのは場面がゴールを決めないスイートだけ**（X-VLA: libero_goal 94%→約 10%、libero_object は 60〜100% のまま）。π0.5 はプロンプト変化の効果が無視できる（p=0.247）のにプローブは 99.3%。Action Atlas（action-atlas.com）を公開 |
| **Paraphrase Consensus Is Not Self-Consistency**（東京科学大学, 2026-10, arXiv 2610.19438） | OpenVLA、OFT、π0.5 | 「指示依存度（language share）」「言い換えコスト」を定義。K 個の言い換えの medoid 行動を実行する LCC を検証 | 言い換え 1 つで OpenVLA は LIBERO-Spatial −38pt。**言い換えの多数決では回復しない**（−16.8〜−32.4pt）。OFT は Goal で成功の 100% が指示依存、Object では 2%。言い換えデータで LoRA 事後学習すると −38.0→+2.0pt に回復し、指示依存も保たれる。提言: 学習時の文だけでなく held-out 言い換えでの成功率を併記 |
| **Rephrase Before You Act**（UCLA, 2026-10, arXiv 2610.10526） | π0（言い換え拡張済み）、π0.5 | 語句の 1 語差し替えで成功率がどれだけ揺れるかを網羅探索 | 「switch on the stove」100% vs「switch on the hot plate」2%（π0.5）。言い換え拡張済みの π0 でも最大 61pt 揺れる。微調整に含まれるタスクほど揺れが大きい（44% vs 5% のペアで 18pt 以上の差）。対策は §8.3 |
| **DAERT: 多様性を考慮した red teaming**（中山大・HKUST, 2026-04, arXiv 2604.05595） | π0、OpenVLA（LIBERO、CALVIN、SimplerEnv） | VLM を RL で学習させ、意味を保ったまま方策を失敗させる指示を生成 | π0 の LIBERO 平均成功率 93.33%→5.85%。「no action」指示でも π0 17.65%、π0.5 54.90% が成功（言語依存の弱さ）。生成した指示を学習データ拡張に使うことを提案 |
| **Robust Skills, Brittle Grounding**（2026-02, arXiv 2602.24143） | SmolVLA、π0.5 | 多物体ピッキングで物体配置の多様性を段階的に上げ、物体-位置の組を held-out に | 操作プリミティブの実行は安定だが指示条件付き成功は配置が難しくなると崩れる。「技能獲得と指示追従は分離している」。数値は本文未確認 |
| **Spa-Bench**（Imperial, 2026-10, arXiv 2610.12696） | VLA-0、MolmoAct2、π0.5、GR00T N1.7（凍結 / 全微調整）。実機 SO-101、4,092 エピソード | 空間語（相対サイズ・序数・関係・個数）で対象を指定。同じ場面を「対象名を直接言う」指示と比較 | 直接指示 81.7〜89.2% vs OOD 空間指示 40.7〜52.0%（差 28〜40pt）。Counting の OOD は 3 モデルとも 0/150。言い換えは空間指示で全モデル低下。GR00T 凍結 LLM は直接指示でも 81.7→70.0% |
| **ConflictVLA-Bench**（吉林大, 2026-09, arXiv 2609.31792） | OpenVLA、OFT、π0、π0-FAST、π0.5、GR00T N1.7、VLA-JEPA、UniVLA | 指示が場面・物理と矛盾する 2,826 タスク。矛盾なしの参照ロールアウトと対にする | 矛盾下でも元のゴールを達成し続ける: π0.5 77.2%、OFT 71.1%。失敗時も元の対象へ接近（target approach ratio 0.59〜0.74）。「前提を確認せよ」とプロンプトに書いても狙った方向には変わらず、正常タスクも崩れる |
| **Do VLAs Really Follow Instructions? How-level**（2026-10, arXiv 2610.21251） | 未確認 | 「何を」ではなく「どのように」（速度・経路など）を指定する指示。SIMPLER-IN ベンチ | 現状 VLA は How-level 指示から利益を得ず、むしろ成功率が下がる。原因仮説: 1 エピソード 1 指示の学習。区間ごとの詳細指示で学習すると反応はするが成功率は一貫しては上がらない |
| **RoboTwin-IF**（Qwen-RobotManip 技術報告, 2026-06, arXiv 2606.17846） | Qwen-RobotManip、π0.5、StarVLA、GR00T N1.7 | RoboTwin 2.0 上で held-out 指示テンプレートの 5 スイート。「言語を本物の制御信号として使うか視覚ショートカットか」を測る設計 | 平均 Qwen-RobotManip 72.2% vs π0.5 49.6%、GR00T N1.7 16.6%。ベンチ特化微調整で「VLA が VA（vision-action パターンマッチャ）に退化する」現象を **VLA-to-VA degradation** と命名。対策は dual-stream 共学習（ロボット:VL = 9:1、λ=0.1） |
| **Shortcut Learning in Generalist Robot Policies**（UESTC・同済, CoRL 2025, arXiv 2508.06426） | Diffusion Policy、MiniVLA、π0（LIBERO）、OXE 解析 | データセットが「断片化（サブデータ間の支持が重ならない）」かつ「低多様性」だと、タスク関連因子（言語・物体位置）と無関連因子（視点・背景）の相互情報量が上がり近道が学ばれる、という理論 | SIMPLER で OXE 学習方策 3 つが「スプーンをタオルへ」と言われて「コーラを掴む」。実機 π0 が視点に紐づいた学習時の指示を実行。言語入力のない DP は配置多様化で近道が減らず、言語手がかりが効いていると解釈。提言: 各サブデータ内で関連・無関連因子を独立に変える、重要因子はサブデータ間で重ねる |
| **π0 論文の言語追従実験**（Physical Intelligence, 2024-10, arXiv 2410.24164 §VI-B） | π0、π0-small（VLM なし） | 片付け・配膳・袋詰めで、全体指示のみ / 人の中間指示 / 高位 VLM の中間指示 | π0 の言語追従精度は π0-small より有意に高く、VLM 初期化の効果と結論（ただしサイズの交絡あり）。数値は図 9 のみ |
| **GR00T N1 論文**（NVIDIA, 2025-03, arXiv 2503.14734） | GR00T N1 | 言語追従の独立評価はなし。GR-1 シミュ（ディストラクタあり）は「言語に注意する必要がある」と記述 | GR-1 平均 50.0%（DP 32.7%）。「より強い VL バックボーンが言語理解を改善する」と限界に明記。N1.5 で言語追従率 46.6→93.3%（§4.1） |
| **Gemini Robotics 技術報告 / Veo 世界シミュレータ評価**（GDM, 2025-03 arXiv 2503.20020 / 2025-12 arXiv 2512.10675） | Gemini Robotics | 指示の言い換え・タイポ・他言語・具体度を変える評価（80 場面×指示）。Veo シミュ評価では「Object」軸の失敗の多くが指示追従の誤り（より馴染みのある物体へ向かう） | 言い換え種別ごとの数値は本文に無し |
| **Evaluating π0 in the Wild**（Penn PAL Lab ブログ, 2025） | π0-FAST-DROID | 非公式な実機試行 | 「右から左へ折る」は 0%（常に左から右へ）。「Close the toilet」0% vs「Close the white lid of the toilet」100%。無意味文字列でペンを掴む（DROID の物体の 16.67% がペン）。ベンチではないが、§1 の現象が実機で日常的に起きる例 |
| **Language2Drive**（2026-10, arXiv 2610.16634） | 運転 VLA（CARLA、実世界再構成、open-loop） | 場面固定で指示だけ変える paired counterfactual unit | **成功走行の 93.2% は、別の実行可能な指示では失敗**。「one-sided completion」（好みの行動だけ実行）が支配的 |
| **ICR-Drive**（Texas Tech・Bosch, 2026-04, arXiv 2604.05378） | LMDrive、BEVDriver（LangAuto） | 指示の言い換え・曖昧化・誤誘導の反事実 | 誤誘導指示で Driving Score −33pt、曖昧化で −17.65pt。一部の意味保存の言い換えでは逆に上がる（非単調） |
| **VLA-Retain**（NUS・0G Labs, 2026-10, arXiv 2610.27698） | 5 モデル（LIBERO-Plus） | 視覚モジュールを特定条件に微調整すると他条件がどう変わるか | X-VLA をカメラ条件に微調整すると目標条件 +69.4pt だが言語摂動 −6.6pt。10 構成中 8 で目標外が平均マイナス。言語追従そのものの評価ではなく「摂動軸の一つ」 |

### 8.2 原因分析の追加
- **符号化と制御の乖離**（Encoded but Not in Control、Not All Features）: 指示の情報は VLM 側に残っている。失われているのは action expert への経路で、attention 寄与が小さい（1.7〜28%）、視覚側の attention が元の対象に貼り付く、微調整で中間層が場面ベースに再編される。π0.5 では「expert 経路＝運動プログラム、VLM 経路＝ゴール意味」と分業しており、expert 側の注入の方が行動を 2 倍動かす。
- **データの断片化理論**（Shortcut Learning, CoRL 2025）: OXE のサブデータは視覚・言語ともに多様性が低く、クラスタが重ならない。この構造自体が近道学習の十分条件になる。§2.1 の「1 配置 1 タスク」の一般化。
- **1 エピソード 1 指示**（How-level 論文、CAST、LA4VLA）: What-level だけでなく How-level（速さ・経路）の指示も、同じエピソードに複数の記述が付かない限り学ばれない。
- **行動トークナイザの情報損失**（SALT, arXiv 2608.10484）: L1/L2 再構成だけで学ぶ離散行動トークンは動詞情報を落とす（連続軌道 1.26 bit に対し再構成型は下回り、圧縮率が上がるほど悪化）。
- **ベンチ特化微調整が VA 化を進める**（Qwen-RobotManip の「VLA-to-VA degradation」、VLA-Retain）。
- **設計要因**（VLANeXt, arXiv 2602.18532 の LIBERO-Plus 言語列）: 過去フレーム履歴を VLM に入れると言語頑健性が 89.2→51.5 に落ちる。VLM-方策の結合は soft が最良（89.2）。Goal スイートが最も弱い（51.5）。これらは「摂動下の成功率」であり言語依存の直接測定ではない。
- **言語は主ボトルネックではない、という反論**（VLM4VLA, 清華・Alibaba, arXiv 2601.03309）: モダリティ別の除去実験では視覚モジュールが主なボトルネックで、言語側だけの微調整はほぼ効かない（27.6→28.0%）。言語無視の問題と矛盾はしない（言語を「使わせる」ことと「言語能力の上限」は別）【推測】。

### 8.3 対策の追加

| 系統 | 手法 | 要点 | 効果 |
|---|---|---|---|
| 推論時・検証 | **CoVer**（Stanford, 2026-02, arXiv 2602.12281） | 「意図と行動のギャップ」を埋めるため、VLM で指示を言い換え、対比学習した 1B の検証器（画像・言語 ↔ 行動の InfoNCE）で言い換えと行動チャンクを選ぶ | SimplerEnv で π0 41.5/29.7% → 57.0/61.0%（ID/OOD）。言い換え拡張学習だけ（44.0/48.7）より大きい。実機 WidowX で +30〜60%。2.2Hz と遅い |
| 推論時・前処理 | **Rephrase Before You Act**（UCLA, 2026-10） | 凍結 VLA の言い換えごとの成功率を測り、LLM に「plate を dish に変えるな」等の 10〜20 ルールを蒸留。実行前に指示を 1 回書き換える | π0 SIMPLER の sealed 12 タスクで相対 +16〜27%。CoVer はこの設定ではベースライン以下（21.6 vs 24.5%） |
| 推論時・否定的結果 | **LCC 言い換え多数決**（東京科学大学, 2026-10） | 言い換え K 個の行動の medoid | どの設定でも改善せず。学習時に言い換えを入れる方が良い |
| 学習・反事実介入 | **CofactVLA**（2026-08, arXiv 2608.04396） | 言語をマスクした反事実枝を同じ前向き計算で走らせ、flow の速度場から視覚バイアス成分を直交射影で除去（OPG）＋特徴の共分散差の正固有空間を罰する（CCR） | 実機 OOD で +52.3pt（絶対）。ベースモデル名・シミュ数値は要旨で未確認 |
| 学習・トークナイザ | **SALT**（Bisk グループ, 2026-08, arXiv 2608.10484） | VQ-VAE の行動トークンから凍結言語モデルが指示を復元できるよう補助損失 | SimplerEnv 42.7（再構成のみ）→71.9%。第三者レビューは対照実験不足を指摘 |
| 学習・階層 | **GPLA**（Manchester, 2026-04, arXiv 2604.05614） | 階層 VLA の高位 VLM が出すサブタスク文と低位行動を、行動条件付き接地モデルでスコアし SimPO で選好学習 | Language Table で教師あり学習と同等の軌道誤差。言語追従率そのものは未報告 |
| 学習・データ | **DAERT の敵対指示を拡張に使う**（§8.1） | 生成した脆弱指示で学習 | 提案のみ |
| 学習・事後学習 | **言い換え LoRA 事後学習**（東京科学大学） | 言い換え文で LoRA | OpenVLA Spatial −38.0→+2.0pt。別シードと held-out 言い換えに転移 |
| 学習・共学習 | **Qwen-RobotManip の dual-stream 共学習** | ロボット:VL=9:1、λ=0.1 で VLM 損失を正則化として残す | RoboTwin-IF 72.2% vs π0.5 49.6%。forgetting の定量（VQA 前後）は未記載 |
| 推論時・活性操作 | **機械論的ステアリング**（Berkeley Häon et al. 2509.00328、Stanford Swann et al. 2603.19183、UPenn COAST 2605.17144） | FFN ニューロン群や SAE 特徴、conceptor を推論時に書き換えて速さ・高さ・成功方向へ誘導 | 「言語追従」の直接改善ではないが、プロンプトに語を足すより活性操作の方が効く（Häon）。言語経路が弱い証拠の裏返し |
| 拒否・確認 | **RationalVLA**（HKUST-GZ ほか, 2025-06, arXiv 2506.10826）、IVA（§7）、ProGAL-VLA（§4.1） | 実行不能・欠陥指示を `<REJ>` で拒否 | RAMA（CALVIN）で平均長 1.30→2.26。ConflictVLA-Bench は「プロンプトで前提確認を促すだけでは効かない」と報告 |

### 8.4 周辺・読めなかったもの
- **PolaRiS**（arXiv 2512.16881）: 実→シミュ評価基盤。言語追従の評価は含まない。
- **Prompt-Driven Exploration**（arXiv 2607.08837）: 言語を RL の探索空間にする提案。PDF が取得できず内容未確認。
- **Beyond Actions / AFCE**（arXiv 2610.13212）、**REKI**（arXiv 2610.22178）、**Look Before Acting**（arXiv 2603.15618）: 言語接地は主題でない。
- **Do VLA Models Mean What They Say?**（Pavone グループ, arXiv 2607.04681）: 運転 VLA の CoT が行動を忠実に反映するかを問う。「推論が良く見えても行動が良くなるとは限らない」。RoboSemanticBench の「CoT 正解でも把持対象誤り」と同じ構図。
- **Gemini Robotics** の言い換え種別ごとの数値、**How-level** 論文の対象モデルと数値、**Robust Skills, Brittle Grounding** の数値、**CofactVLA** のシミュ数値は本文未取得。
- VLN 領域の先行研究（Counterfactual VLN, NeurIPS 2020; Diagnosing VLN, 2021）にも「指示を無視して環境バイアスで動く」報告があるが、本調査の範囲外として URL のみ掲載。

### 8.5 追加分の出典（URL）
- Encoded but Not in Control: https://arxiv.org/abs/2610.06235
- Not All Features Are Created Equal（Action Atlas）: https://arxiv.org/abs/2603.19233
- Paraphrase Consensus Is Not Self-Consistency: https://arxiv.org/abs/2610.19438
- Rephrase Before You Act: https://arxiv.org/abs/2610.10526
- DAERT（Diversity-Aware Red Teaming）: https://arxiv.org/abs/2604.05595
- Robust Skills, Brittle Grounding: https://arxiv.org/abs/2602.24143
- Spa-Bench: https://arxiv.org/abs/2610.12696
- ConflictVLA-Bench: https://arxiv.org/abs/2609.31792
- Do VLAs Really Follow Instructions?（How-level, SIMPLER-IN）: https://arxiv.org/abs/2610.21251
- Qwen-RobotManip 技術報告（RoboTwin-IF）: https://arxiv.org/abs/2606.17846
- Shortcut Learning in Generalist Robot Policies（CoRL 2025）: https://arxiv.org/abs/2508.06426 ／ https://proceedings.mlr.press/v305/xing25a.html
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics 技術報告: https://arxiv.org/abs/2503.20020 ／ Veo 世界シミュレータ評価: https://arxiv.org/abs/2512.10675
- Evaluating π0 in the Wild（Penn PAL Lab）: https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/
- Language2Drive: https://arxiv.org/abs/2610.16634
- ICR-Drive: https://arxiv.org/abs/2604.05378
- VLA-Retain: https://arxiv.org/abs/2610.27698
- VLANeXt: https://arxiv.org/abs/2602.18532
- VLM4VLA: https://arxiv.org/abs/2601.03309
- CoVer（Scaling Verification）: https://arxiv.org/abs/2602.12281 ／ https://scalingintelligence.stanford.edu/pubs/cover_vla/
- CofactVLA: https://arxiv.org/abs/2608.04396
- SALT（Lost in Reconstruction）: https://arxiv.org/abs/2608.10484
- GPLA（Grounding Hierarchical VLA）: https://arxiv.org/abs/2604.05614
- Mechanistic Interpretability for Steering VLA: https://arxiv.org/abs/2509.00328
- Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA: https://arxiv.org/abs/2603.19183
- COAST: https://arxiv.org/abs/2605.17144
- RationalVLA: https://arxiv.org/abs/2506.10826
- Do VLA Models Mean What They Say?: https://arxiv.org/abs/2607.04681
- PolaRiS: https://arxiv.org/abs/2512.16881
- Prompt-Driven Exploration（未読）: https://arxiv.org/abs/2607.08837
- LIBERO-Plus CVPR 2026 版: https://cvpr.thecvf.com/virtual/2026/poster/38735
- Counterfactual VLN（NeurIPS 2020）: https://papers.nips.cc/paper/2020/hash/39016cfe079db1bfb359ca72fcba3fd8-Abstract.html ／ Diagnosing VLN: https://arxiv.org/abs/2103.16561

---

## 9. 範囲拡張: 言語条件付き世界モデル（WM）・World-Action Model（WAM）での指示無視（2026-10-24）

依頼: yuuya（2026-10-24）「言語指示を用いる WM や WAM にも範囲広げて再度言語指示を無視する問題を調査してほしい」

### 9.0 範囲と用語
- **WM（世界モデル）**: 現在の観測＋条件（言語、行動、またはその両方）から将来の映像・潜在状態を予測する生成モデル。本節では言語を条件に取るもの（text/image-to-video 系: Cosmos-Predict、Wan、Veo、Genie Envisioner GE-Base、Qwen-RobotWorld、Kairos、DreamGen 用 fine-tune 版）を主対象にした。
- **WAM（World-Action Model）**: 映像予測モデルを土台にして行動を出す方策。映像と行動を同時生成する型（DreamZero、LingBot-VA、MotuBrain、τ0-WM、AW-0、FlowWAM）、学習時だけ映像を使う型（Fast-WAM、GigaWorld-Policy、Cosmos Policy）、映像生成→逆動力学の型（UniPi、Gen2Act、LVP、Veo-Act、Say-Dream-Act）、潜在予測を補助にする型（VLA-JEPA、WoG）に分かれる。ほぼ全てが言語指示を取る。
- **言語を条件にしない WM** も多く、区別が要る: V-JEPA 2（ゴール画像）、1X World Model・Ctrl-World・World-VLA-Loop（行動条件。1X は「曖昧な指示ではなく正確な軌道で操舵すべき」と明記）、DreamDojo（事前学習でテキストは空プロンプト固定、潜在行動で条件付け）。これらは「言語を無視する」のではなく最初から言語を使わない設計で、**言語の代わりに行動や軌跡で条件付けること自体が、言語条件の不安定さへの一つの回答**になっている【推測】。

### 9.1 結論
1. **WM・WAM でも同型の現象が確認できる。証拠は 3 系統ある。**
   - (a) **生成映像が指示に従わない**: WorldArena（14 モデル、RoboTwin 2.0）では「同じ初期フレームに 3 つの異なる指示を与えたときの映像の差」を測る Action Following が全モデルで 0.14 以下（0〜1 スケール）。Genie Envisioner の Instruction Following 0.203、Cosmos-Predict 2.5（テキスト条件）0.266。DreamGen Bench では汎用動画モデルのゼロショット指示追従率がほぼ 0（Wan2.1 で 0.0、Cosmos で 4.2）で、ロボットデータで微調整して初めて 70〜90 に上がる。RoboGaze の失敗イベント 1,847 件のうち 11% が「指示との不整合」（誤物体 61、誤目標位置 54）。Veo-Act では失敗の 75% が生成映像に由来し、うち「誤った物体を選ぶ」が 8/21。
   - (b) **WAM 方策の言語感度**: Same Goal, Different Words（Cosmos3 Nano/Edge、FLUX 3 Action）は、同じ空間目標を「対象物を主語にする」か「参照物を主語にする」かで配置成功率が 12.2pt 変わり、FLUX のテキストエンコーダは参照物主語の文で関係を全件反転していた。IndustrialVLA-Bench では FastWAM が LIBERO-Para（言い換え）で 51.16%、クリーンから −45.39pt と最下位で「言語接地のボトルネック」と評された。Encoded but Not in Control（§8.1）は FastWAM・LingBot-VA も対象にしており、対象差し替え時の挙動は VLA と同じだった。
   - (c) **行動条件も無視する姉妹現象（action unfaithfulness）**: 行動条件付き WM は専門家外の行動を与えると「もっともらしいが行動と無関係な映像」を出す（Do Robotic World Models Really Follow Actions?、6 モデルで誤差 +0.029〜0.099）。DreamDojo は既存 WM が「反事実の行動に無反応」と指摘。DreamTrue は反事実行動で相互作用欠陥 48%。**条件が弱いと視覚事前分布に崩壊する**という構造は、言語でも行動でも同じ。
2. **ただし、言い換え頑健性だけを見ると WAM は VLA と同等以上**という結果もある（Huawei の比較研究: LIBERO-Plus 言語列で Cosmos-Policy 81.7、GE-Act 77.4、VLA-JEPA 88.1 vs π0.5 85.6; RoboTwin 2.0-Plus では LingBot-VA 87.3 が最高）。§1 で見たとおり「言い換えで落ちない」は「言語を使っていない」とも読めるため、**言語利用の有無は反事実指示（同一場面・別指示）で測る必要がある**。WAM でそれを測った研究は Encoded but Not in Control と WorldArena の Action Following だけで、どちらも「ほぼ場面で決まる」と出ている。
3. **WAM 固有の経路**: 行動が映像を経由して出るので、言語→映像の誤りがそのまま行動誤りになる（DreamZero「失敗の大半は映像生成誤り」）。一方で、言語を使わず行動・軌跡・参照画像で条件付けする WM（Ctrl-World、1X、WorldCanvas、FlowWAM の flow 条件）は、言語無視の問題を回避する代わりに「何をするか」を別の手段で与える必要がある。
4. **対策は VLA と同じ 4 系統に、WM 特有の 2 系統が加わる**: 指示の分解合成（RoboDreamer）、キャプションの密度・設計（Qwen-RobotWorld、WoW-Eval「密な指示で指示理解が一貫して改善」）、VLM 判定器によるフィルタ／報酬（GR00T N1 の neural trajectory フィルタ、WorldModelBench の判定器報酬、DreamTrue）、反事実ペア学習（WorldSync の Intervention-Effect 損失、DreamTrue。**言語版の反事実学習は WM ではまだ見当たらない【推測】**）、**WM を方策の指示追従評価器として使う**（Ctrl-World: 指示追従の失敗を想像内で検出し合成データで 38.7→83.4%）、**言語を予測対象にする**（Dynalang、LWM、LED-WM。言語を条件ではなく「予測する対象」に入れて接地する発想。ロボット映像 WM への適用はまだ【推測】）。

### 9.2 現象・診断の追加（WM・WAM）

| 論文（時期） | 対象 | 何をしたか | 主な数値・知見 |
|---|---|---|---|
| **WorldArena**（清華ほか, 2026-02, arXiv 2602.08971） | Veo 3.1、Wan 2.6/2.2、Ctrl-World、CogVideoX、Cosmos-Predict 2.5（text / action）、IRASim、Vidar、TesserAct、GigaWorld-0、WoW、RoboMaster、Genie Envisioner | Instruction Following（Qwen3-VL-8B が行動種・対象・最終状態を 1〜5 で判定）、Semantic Alignment、**Action Following（同一初期フレーム×3 指示の映像の相違度）** | IF: Veo 3.1 0.933、Wan 2.6 0.854、Ctrl-World 0.727、Cosmos-Predict 2.5 text 0.266、Genie Envisioner 0.203。**Action Following は全モデル ≤0.142**（指示を変えても映像がほぼ同じ。図 11 に「どの指示でも同じ動作」の例）。EWMScore と行動計画性能の相関は r=0.36。※ v2 では 0〜100 スケールで Veo 99.64、Cosmos text 98.22 と大きく異なる値が出ており、版で判定基準が変わっている |
| **DreamGen / DreamGen Bench**（NVIDIA, 2025-05, arXiv 2505.12705） | Wan2.1、Hunyuan、CogVideoX、Cosmos（ゼロショットと SFT） | IF を Qwen-VL-2.5 / GPT-4o の 0/1 判定、PA を VideoCon-Physics で。人手判定との相関 r≈0.92〜1.00 | **ゼロショットの IF はほぼ 0**（Wan 0.0、Cosmos 4.2）。SFT 後は Cosmos 79.2（RoboCasa）、90.0（GR1-Object）。IF+PA と下流方策性能に正の相関（図 6）。GR00T N1 論文では生成映像を MLLM で判定し「指示に正確に従っていないものを除外」して再キャプションしている |
| **Same Goal, Different Words**（Northeastern・GM, 2026-10, arXiv 2610.16242） | Cosmos3 Nano / Edge、FLUX 3 Action（DROID 版 WAM） | 同じ初期状態で空間目標の言い回しだけ変える（Direct / Target-first / Reference-first）。864 エピソード | TF は RF より +12.2pt（p<0.001）。Mustard-right は TF 20/24 vs RF 2/24。FLUX のテキストエンコーダは RF の誤答 6 件全てで関係を反転。Nano は文のみなら全問正解だが画像を加えると RF 正答率 89.9% に低下。「成功」判定の一部は**間違った物体を動かして関係だけ満たした**もの。安定配置率は途中 43.2% → 終了時 21.8% |
| **IndustrialVLA-Bench**（Griffith ほか, 2026-09, arXiv 2609.25562） | π0.5、GR00T N1.7、UnifoLM-VLA-0、Xiaomi-Robotics-0、**FastWAM、Cosmos Policy** | LIBERO / LIBERO-Plus / LIBERO-Para の 3 軸＋展開性 | LIBERO-Para: FastWAM 51.16%（クリーンから −45.39pt、最下位）、GR00T N1.7 74.26%、π0.5 71.33%。LIBERO-Plus 言語列と LIBERO-Para の順位相関は ρ=0.60 で、別の性質を測っていると結論 |
| **Do WAMs Generalize Better than VLAs?**（Huawei, 2026-03, arXiv 2603.22078） | GE-Act、Cosmos-Policy、LingBot-VA、MOTUS、VLA-JEPA vs π0.5 ほか | LIBERO-Plus と自作 RoboTwin 2.0-Plus の 7 軸 | 言語列: Cosmos-Policy 81.7、GE-Act 77.4、VLA-JEPA 88.1、π0.5 85.6（LIBERO-Plus）。RoboTwin: LingBot-VA 87.3 が最高。WAM の強みは光・ノイズ・配置で、カメラ・初期姿勢には弱い。推論は π0.5 の 4.8〜83 倍遅い |
| **Encoded but Not in Control**（§8.1 再掲） | π0.5、GR00T N1.7、**FastWAM、LingBot-VA** | 場面固定で対象差し替え | WAM 2 種も VLA と同じく元の対象へ向かう（LIBERO-Object で 86.5〜98.3%） |
| **RoboGaze**（VinUni・DFKI ほか, 2026, arXiv 2606.28385） | Cosmos-Predict2.5、DreamGen、DreamDojo、Wan 2.2、Veo の生成 914 クリップ | 6 次元 30 種の失敗を VLM で構造化報告 | 失敗 1,847 件中 **Instruction Consistency 198 件（11%）**: 誤物体 61、誤目標位置 54、誤効果器、順序誤り、制約無視。この次元はアノテータ間一致が最高（κ=0.81） |
| **Veo-Act**（清華, 2026-04, arXiv 2604.04502） | Veo-3.1-fast → IDM → π0.5 引き継ぎ | 生成映像を高位計画に使う | 失敗 28 件中 21 件（75%）が映像由来。うち幻覚物体 8、**誤った物体選択 8**、手の変形 7。方策側の指示追従は 1/30→15/30（pass-by）、14/30→28/30（類似物体） |
| **DreamZero / World Action Models are Zero-shot Policies**（NVIDIA, 2026-02, arXiv 2602.15922） | 14B WAM（Wan2.1 ベース、凍結テキストエンコーダ） | 未見タスクの進捗率 | 未見タスク 39.5% vs 事前学習 VLA 16.3%。「事前学習 VLA は指示に関係なく物体へ手を伸ばして掴もうとする」と VLA 側の指示無視を観察。**DreamZero の失敗の大半は映像生成誤り**で、方策は映像に従うだけ |
| **Dream.exe**（NUS Show Lab ほか, 2026-06, arXiv 2606.04811） | SeedDance 2.0、Kling 3.0、Wan 2.7/2.2、Hailuo、Veo 3.1、LTX、Cosmos Policy | 生成映像を軌道に変換してシミュで実行。task adherence も VLM で採点 | 実行成功率は最高でも 16.5%（SeedDance）。**物理的もっともらしさと成功率は無相関（r=−0.03）**。指示は Gemini で書き換えると改善 |
| **Do Robotic World Models Really Follow Actions?**（北京大ほか, 2026-08, arXiv 2608.24885） | Ctrl-World、Cosmos-Predict2.5、Cosmos3、DreamDojo、Motus、LingBot-VA | WorldEcho: 専門家外の行動で映像がどう変わるか | 6 モデルとも専門家外では誤差 +0.029〜0.099。「もっともらしいが行動を無視した映像」か「腕が崩れる」のどちらか。言語は固定で解析なし |
| **DreamTrue**（CASIA・Amap, 2026-10, arXiv 2610.12468） | Wan2.1-VACE-14B ベース | 反事実行動（終端姿勢の SE(3) 摂動）での欠陥率 | 反事実で相互作用欠陥 48.12%、物体欠陥 31.88%（Stage I）。原因はキャリブレーション誤差と「成功デモばかりで失敗の継続を学んでいない」こと |
| **DreamDojo**（NVIDIA, 2026-02, arXiv 2602.06949） | 人間動画から学ぶ汎用 WM | 反事実行動評価セット | 既存 WM は「反事実行動に無反応」と指摘。**テキスト条件は空プロンプト固定**で言語を使わない設計 |
| **Evaluating Gemini Robotics in a Veo World Simulator**（§8.1 再掲） | Veo ベースの評価器 | 指示の言い換え・タイポ・他言語・具体度 | Object 軸の失敗の多くが「指示追従の誤り（より馴染みのある物体へ向かう）」。数値なし |
| **Say, Dream, and Act**（復旦, 2026-02, arXiv 2602.10717） | Cosmos-Predict2 を蒸留した WM＋ACT 型方策 | 色で指定したブロックを拾うゼロショット参照テスト | 参照成功率 Cosmos-2B 96 / 14B 90 / Wan-14B 54。指示成功率 86 / 82 / 44。反事実指示の直接テストはなし |
| **WoW-World-Eval / Turing test**（北京大ほか, 2026-01, arXiv 2601.04137） | Hailuo、Kling、WoW-cosmos2、WoW-wan | 22 指標。指示意味整合を含む | 指示理解と人の選好の相関 r=0.75。**簡潔な指示を密な記述に展開すると指示理解が一貫して改善** |
| **Large Video Planner**（MIT ほか, 2025-12, arXiv 2512.15840） | Wan 2.1 ベース 14B 映像プランナー | 第三者の 100 タスクで 4 段階採点 | L3「タスク完了」59.3%（Wan 2.1 素は 39.3%、Cosmos-Predict 2 は 7.5%）。指示は Gemini で整形してから入力。言い換え比較はなし |

### 9.3 原因分析（WM・WAM 固有の観点）
- **視覚事前分布の強さが VLA より大きい**: 映像生成モデルは Web 動画で「場面から最ももっともらしい継続」を学んでおり、テキスト条件はクロスアテンションで後付けされる（Robotic Video World Models サーベイ §2.2.4）。ロボットデータで微調整すると、§2.1 と同じ「1 場面 1 タスク」構造のため、指示を変えても映像が変わらない（WorldArena の Action Following、DreamGen ゼロショット IF≈0）。
- **言語が行動に届く経路が長い**: WAM では 言語 → 映像（または潜在）→ 逆動力学 → 行動。言語の誤解や無視は映像段階で起き（RoboGaze の誤物体・誤位置）、行動段階では修正されない（DreamZero、Veo-Act）。Video Policy（Columbia）や Say-Dream-Act のように行動ヘッドがテキストを直接受け取らない設計では、言語の影響は映像経由のみ。
- **テキストエンコーダ自体の弱さ**: 凍結 T5 / UMT5 / CLIP エンコーダ（GE-Base、FlowWAM、Video Policy、DreamZero）は空間関係の語順に弱く、FLUX のエンコーダは参照物主語の関係を反転した（Same Goal, Different Words）。Qwen-RobotWorld は「T5/CLIP ではなく MLLM で複合指示を解析する」ことを設計理由に挙げる。
- **条件付け一般の問題**: 行動条件でも同じ崩壊が起きる（WorldEcho、DreamTrue）。原因として挙げられているのは、成功デモのみの学習（失敗の継続を知らない）、キャリブレーション誤差、専門家外行動の未被覆。これは「言語の多様性不足」の行動版。
- **評価指標の罠**: 映像品質・物理もっともらしさと指示追従・実行可能性は相関しない（WorldArena r=0.36、Dream.exe r=−0.03）。CLIP-T のような自動指標は細かい動作の違いを捉えない（WorldCanvas の自己申告）。「成功」が誤った物体を動かして達成される（Same Goal, Different Words）。

### 9.4 対策（WM・WAM）

| 系統 | 手法 | 要点 | 効果 |
|---|---|---|---|
| 指示の分解・合成 | **RoboDreamer**（HKUST・MIT, ICML 2024, arXiv 2404.12377） | 指示を動詞句と空間関係句に構文解析し、成分ごとの拡散モデルを product of experts で合成 | 未見の組合せで人手評価 68.8→81.3。「pepsi を bottle の近くへ」で誤った物体の近くに置く失敗を動機に挙げる |
| | **ManipDreamer**（北京大, 2025-04, arXiv 2504.16464） | 動詞・前置詞の action tree＋深度・SAM の視覚ガイダンス | RLBench 0.480→0.503。多段タスクで改善 |
| キャプション設計・共学習 | **Qwen-RobotWorld**（Alibaba, 2026-06, arXiv 2606.17030） | 凍結 Qwen2.5-VL でテキストを符号化。5 層構造のキャプション（目標・動作詳細・物理フィードバック）を長短 2 形式で 50% ずつ混ぜ、LLM＋人手でキャプションをフィルタ。具身 70%・一般 30% の共学習 | DreamGen Bench GR1-Object IF 0.878（最高）、WorldModelBench 指示スコア 2.33（Veo3 2.52、Wan2.6 2.50、Cosmos 2.14）。キーワード 1 語の差し替えで動作が変わる定性例 |
| | **Kairos**（2026-07, arXiv 2606.16533）、**MotuBrain**（arXiv 2604.27792） | 専用テキストストリーム（MotuBrain は「専用ストリームが指示追従を改善」と主張） | MotuBrain の WorldArena IF 0.8072（Veo3.1 0.9714、ABot-PW 0.9210 より低い） |
| 判定器によるフィルタ・報酬 | **GR00T N1 の neural trajectory フィルタ**（arXiv 2503.14734） | 生成映像を MLLM で判定し指示不一致を除外、再キャプション | 共学習で RoboCasa +4.2〜8.8pt |
| | **WorldModelBench**（NeurIPS 2025 D&B, arXiv 2502.20694） | 人手 67K ラベルで 2B 判定器を学習（GPT-4o より +8.6pt）。判定器報酬で生成モデルを微調整 | 「世界モデル能力が顕著に改善」（数値は本文未確認） |
| | **DreamTrue**（反事実事後学習） | 反事実行動を生成し、欠陥を採点する Qwen3.5-9B 報酬モデルで RL | 相互作用欠陥 48.12→6.25%、物体欠陥 31.88→3.12% |
| | **EVA**（CUHK-SZ, 2026-03, arXiv 2603.17808） | IDM を報酬にして実行可能性で RL | 完全実行 68.1→83.8%。指示追従は 90.5→89.5% で「維持」。指示追従を上げる報酬ではない |
| 反事実ペア学習 | **WorldSync**（arXiv 2608.24885） | 同じ観測・指示・ノイズで行動だけ違う対を作り、速度場の差を将来潜在の差に合わせる Intervention-Effect 損失 | 専門家外行動での誤差改善（数値は本文未取得）。言語を変える対は未実施 |
| 言語以外の条件で補う | **WorldCanvas**（HKUST・Ant, 2025-12, arXiv 2512.16924） | 「テキストだけでは出来事の時空間を指定できない」として軌跡・参照画像・テキストの 3 条件 | 人手のプロンプト追従投票 73.67% vs Wan2.2 11.00%。CLIP-T では差がほぼ出ない |
| | **Ctrl-World**（Stanford・清華, ICLR 2026, arXiv 2510.10125）、**1X World Model**、**World-VLA-Loop** | 行動条件のみで WM を作り、言語は方策側に残す | Ctrl-World は方策の指示追従失敗を想像内で検出し、合成データで π0.5-DROID を 38.7→83.4%。「WM は成功率を過小評価しがち」 |
| | **FlowWAM**（CASIA ほか, 2026-07, arXiv 2607.13017）、**DreamDojo**（潜在行動）、**V-JEPA 2**（ゴール画像） | 光学フロー・潜在行動・ゴール画像で条件付け | FlowWAM は WorldArena v2 で IF 94.56（Veo 3.1 99.64 より低い）、Semantic Alignment 89.93（最高） |
| 言語を予測対象にする | **Dynalang**（Berkeley, ICML 2024, arXiv 2308.01399）、**LWM**（Princeton, arXiv 2402.01695）、**LED-WM**（Oregon State, NeurIPS 2025 WS, arXiv 2511.22904） | 言語を条件ではなく世界モデルの予測対象（将来のテキスト・観測）に入れて接地。LED-WM は実体と文を attention で整合 | HomeGrid で言語条件付き R2D2 は言語を増やすと悪化、Dynalang は改善。MESSENGER では接地機構のない Dynalang が未見ゲームで 0.03、LED-WM 34.97。**ロボット映像 WM にはまだ適用例が見当たらない【推測】** |
| 階層化・プログラム | **AW-0: Agentic World Action Models**（Berkeley・NVIDIA, 2026-10, arXiv 2610.30136） | コード（スキルプログラム）・映像・行動の 3 エキスパートを同時デノイズ。指示は 15% のバッチでマスク | 物体選択・個数指定を要する実機タスクで 91.7%（π0.5 70.8%）。注釈付きプログラムは言語サブゴールのみより 8 タスク中 7 で優位 |
| 推論時の指示整形 | **LVP**、**Dream.exe** | 指示を Gemini で映像モデル向けに書き換えてから入力 | §8.3 の Rephrase Before You Act と同じ発想。WM 側では標準的な前処理になりつつある |

### 9.5 VLA との対応表

| 観点 | VLA（§1〜§8） | WM・WAM（§9） |
|---|---|---|
| 症状 | 指示を変えても行動が変わらない。学習タスクを実行 | 指示を変えても生成映像が変わらない（Action Following ≤0.14）。誤物体・誤位置の映像を生成し、方策がそれに従う |
| 直接の診断 | 空指示・反事実指示・モダリティ除去 | DreamGen Bench IF、WorldArena IF/Action Following、RoboGaze、言い回し感度（Same Goal, Different Words） |
| 主因 | 1 場面 1 タスク、言語の多様性不足、トークン不均衡、勾配干渉 | 上記に加え、Web 動画由来の強い視覚事前分布、凍結テキストエンコーダの弱さ、言語→映像→行動の長い経路 |
| 姉妹現象 | 視覚ショートカット（視点・背景） | 行動条件の無視（action unfaithfulness） |
| 効く対策 | 同一場面・複数指示データ、SAM、反事実学習、推論時ガイダンス | 密なキャプション＋VL 共学習、VLM 判定フィルタ／報酬、指示分解合成、言語以外の条件（行動・軌跡・参照画像）、WM で方策の指示追従を評価 |
| 未解決 | 多タスク容量、物体参照の接地 | 言語版の反事実学習、映像品質と指示追従の乖離、言い回し感度の因果 |

### 9.6 周辺・読めなかったもの
- **言語条件の評価がない、または言語を使わない WAM/WM**: LingBot-VA（arXiv 2601.21998。言語条件 c は式にあるが評価なし）、Fast-WAM（2603.16666）、GigaWorld-Policy（2603.17240。クロスアテンションで言語を入れるが言語評価なし）、Cosmos Policy（2601.16163）、τ0-WM（2606.01027）、World Guidance（ICML 2026）、WLA（2606.05979）、UVA（CLIP 1 トークンを繰り返し付加、言語除去実験なし）、Video Policy（2508.00795。行動ヘッドはテキストを受け取らない）、Gen2Act（2409.16283。失敗分析節は未取得）、Pandora（2406.09455。自由テキスト行動の追従評価は要旨になし）。
- **Cosmos-Predict2.5**（2511.00062）は「instruction alignment の大幅改善」を主張するが、数値は要旨に無く本文未取得。**Cosmos 3**（2606.02800）はテキスト整合の人手評価項目を持たない。
- **Kairos** の DreamGen Bench / WorldModelBench の数値、**Ctrl-World** 図 7（実機 vs 想像内の指示追従率）、**WorldModelBench** の判定器報酬による改善幅、**EWMBench**（arXiv 2505.09694）のモデル別意味整合は本文未取得。
- **WorldArena** は v1（0〜1）と v2（0〜100）で IF の値が大きく違う（Cosmos text が 0.266 → 98.22）。採点プロンプトか判定器が変わった可能性が高く、引用時は版を明記する必要がある。
- 位置論文: Do World Models Make Better Robots?（2609.29669。160 ベンチの分類。「反事実能力はほぼ測られていない」）、Robots Need More Than VLAs & World Models（2606.06556）、Autonomous Video Generation with Counterfactual Controllability（2606.24152。「プロンプト追従は観測の忠実度であり、行動可能性ではない」）。

### 9.7 追加分の出典（URL）
- WorldArena: https://arxiv.org/abs/2602.08971
- DreamGen / DreamGen Bench: https://arxiv.org/abs/2505.12705 ／ https://research.nvidia.com/labs/gear/dreamgen
- Same Goal, Different Words: https://arxiv.org/abs/2610.16242
- IndustrialVLA-Bench: https://arxiv.org/abs/2609.25562
- Do World Action Models Generalize Better than VLAs?: https://arxiv.org/abs/2603.22078
- RoboGaze: https://arxiv.org/abs/2606.28385
- Veo-Act: https://arxiv.org/abs/2604.04502
- DreamZero（World Action Models are Zero-shot Policies）: https://arxiv.org/abs/2602.15922
- Dream.exe: https://arxiv.org/abs/2606.04811
- Do Robotic World Models Really Follow Actions?（WorldEcho / WorldSync）: https://arxiv.org/abs/2608.24885
- DreamTrue: https://arxiv.org/abs/2610.12468
- DreamDojo: https://arxiv.org/abs/2602.06949
- Say, Dream, and Act: https://arxiv.org/abs/2602.10717
- WoW-World-Eval（Wow, wo, val!）: https://arxiv.org/abs/2601.04137
- Large Video Planner: https://arxiv.org/abs/2512.15840
- RoboDreamer（ICML 2024）: https://arxiv.org/abs/2404.12377
- ManipDreamer: https://arxiv.org/abs/2504.16464
- Qwen-RobotWorld 技術報告: https://arxiv.org/abs/2606.17030
- Kairos: https://arxiv.org/abs/2606.16533
- MotuBrain: https://arxiv.org/abs/2604.27792
- GR00T N1（neural trajectory フィルタ）: https://arxiv.org/abs/2503.14734
- WorldModelBench（NeurIPS 2025 D&B）: https://arxiv.org/abs/2502.20694
- EVA（Inverse Dynamics Rewards）: https://arxiv.org/abs/2603.17808
- WorldCanvas: https://arxiv.org/abs/2512.16924
- Ctrl-World（ICLR 2026）: https://arxiv.org/abs/2510.10125 ／ https://sites.google.com/view/ctrl-world
- 1X World Model: https://www.1x.tech/1x-world-model.pdf
- World-VLA-Loop: https://arxiv.org/abs/2602.06508
- FlowWAM: https://arxiv.org/abs/2607.13017
- Dynalang（ICML 2024）: https://arxiv.org/abs/2308.01399
- Language-Guided World Models: https://arxiv.org/abs/2402.01695
- LED-WM: https://arxiv.org/abs/2511.22904
- AW-0（Agentic World Action Models）: https://arxiv.org/abs/2610.30136
- Genie Envisioner（ICLR 2026）: https://arxiv.org/abs/2508.05635 ／ EWMBench: https://arxiv.org/abs/2505.09694
- LingBot-VA（Causal World Modeling for Robot Control）: https://arxiv.org/abs/2601.21998
- Fast-WAM: https://arxiv.org/abs/2603.16666 ／ Faster-WAM: https://arxiv.org/abs/2608.02365
- GigaWorld-Policy: https://arxiv.org/abs/2603.17240
- Cosmos Policy: https://arxiv.org/abs/2601.16163 ／ Cosmos-Predict2.5: https://arxiv.org/abs/2511.00062 ／ Cosmos 3: https://arxiv.org/abs/2606.02800
- τ0-WM: https://arxiv.org/abs/2606.01027
- World-Language-Action Model: https://arxiv.org/abs/2606.05979
- World Guidance（ICML 2026）: https://proceedings.mlr.press/v306/su26c.html
- Unified Video Action Model: https://unified-video-action-model.github.io/static/UVA_paper.pdf
- Video Generators are Robot Policies: https://arxiv.org/abs/2508.00795
- Gen2Act: https://arxiv.org/abs/2409.16283
- UniPi: https://arxiv.org/abs/2302.00111
- Pandora: https://arxiv.org/abs/2406.09455
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- サーベイ: Robotic Video World Models: https://arxiv.org/abs/2601.07823 ／ World Model for Robot Learning: https://arxiv.org/abs/2605.00080 ／ Do World Models Make Better Robots?: https://arxiv.org/abs/2609.29669
- 位置論文: Robots Need More Than VLAs & World Models: https://arxiv.org/abs/2606.06556 ／ Counterfactual Controllability: https://arxiv.org/abs/2606.24152
- Awesome-WAM（論文リスト）: https://github.com/OpenMOSS/Awesome-WAM
