# VLA の言語指示無視・場面依存・言語の誤用の研究に新規性は残っているか（2026-10-25）

- 依頼: yuuya（2026-10-25）「この言語指示を無視する・場面や観測画像から動作する・言語指示を正しく使えていないなどの問題の研究に新規性がまだあるか調査してほしい」
- 方法: 調査メモ `research/vla_language_ignoring_survey_2026-10-17.md`（§6・§8.0・§9.1 の未解決点）を起点に、未解決点ごとに「埋めた論文があるか」を Web 検索（arXiv、ICLR/CVPR/ICML/IJCAI/ECCV 2026、プロジェクトページ）で確認した。調査メモに未収録の論文を 22 本見つけ、§5 に載せた。
- 判定の基準: 「飽和」= 同じ問いに独立した 4 本以上があり、新しい 1 本の差分を示しにくい。「空白」= 未解決だと複数の論文が明記しているか、検索しても該当が 0〜1 本。
- 【推測】は論文に明記がなく私が補った判断。

---

## 0. 結論

1. **新規性は残っている。ただし場所が移った。** 「VLA が言語を無視する」という現象の報告、LIBERO 派生の診断ベンチ、CFG 型の推論時ガイダンス、言い換えデータ拡張、ベイズ分解の定式化、「符号化されているが使われない」の相関的な機械論解析は、いずれも 2025-10 以降の 12 か月で 4〜20 本が出て飽和している。ここに 1 本足しても差分を示しにくい。
2. **残っている空白は 6 つ。** (1) 原因の因果的特定（介入で経路を変えると言語追従が変わることを示す）、(2) 対策の一般性（同一条件で複数対策 × 複数モデル）、(3) 学習ダイナミクス（微調整のいつ・どこで言語依存が崩れるか）、(4) RL 事後学習と言語追従の関係、(5) 展開時に「指示を無視している」ことを検出する仕組み、(6) 物体参照の接地を言語経路で直す方法。いずれも検索で該当が 0〜1 本だった。
3. **競争は速い。** 2026-09-30〜10-22 の 3 週間だけで、この問題を主題にした論文が 13 本出ている（§3）。否定指示（NegaAlign）、How-level 指示、ナビゲーションでの指示—行動 binding（NavBind）、因果的 patching（Causal Analysis）は、この 1 か月で「初の 1 本」が出た。空白は数か月単位で埋まるので、**新規性を守るには「厳密な評価設計（同一場面・別指示、接地率、held-out 言い換え）」「因果介入」「複数モデル」を組み合わせる**必要がある。単独のベンチや単独の手法では守れない。

---

## 1. 飽和している領域（ここで新規性を主張するのは難しい）

| 領域 | 本数（調査メモ＋今回） | 代表 | 新しく書くなら最低限必要なこと |
|---|---|---|---|
| 現象の報告（空指示・言い換え・反事実指示で LIBERO 上の VLA が崩れる） | 約 20 | LIBERO-Plus、LIBERO-PRO、LIBERO-CF、LangGap、ICBench、VLA-Arena、RADAR | LIBERO 外の実機、または新しい指示クラス（否定・How-level など）でないと差分がない |
| LIBERO 派生の診断ベンチ | 約 12 | LIBERO-Plus/PRO/CF/Para、LangGap、ICBench、VLA-Arena、ConflictVLA-Bench、RoboTwin-IF、NegaBench、SIMPLER-IN | ベンチ単体は弱い。指標（language share、LGS、接地率、PRIDE、nSG）も乱立しており、標準化の提案も競合が多い |
| CFG 型の推論時ガイダンス | 6 | CAG、RAS、IGAR、TAG、ACG、CFG 比較（Flatness 論文内） | 伸びが 10〜20pt で頭打ち、π0.5 で効かない報告が 2 本。CRAFT 論文でも CAG-VA は「モデル・ベンチで効果が大きくばらつく」 |
| 推論時の言い換え・検証・プロンプト最適化 | 5 | CoVer、Rephrase Before You Act、LCC（否定的結果）、Learning What to Say to Your VLA（CMU、2606.12299）、VAP | 凍結モデル向けの外付けは出揃った。CMU の論文は conformal 校正まで入れている |
| 言い換えデータ拡張・LLM リラベル | 7 | DIAL、MCSI、VLA-IT、CAST、合成拡張、言い換え LoRA、Cosmos3 Greek（7 言い回しで罰則半減） | 「言い換えを増やすと効く」は確定。残るのは「何本必要か」のスケーリング（§2-G） |
| 反事実ペア学習 | 3 | ECT（2609.39971）、CRAFT（2610.00524）、CofactVLA | 2026-08〜10 に 3 本。CRAFT は π0・π0.5・GR00T N1.7 で未実演の組合せ 1〜8% → 32〜84%。物体参照版・How-level 版はまだ（§2-F） |
| ベイズ分解の定式化 | 4 | BayesianVLA、BayesVLA、APT、CAG | 式は同じ。理論の深掘り（§2-I）は別 |
| 「言語は符号化されているが制御に使われない」の相関的解析 | 4 | Encoded but Not in Control、Not All Features、VLA-Trace（2605.30117）、Wulff & Cangelosi | プローブ・CKA・attention knockout は出揃った。因果介入（§2-A）が次 |
| 多言語指示 | 3 | 2606.11906（LIBERO 10 言語、30〜50pt 低下、ステップごとの言語感度）、2606.15714（PCA 整合）、2609.07470（Cosmos3 に Greek） | 2026-06〜09 に集中。ステップごとの言語感度という視点は他にも使える |
| WM・WAM の指示追従ベンチ | 5 | WorldArena、DreamGen Bench、WorldModelBench、EWMBench、RoboGaze | 指標はある。対策（§2-H）がない |

---

## 2. 残っている空白（新規性が高い順）

各項目に「何が未解決か」「根拠（誰が未解決と言っているか、何を検索して見つからなかったか）」「研究の形」「現在の計算環境（GR00T N1.7 と π0.5、LIBERO 系ハーネス）での実現性」を書いた。

### A. 原因の因果的特定: 「この経路を変えると言語追従が変わる」を介入で示す
- **未解決の中身**: 「言語は VLM 側に符号化されているが action expert に届かない」は相関（プローブ、attention 寄与、CKA）でしか示されていない。Encoded but Not in Control は「相関的解析であり因果は示していない」と明記。Not All Features は層 0 の活性注入で行動を復元したが、微調整でどの経路が壊れるかは追っていない。
- **直近で部分的に埋めた論文**: **Causal Analysis of Robot Foundation Models**（Swann、Geiger、Fel ほか、2026-10-14、arXiv 2610.20750）。π0.5・MolmoAct2・MolmoBot に 77 万回の activation patching を行い、言語→行動の情報が中〜後期の 1〜3 層の cross-attention（action-conditioning interface）を通ること、指示の名詞が「どの物体か」、後方のテキスト位置が「どこか」を運ぶこと、π0.5 では双方向 prefix attention で画像トークンが指示情報を吸収するが MolmoAct2 では吸収しないことを示した。**ただし目的は推論効率化で、「なぜ無視するか」「微調整で経路がどう変わるか」「データ側対策とモデル側対策が経路に与える差」は扱っていない。** 対象も 13 物体対・初期フレームのみ。
- **研究の形**: Causal Analysis の patching を、(a) 微調整前後のチェックポイント、(b) LangGap 型の反事実データで学習したモデル、(c) SAM・KI で学習したモデル、に適用し、「言語成分が interface で落ちるのか、VLM 内で既に落ちるのか」「対策はどの層の因果効果を回復させるのか」を同じ指標で示す。GR00T N1.7（cross-attention が 4 層ごと、言語感度が早期層に集中）と π0.5（分散）の違い（Wulff & Cangelosi）を因果で確認すれば、§6-7 の「アーキテクチャ依存」に答えが出る。
- **実現性**: 高。学習は不要または小規模。patching の実装は Causal Analysis と Not All Features（Action Atlas）が公開している前提【未確認】。
- **リスク**: Swann らのグループ（Stanford）が次の論文で同じことをする可能性が高い【推測】。急ぐ価値がある。

### B. 対策の一般性: 同一条件で複数対策 × 複数モデル × 複数指標
- **未解決の中身**: §2.2（読書ガイド）のとおり、SAM は π0.5 のみ、IGAR は π0 で効き π0.5 で効かず、RAS も π0.5 で弱い。CRAFT 論文の Table 1（Full FT / LoRA / Frozen / CAG-VA / Skill Composition × π0・π0.5・GR00T N1.7）が最大の横断比較で、そこでも「CAG-VA の効果はモデルとベンチで大きくばらつく」「最も難しい組合せはモデルごとに違う」と書かれている。
- **検索結果**: 「対策の比較研究」「SAM の他モデル再現」「which method」で該当なし。
- **研究の形**: データ側（言い換え LoRA、反事実ペア ECT/CRAFT）、学習側（SAM、VLM 凍結＋LayerNorm = GR00T N1.5 の処方、KI）、推論側（CAG）を、GR00T N1.7 と π0.5 で、LIBERO-CF（接地率・biased 率）、LangGap（多タスク）、LIBERO-Para（action 表現 vs 物体参照）の 3 指標で統一評価する。VQA 保持（VLM2VLA の指標）も併記すると「言語能力」と「言語利用」を分けられる。
- **実現性**: 高〜中。微調整が複数走るが、LIBERO なら 6 時間枠に収まる構成が多い【推測】。
- **新規性の守り方**: 工学的と見られやすいので、A の因果指標と組み合わせて「なぜその対策がそのモデルで効く／効かないか」まで言う。

### C. 学習ダイナミクス: 微調整のいつ・どこで言語依存が崩れるか
- **未解決の中身**: Encoded but Not in Control は微調整前後の 2 点しか見ていない。GDM の Hi-VLA 研究（2606.10267）は「π0.5 を 1 つの指示で狭く微調整すると言い換えへの応答性が落ち、階層制御の成功率が 73.3 → 32.0%」と観察したが、過程は追っていない。VLA-Retain は摂動軸間の干渉を示したが時間軸はない。Qwen-RobotManip は「ベンチ特化微調整で VLA が VA に退化する」と命名しただけ。
- **検索結果**: 「training dynamics」「during fine-tuning」「language reliance emerges/collapses」で該当なし。
- **研究の形**: 微調整ステップ・データ量・タスク数・LoRA ランク・学習率の関数として、language share（Paraphrase Consensus の定義）、LGS（ICBench）、接地率（LIBERO-CF）、sharpness（Flatness の指標）を同時に追跡する。「崩壊はいつ起きるか」「早期停止・学習率・凍結範囲で防げるか」「SAM が効く理由は崩壊の時刻を遅らせるからか」に答える。
- **実現性**: 高。チェックポイントを細かく保存して評価するだけ。
- **新規性**: 「VLA-to-VA degradation の時間分解」は誰も出していない。A・B と同じ基盤で回せる。

### D. RL 事後学習と言語追従
- **未解決の中身**: SimpleVLA-RL（ICLR 2026）、πRL、RLinf、VLA-RL、FlowPRO、Reinforcing VLAs in Task-Agnostic World Models など RL 事後学習が 2025-09 以降に急増しているが、**言語追従への影響を測った論文が見つからない**。成功率だけを報酬にすると「場面から解く」方策が強化され、VA 化を加速する可能性がある【推測】。逆に、同一場面で指示を変えながら RL すれば反事実データを集めずに言語依存を回復できる可能性もある【推測】。
- **検索結果**: 「SimpleVLA-RL / πRL / RLinf × language / paraphrase / instruction following」で該当なし。
- **研究の形**: 公開 RL 基盤（SimpleVLA-RL、RLinf）で事後学習した π0.5 / OpenVLA-OFT を LIBERO-CF・LangGap で評価し、RL 前後の language share を比較する。次に、反事実指示を含むタスク分布で RL し、言語依存が回復するかを見る。
- **実現性**: 中。RL 基盤の導入と GPU 時間が要る。
- **新規性**: 高い。RL 系の読者とこの問題の読者の両方に届く。

### E. 展開時に「指示を無視している」ことを検出する
- **未解決の中身**: 失敗検出（SAFECAST 2608.04246、INSIGHT 2510.01389、ReconVLA、FPC-VLA、RoboFAC）はあるが、**「誤った物体へ向かっている」「指示ではなく場面で動いている」こと自体を検出した研究はない**。SAFECAST は言い換え contrast set で検出器を学習するが、誤物体の検出は評価していない。CoVer は検証器で行動を選ぶが検出が目的ではない。Ask-to-Clarify、Adaptive Question Generation（2610.23874）は曖昧な指示に聞き返すが、方策側の無視は扱わない。
- **研究の形**: CAG の「言語あり／なしの行動分布の差」（ω 項の大きさ）や、Causal Analysis の interface 層の言語成分の大きさを、推論時の monitoring 信号にする。「言語を落としても行動が変わらない」瞬間に実行を止めて確認を求める。language share のオンライン推定。
- **実現性**: 高。学習不要で、既存モデルに後付けできる。
- **新規性**: 高い。安全性・HRI の文脈にもつながる。

### F. 物体参照の接地を言語経路で直す
- **未解決の中身**: LIBERO-Para で物体参照の言い換えが最大の落ち（20〜51pt）。RoboSemanticBench では CoT が正解でも把持対象を誤る。VLAs Are Confined（IJCAI 2026、2505.03500）は「spatial overfitting: 物体名がデモ中の位置に結びつく」と命名。既存の対策は視覚プロンプト（Point-VLA、VP-VLA、Interleave-VLA、VAP）で**言語経路を迂回**している。言語経路で直したものはない。
- **研究の形**: ECT / CRAFT の物体参照版。「同じ物体を別の位置に」「同じ位置に別の物体を」の反事実ペアで、名前と位置の結合を切る。LIBERO は物体配置をプログラムで変えられるので反事実データを作れる。Causal Analysis の「名詞が物体同一性、後方テキストが位置を運ぶ」という知見を使い、名詞表現への介入で確認する。
- **実現性**: 中。データ生成の実装が要る。
- **新規性**: 高い。「最難関」と複数論文が書いている箇所。

### G. 指示多様性のスケーリング則
- **未解決の中身**: 「言い換えを増やすと効く」は確定だが、「場面あたり何本の指示が要るか」「言い換えの種類（action 表現 / 物体参照 / 語順）ごとの必要量」「タスク数が増えたときの容量限界（LangGap の 28%）」は測られていない。Cosmos3 Greek 論文は「7 言い回しで罰則が半減」と 1 点だけ報告。
- **研究の形**: 場面あたり指示数 k と言い換え種類を振って、language share と成功率の曲線を引く。
- **実現性**: 高。B・C の副産物にできる。
- **新規性**: 中。単体だと「データ論文」と見られる。

### H. WM・WAM の言語版反事実学習
- **未解決の中身**: 行動条件の無視（action unfaithfulness）には WorldSync・DreamTrue の反事実事後学習があるが、言語条件の無視（WorldArena の Action Following ≤0.14）に対する学習法はない（調査メモ §9.1-4）。
- **実現性**: 低〜中。映像生成モデルの学習コストが大きい。
- **新規性**: 高いが、環境に合わない。

### I. 理論: 指示—行動 binding の同定可能性
- **未解決の中身**: Shortcut Learning（CoRL 2025）のデータ断片化理論とベイズ分解以外に、「どの条件で模倣損失が接地した解と検索キー解を区別できるか」の理論はない。ECT は「条件付き行動分布の台が狭いと区別できない」と述べたが定式化していない。
- **検索結果**: 「identifiability / language-conditioned imitation / conditional mutual information」で該当なし（古い言語条件付き RL の理論のみ）。
- **実現性**: 計算は不要だが、理論単体は通りにくい。A〜C の実験に小さな命題として添える形が現実的【推測】。

### J. 埋まり始めたが「2 本目」の余地がある領域
- **否定指示**: NegaAlign（2610.11952、2026-10-08）が初。GR00T・π0・π0.5 の否定指示成功率 1〜15% を、凍結モデルに挿入層だけで 58〜88% に。ただし否定文は手作りで、自由な言い回しは未検証。
- **How-level 指示**: 2610.21251 が診断のみで対策なし。
- **空間語・序数・個数**: Spa-Bench が実機で診断。Counting は 0/150。対策なし。
- **ナビゲーション**: NavBind（2610.29203、2026-10-21）が R2R-CE/RxR-CE の分岐点 3,252 件で「指示を変えても行動分布が変わらない」を診断。操作と同じ構図で、対策はまだ。
- **階層制御での高位指示の追従**: GDM の Hi-VLA 研究は「低位 VLA の指示追従が階層全体を決める」と結論したが、サブゴール追従率そのものは測っていない。Explicit Language Memory（2608.04765）も進捗追跡が主眼。
- **実機での系統的診断**: RADAR、Spa-Bench、Cosmos3 Greek（「単一ゴールのベンチは誤った指示でも 82.6% で、測定の罠」と明記）が少数。

---

## 3. 直近 3 週間（2026-09-30〜10-22）に出たこの主題の論文

競争の速さの証拠として列挙する。いずれも調査メモ §8・§9 または本メモ §5 に収録。

| 日付 | 論文 | 位置づけ |
|---|---|---|
| 09-30 | CRAFT: Same Scene, Different Task（KAIST, 2610.00524） | 反事実ペア学習。π0・π0.5・GR00T N1.7 |
| 09-末 | When Instructions Retrieve Trajectories / ECT（2609.39971） | 「検索キー」説と反事実ペア学習 |
| 10-07 | Encoded but Not in Control（2610.06235） | 相関的機械論解析。WAM も対象 |
| 10-08 | Tell Robot What Not to Do / NegaAlign（2610.11952） | 否定指示の初のベンチ＋対策 |
| 10-初 | Wulff & Cangelosi（2610.10178） | patching・CKA |
| 10-初 | Rephrase Before You Act（2610.10526） | 推論時の指示書き換え |
| 10-中 | Spa-Bench（2610.12696） | 実機の空間語 |
| 10-14 | Causal Analysis of Robot Foundation Models（2610.20750） | 因果的 patching |
| 10-中 | Same Goal, Different Words（2610.16242） | WAM の言い回し感度 |
| 10-中 | Paraphrase Consensus Is Not Self-Consistency（2610.19438） | language share、否定的結果 |
| 10-中 | Do VLAs Really Follow Instructions? How-level（2610.21251） | How-level 診断 |
| 10-中 | VLA-Retain（2610.27698） | 摂動軸間の干渉 |
| 10-21 | NavBind（2610.29203） | ナビでの指示—行動 binding 診断 |

---

## 4. 推奨

### 4.1 やらない方がよいもの
- LIBERO 派生の新しい診断ベンチ単体、CFG の変種、言い換え拡張単体、「π0.5 も GR00T も言語を無視する」という現象の再報告。いずれも §1 のとおり 4 本以上が先行し、差分を示せない。

### 4.2 現在の環境（GR00T N1.7 と π0.5、LIBERO 系）で新規性が高く実現可能な組合せ
1. **A + C: 「微調整中に言語依存がいつ・どの経路で崩れるか」を因果的に追跡する。** チェックポイント列に Causal Analysis 型の patching と language share / LGS / 接地率を適用し、GR00T N1.7 と π0.5 で崩壊の時刻と層位置を比較する。学習は通常の微調整 1〜2 本で足り、既存のどの論文とも重ならない。
2. **B を 1 の上に載せる**: SAM・凍結＋LayerNorm・言い換え LoRA・反事実ペアの 4 対策が「崩壊のどの段階・どの層を救うか」を同じ指標で示す。これで「VLA 全般に効くか」という問い（読書ガイド §2）に初めて直接答えられる。
3. **E を副産物として出す**: 1 で得た「言語成分の因果効果」を推論時の monitoring 信号にし、指示無視をオンラインで検出する。学習不要で、別の短い論文になる【推測】。
4. **F は 2 本目の主題候補**: 物体参照の反事実ペア（名前と位置の結合を切る）。LIBERO で配置を変えてデータを作れる。

### 4.3 新規性を守るための評価設計（どれを選んでも）
- 同一場面・別指示（反事実指示）で測る。言い換え頑健性だけでは「言語を使っている」証拠にならない。
- 成功率に加えて接地率（正しい対象に触れたか）と biased 率（学習タスクを実行したか）を出す。成功率は誤った物体を動かして上がる。
- held-out 言い換えでの成功率を併記する（Paraphrase Consensus の提言）。
- 2 モデル以上。1 モデルの結果は §6-7 のアーキテクチャ依存で一般化できない。
- 「言語能力の保持」（VQA）と「言語利用」（language share）を分けて報告する。

---

## 5. 今回新たに見つけた論文（調査メモ未収録、22 本）

### この問題を直接扱うもの
- CRAFT: Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs（KAIST, 2026-09-30）: https://arxiv.org/abs/2610.00524
- Causal Analysis of Robot Foundation Models Enables Efficiency Gains（Swann, Geiger, Fel ほか, 2026-10-14）: https://arxiv.org/abs/2610.20750
- VLA-Trace: Diagnosing VLA Models through Representation and Behavior Tracing（X-Humanoid, 2026-05/08）: https://arxiv.org/abs/2605.30117 ── π0.5 は生成時に指示トークンを除いても 1.3pt しか落ちない。反事実指示（moka pot → frypan）で 75 → 30% だが他の編集は効かず「元の視覚タスク構成に偏ったまま」
- Tell Robot What Not to Do: A Negation Understanding Perspective / NegaAlign・NegaBench（華南理工・HIT, 2026-10-08）: https://arxiv.org/abs/2610.11952
- NavBind: Diagnosing Instruction-Action Binding at Route Forks in Navigation VLAs（2026-10-21）: https://arxiv.org/abs/2610.29203
- VLAs Are Confined yet Capable of Generalizing to Novel Tasks（Quanyi Li, IJCAI 2026）: https://www.ijcai.org/proceedings/2026/843 ／ arXiv 2505.03500 ── text latent の混合で π0 の libero-ood 9 → 83%。「spatial overfitting」
- Learning What to Say to Your VLA: Mostly Harmless VLA Steering（CMU Jeong・Swamy・Bajcsy, 2026-06-10）: https://arxiv.org/abs/2606.12299 ── 「言語で引き出せない行動がある」「意味の近い指示が全く違う行動を生む」。閉ループ言語フィードバック方策＋conformal 校正。π0.5 で +24.7%（シミュ）、+65.0%（実機）
- When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in VLA Models（2026-06-10）: https://arxiv.org/abs/2606.11906
- Beyond English: Uncovering the Multilingual Gap in VLA Models（2026-06-14）: https://arxiv.org/abs/2606.15714
- Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 VLA Policy（2026-09-07）: https://arxiv.org/abs/2609.07470
- Bring My Cup! Personalizing VLA Models with Visual Attentive Prompting（POSTECH, 2512.20014）: https://arxiv.org/abs/2512.20014 ── Correct Movement Ratio（正しい対象に触れたか）を指標化
- What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents（GDM, 2026-06/09）: https://arxiv.org/abs/2606.10267 ── 狭い微調整で π0.5 の言い換え応答性が落ち 73.3 → 32.0%

### 周辺（失敗検出・RL・階層・その他）
- SAFECAST: Robust Failure Detection for VLA Policies with Contrast-Set Training and Calibration（USC Thomason, 2026-08-04）: https://arxiv.org/abs/2608.04246
- INSIGHT: Inference-time Sequence Introspection for Generating Help Triggers（Yale, 2510.01389）: https://arxiv.org/abs/2510.01389
- Explicit Language Memory for Long-Horizon Planning in VLA Models（復旦, 2026-08-05）: https://arxiv.org/abs/2608.04765
- Steerable VLA Policies for Embodied Reasoning and Hierarchical Control（Berkeley Levine, 2602.13193）: https://arxiv.org/abs/2602.13193 ── 「VLM と VLA の界面が自然言語であることが操舵を制限する」として多段階の合成コマンドで学習
- Robust VLN with Awareness of Instruction-Execution Progress（2026-10-09）: https://arxiv.org/abs/2610.13104
- Ask-to-Clarify（2509.15061）: https://arxiv.org/abs/2509.15061 ／ Adaptive Question Generation for Ambiguous Robot Instructions（2610.23874）: https://arxiv.org/abs/2610.23874
- SimpleVLA-RL（ICLR 2026, 2509.09674）: https://arxiv.org/abs/2509.09674 ／ πRL（2510.25889）: https://arxiv.org/abs/2510.25889 ── 言語追従の評価なし（§2-D の根拠）
- Counterfactual VLA（NVIDIA, CVPR 2026, 2512.24426）: https://arxiv.org/abs/2512.24426 ── 運転。自己の計画の反事実推論で、言語指示の接地は扱わない
- VISTA: Track-Following Preference Optimization（2602.05049）: https://arxiv.org/abs/2602.05049 ── 視覚依存の弱さ（vision-action misalignment）を測る。言語側の姉妹問題
- vla-eval: A Unified Evaluation Harness for VLA Models（2603.13966）: https://arxiv.org/abs/2603.13966 ── 14 ベンチ・6 モデルサーバ。言語依存の指標は持たない
- VLA in Robotics: A Survey of Datasets, Benchmarks, and Data Engines（2604.23001）: https://arxiv.org/abs/2604.23001 ── 「成功率が支配的な指標」「既存ベンチは推論失敗の症状を示すが診断構造に乏しい」。指示追従の指標には触れない

---

## 6. 検索して見つからなかったもの（空白の根拠）

| 検索した問い | 結果 |
|---|---|
| SAM（Flatness）の π0.5 以外での再現 | なし |
| 複数の対策を同一条件で複数モデルに適用した比較研究 | CRAFT の Table 1（5 設定 × 3 モデル）のみ |
| 微調整中の言語依存の時間変化（training dynamics） | なし |
| RL 事後学習（SimpleVLA-RL、πRL、RLinf など）が言語追従に与える影響 | なし |
| 展開時に「誤物体へ向かっている／言語を使っていない」を検出する研究 | SAFECAST が言い換え contrast set を使うが、誤物体検出は未評価 |
| 言語条件付き模倣学習での指示—行動 binding の同定可能性（理論） | なし（Shortcut Learning のデータ断片化理論、ベイズ分解のみ） |
| 場面あたり指示数のスケーリング | Cosmos3 Greek の 1 点（7 言い回しで罰則半減）のみ |
| WM・WAM の言語版反事実学習 | なし（行動版の WorldSync・DreamTrue のみ） |
| この主題の専用ワークショップ・チャレンジ | なし（LIBERO-Plus は CVPR 2026 ポスター、LangGap はコード公開のみ） |

- 読めなかったもの: Causal Analysis の所属（本文未取得）、NegaAlign の結論節、NavBind の対象モデルと数値、Beyond English のモデル名と数値、CRAFT の付録。
