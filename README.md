# Transfer

Claude の継続学習プロジェクトから spark-0ce8 (`/home/yuyaomg/work`) へ
調査メモなどのファイルを受け渡すためのリポジトリ。

- `research/` : 論文調査メモ (Markdown)。spark-0ce8 側では `/home/yuyaomg/work/research/` に配置する。

更新の取り込み: spark-0ce8 上で `cd ~/work/Transfer && git pull` のあと、
`cp research/*.md ~/work/research/` で反映する。
