実験で使用したwebアプリの説明

[開発ツール]
processingのp5.js

[フォルダ名で条件を見分ける]
chatItoI_name_js　→　人格固定・名乗りあり
chatItoI_noname_js　→　人格固定・名乗りなし
chatItoE_name_js　→　人格切り替え・名乗りあり
chatItoE_noname_js　→　人格切り替え・名乗りなし

[確認事項]
各条件ごとでLLM側の発話内容が異なる部分がある
LLMが出力する画像はすべての条件で同一

[発話内容を変更したい場合]
chatItoE_name_jsを例とする

186行目から212行目の間を変更することで内容を変更可能
scenario.push(new ChatMessage(false, "〇〇", null));の〇〇を変更する．「null」の部分を画像のファイル名にすることで画像を出力することが可能

[実行方法]
processingの場合は実行ボタンを押す
アンケートページで組み込んで実行させるには，quesフォルダにあるiframeProvider.phpとquesMgr.jsを用いて表示させる