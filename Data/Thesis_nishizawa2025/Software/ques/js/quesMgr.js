/* ====================================
 アンケートマネージャ

 TODO
 - マウスクリック時のエラー除去
 - repFileMgrは1つ目のrepeatStartのファイルしか管理できない問題がある
   → phpでファイル名を供給するのが早いかも
    提供したファイル名はどうやって記録する？
- embededvidemoも非jwplayer化したい

2023-10-19 by YSWR
-B4の人たちが作成したTextArea3をマージ
  -命令形態を変更
-timeStampを追加　命令はtimeStampのみ

-Unity用のコードを追加
  -writeInfoCookieでクッキーを記録している前提の作りになっているので注意　
  -第１引数でUnityのBuildファイルを指定　（検索ワード:"UnityFrame"）
　-第２引数でpreloadかどうかを指定
  --preloadの場合は，qTalbeにdivを追加するのではなく，bodyに追加する
  --その後の再度命令文が来た場合は，qTableに要素を追加する
  --これによって事前のファイルロードを可能としている
 

2023-01-13
- likertExp3を追加
  - センタリングしてスケールと説明文を表示

2022-12-31
- makeAudioを新規追加
 - makeVideoと同様に再生チェック機能付き
  - 再生後にページボタンの表示
  - 再生後にリッカートスケールの表示(LikertYoko, LikertExp2, LikertExpと連動)
- makeMusicを廃止

2021-12-29
- デバッグモードでURL書き換えを行わないように
- リッカートスケールクリック時に塗りつぶしなしにする挙動を追加 

2021-11-17
- jwplayerを使わない方法を追加
  - 一時停止できない仕様に
 2021-11-06
- repeatの扱いを変更
 - 指定回数だけ繰り返す．ファイル数は関係ない
- videoとimageにphpでのファイル名提供機能を追加
 - どのファイルが提供されたのかは保存ファイルの先頭付近にまとめて保存
- LikertExp2の表示がずれる問題を修正
  - LikertExp2では説明文のところが未入力ハイライトされる．文がないと違和感があるので注意
- jwplayerが複数あると，2つ目以降は自動で再生してしまう問題や
 デバッグ時にプレーやが1つしか表示されない問題に対処

 2020-05-06
 - scaleExp2をscaleExpに名称変更
 - likertYokoを追加

 2019-07-07
 ok - xlsx形式で数値として扱えるように', 'からスペースを削除
 ok - 回答のチェックがうまく動作していないのを修正
 ok - チェックボックスに値を設定できるように
 ok - 画像とチェックボックスのセットを横配置できるように
 ok - imageboxの追加．この要素を親として座標指定で子要素を配置する
 ok - 空の親要素box命令を追加．この中に座標指定で子要素を配置する
 ok - Likert Scaleを横に並べて配置できるように(LikertExp2の追加)
 ok - vspaceがレイアウトに反映するように( clear: leftが重要っぽい)
      clear: leftで改行を明示できるかな
 ok - "px"の記述をconfファイルでは省略するように仕様変更
 ok - 相対座標指定のテキストエリア textArea2 (makeTextArea2()）の追加
 ok - 相対座標指定のテキスト text3 (makeText3()）の追加
 ok - Likert Scaleで文字を下に配置できるように(元からできた)
 ok - Likertのスペルを修正
 ok - JSでの音楽再生をjQueryで記述
 ok - Hspace(inline-block)の追加
 ok - selectionをjqueryで書く&位置の調整
 ok - flashチェックの削除
 ok - inline-blockでtextareaが使えるように
 ok - 連続して入力できる機能の追加(link命令追加)

 2019-07-04
 - scaleExp2: 内容指定機能付き説明
 - likert2: 左詰め配置で左右ラベルなしのlikert
 - hspace: 横スペース挿入
 - selectionやtextareaから下部のスペースを除去
 - musicで左詰めで配置するように変更

 2019-06-28
 - JSでの音楽再生(jplayerの採用)

 2016-07-22
  - confirm()の代わりにJQ UIのDialogを使う
  - electiveのチェックが動作しない問題の解決
  - リッカートスケールの度合い表示がずれる問題の解決
  - imageに対してもRepeatがきくように改良

  ========================================
*/
let quesMgr = {};

//------------------------------
// ロード
//------------------------------
quesMgr.load = function () {
    //戻る・進むボタンを無効化
    history.pushState(null, null, null);
    window.onpopstate = function (e) {
        history.pushState(null, null, null);
        return;
    };

    //更新ボタンを押した時
    window.onbeforeunload = function (e) {
        var message = '本当に更新してよろしいですか？';
        e.returnValue = message;
        return message;
    };

    this.qPages;
    this.qAnswers;
    this.crrPage;

    this.bgColor = "#FFFFFF";
    this.fgColor = "#000000";
    this.hlColor1 = "#e78f08";
    this.hlColor2 = "#e78f08";
    //this.hlColor3 = "#f8cfb8";
    this.hlColor3 = "#fadfc8";
    this.hlColor4 = "#dffac8";


    this.panelWidth = 320;

    this.confFile = { "ja": "questions.csv", "en": "questions_en.csv" };

    //システムメッセージ
    this.systemMsg = { "ja": {}, "en": {} };
    this.systemMsg.ja.cookieAlert = "ブラウザのCookieがオフになっています．<br>Cookieをオンにしてください．";
    this.systemMsg.en.cookieAlert = "Cookies are disabled. Please enable Cookies.";
    //this.systemMsg.ja.footerMsg = "※ ブラウザの戻る・進む・更新ボタンは押さないでください．";
    //this.systemMsg.en.footerMsg = "*NOTE* Please Do Not Use Your Browsers' Navigational Buttons (Back, Forward, Refresh)";
    this.systemMsg.ja.footerMsg = "ブラウザの更新ボタンは押さないでください．";
    this.systemMsg.en.footerMsg = "*NOTE* Please do not use Your browser's refresh button";
    //this.systemMsg.ja.confirmation = "このページへは戻れません．<br>この回答でよいですか？<br>";
    //this.systemMsg.en.confirmation = "You cannot go back to this page.<br>Would you like to proceed?<br>";
    this.systemMsg.ja.confirmation = "記入漏れはありませんか？<br>";
    this.systemMsg.en.confirmation = "Would you like to proceed?<br>";
    this.systemMsg.ja.sendButtonMsg = "回答を送信する";
    this.systemMsg.en.sendButtonMsg = "Submit";
    this.systemMsg.ja.noInputElective = "任意記入の項目に未記入のものがあります．<br>";
    this.systemMsg.en.noInputElective = "One or more optional fields are not filled.<br>";
    this.systemMsg.ja.noInputTitle = "未入力の項目";
    this.systemMsg.en.noInputTitle = "Non-filled Item";
    this.systemMsg.ja.noMapDataTitle = "未入力の地図情報";
    this.systemMsg.en.noMapDataTitle = "Incomplete Map Data";
    this.systemMsg.ja.noInputBody = "未入力の項目があります．";
    this.systemMsg.en.noInputBody = "One or more fields are not filled.";
    this.systemMsg.ja.noMapDataBody = "未入力の地図情報があります．";
    this.systemMsg.en.noMapDataBody = "Map data is not completed.";
    this.systemMsg.ja.noInputBack = "戻る";
    this.systemMsg.en.noInputBack = "Back";
    this.systemMsg.ja.wrongInputTitle = "入力値のエラー";
    this.systemMsg.en.wrongInputTitle = "Input value error";
    this.systemMsg.ja.wrongInputBody = "数字指定の項目に数字以外が含まれています．";
    this.systemMsg.en.wrongInputBody = "A non-numeric value is entered in the numeric field.";
    this.systemMsg.ja.wrongInputBack = "戻る";
    this.systemMsg.en.wrongInputBack = "Back";
    this.systemMsg.ja.confirmTitle = "確認";
    this.systemMsg.en.confirmTitle = "Confirmation";
    this.systemMsg.ja.confirmBody = "未入力の項目があります．";
    this.systemMsg.en.confirmBody = "One or more fields are not filled.";
    this.systemMsg.ja.confirmBack = "戻る";
    this.systemMsg.en.confirmBack = "Back";
    this.systemMsg.ja.confirmForward = "進む";
    this.systemMsg.en.confirmForward = "Forward";
    this.systemMsg.ja.buttonForward = "次へ";
    this.systemMsg.en.buttonForward = "NEXT";
    this.systemMsg.ja.IE_Error = "IEには対応していません．最新版のChromeかFirefox, Edgeをご使用ください．<br>";
    this.systemMsg.en.IE_Error = "This page does not support IE. Please use the latest version of Chrome, Firefox or Edge.<br>";
    this.systemMsg.ja.onlyOnce = "アンケートは一人一回のみ回答できます．<br>";
    this.systemMsg.en.onlyOnce = "Please note that you can answer this questionnaire only once.<br>";
    this.systemMsg.ja.quesClosed = "アンケートの実施は終了しました．";
    this.systemMsg.en.quesClosed = "This page was closed.";

    //ハッシュパラメータとURLパラメータの取得
    this.checkURLparams();

    //IEかどうかのチェック
    if (navigator.userAgent.indexOf('MSIE ') > -1 || navigator.userAgent.indexOf('Trident/') > -1) {
        this.showErrorMessage(this.systemMsg[this.lang].IE_Error);
        return false;
    }

    //IDの設定
    this.gid = this.code + this.randobet(10);

    //開始時刻
    this.startDate = this.makeStartDate();

    //指定の言語によって読み込みファイルを切り替え
    let fname = this.confFile[this.lang];

    //クッキーが設定されていなかったら中断
    
    if (!this.checkCookie()) {
        this.showErrorMessage(this.systemMsg[this.lang].cookieAlert);
        return false;
    }
    

    /*
    //実験IDを取得
    if ((this.gid = this.readCookie("gid")) === "") {
    let msg = "実験に参加していない可能性があります．<br><br>";
    msg += "<a href=\"start.html\">実験説明ページへ</a>";
    this.showErrorMessage(msg);
    return false;
    }

    //実験開始時間を取得
    if ((this.startDate = this.readCookie("startDate")) === "") {
    let msg = "実験に参加していない可能性があります．<br><br>";
    msg += "<a href=\"start.html\">実験説明ページへ</a>";
    this.showErrorMessage("実験に参加していない可能性があります．<br>");
    return false;
    }
    */

    //既に回答済みの場合
    if (this.isAgain === "off" && this.readCookie("status") === "finish") {
         this.showErrorMessage(this.systemMsg[this.lang].onlyOnce);
        return false;
      }
    

    //アスマーク調査(日本語版)のときにClosedになっているかどうか
    if (this.isASMClosed() && this.asm !== "") {
        this.showErrorMessage(this.systemMsg[this.lang].quesClosed);
        return false;
    }

    //アスマーク調査(英語版)のときにClosedになっているかどうか
    if (this.isASMEnClosed() && this.asme1 !== "" && this.asme2 !== "") {
        this.showErrorMessage(this.systemMsg[this.lang].quesClosed);
        return false;
    }

    //phpでファイル名を提供した場合の記録
    //this.providedFnames = "Provided file name and order,";

    this.providedFnames = "";

    this.crrPage = 0;

    this.setNoSelect();

    this.loadQuestions(fname);
    this.parseQuestions(this.qPages[this.crrPage]);

    //$('#qTable').css('display', 'flex');

    //アドレスバーの変更 (同じディレクトリでないとSPAで画像が読み込めない)
    //ダミーのhtmlにしておかないと，パラメータがセットされていない状態で再読み込みされてしまう
    if (this.debug !== "on") {
        const rurl = new URL(document.location).toString().replace(/index.html.*/, 'work-in-progress.html');
        history.replaceState(null, null, rurl);
    }
};

// ------------------------------
// ヘッダー
// ------------------------------
quesMgr.makeHeader = function () {
    let div = $('<div></div>')
        .attr('id', 'header')
        .html(this.systemMsg[this.lang].footerMsg)
        .css('right', '10%')
        .css('height', '15px')
        .css('font-size', '10pt')
        .css('margin-bottom', '20px')
        .appnend('<hr>');

    return div;
};

// ------------------------------
// フッター
// ------------------------------
quesMgr.makeFooter = function () {
    let pageNum = "- " + (this.crrPage + 1) + "/" + (this.qPages.length) + " -";

    $('<div class="footer"></div>')
        .append($('<div class="page_info"></div>').text(pageNum))
        .append($('<div class="footer-line"></div>'))
        .append($('<div class="footer-msg"></div>').text(this.systemMsg[this.lang].footerMsg))
        .appendTo('#qTable');
};

// ------------------------------
// リサイズ
// ------------------------------
quesMgr.resize = function (ev) {
    //let h = parseInt(YAHOO.util.Dom.getClientHeight());
    let w = parseInt(YAHOO.util.Dom.getClientWidth(), 10);
    //let panelX = w - (quesMgr.panelWidth) - 10;
    let panelX = (w - quesMgr.panelWidth) - (w - 800) / 2;

    if (quesegr.panel !== undefined) {
        quesMgr.panel.cfg.setProperty("x", panelX);
        quesMgr.panel.render("qTable");
    }
};

// ------------------------------
// エラーメッセージの表示
// ------------------------------
quesMgr.showErrorMessage = function (msg) {
    document.getElementById("qTable").innerHTML = msg;
};

// -------------------------------------
// クッキー書き込み
// -------------------------------------
quesMgr.writeCookie = function (key, value, days) {
    let d = new Date();
    d.setDate(d.getDate() + days);
    document.cookie = key + "=" + escape(value) + ";" +
        "expires=" + d.toGMTString() + ";";
};

// -------------------------------------
// クッキー読み込み
// -------------------------------------
quesMgr.readCookie = function (key) {
    if (key === "") { retrun; }

    let rexp = new RegExp(key + "=(.*?)(?:;|$)");
    if (document.cookie.match(rexp)) {
        return unescape(RegExp.$1);
    } else {
        return "";
    }
};

// -------------------------------------
// クッキー消去
// -------------------------------------
quesMgr.clearCookie = function (key) {
    let d = new Date();
    d.setDate(d.getDate() - 1);
    document.cookie = key + "=false" + ";" +
        "expires=" + d.toGMTString() + ";";
};

// ------------------------------
// テキストの選択禁止
// ------------------------------
quesMgr.setNoSelect = function () {
    document.onselectstart = function () { return false; };
};

// ------------------------------
// テキストの選択解禁
// ------------------------------
quesMgr.setTextSelect = function () {
    document.onselectstart = function () { return true; };
};

//------------------------------
//0で桁をそろえる
//------------------------------
quesMgr.nf = function (num, digit) {
    let base = String(num + Math.pow(10, digit));
    let formated = base.substr(base.length - digit, digit);
    return formated;
};

// -------------------------------------
// 質問のロード
// -------------------------------------
quesMgr.loadQuestions = function (fname) {
    let ret = $.ajax({
        //url: proxyURL + "loadCSV.php?",
        url: "php/loadCSV.php?",
        data: { file: fname },
        async: false
    }).responseText;

    this.makePages(ret);
    //parseQues(ret);
};

// -------------------------------------
// ページ分割
// -------------------------------------
quesMgr.makePages = function (ques) {
    let qmax, i, numUseFiles, files, startIdx, endIdx, numRepPage;

    //デバッグモードのチェック
    if (this.debug === "on") {
        this.qPages = [ques];
        this.setTextSelect();
    } else {
        this.qPages = ques.split(/\s+newpage.*/);
    }
    qmax = this.qPages.length;
    startIdx = 0;
    endIdx = 0;
    prop = "";

    //ページ内のrepeat命令を走査
    for (i = 0; i < qmax; i += 1) {
        //repeatStartの検出
        if (this.qPages[i].match(/\s+repeatStart\s*,\s*([0-9]+)\s*,\s*([a-z]+)\s*,\s*(.*)/)) {
            //console.log(RegExp.$1 + ", " + RegExp.$2 + ", " + RegExp.$3);
            numUseFiles = parseInt(RegExp.$1, 10);
            prop = RegExp.$2;
            files = RegExp.$3;

            if (files.match(/\.php/)) {
                // phpファイルだった場合
                files = $.ajax({
                    url: files + "?",
                    data: { cmd: "up" },
                    async: false
                }).responseText;
            }

            files = files.replace(/\"/g, "");
            files = files.split(/ +/);

            //console.log("files.length: " + files.length);
            startIdx = i;
        }

        //repeatEndの検出
        if (this.qPages[i].match(/\s+repeatEnd.*/)) {
            endIdx = i;
            break;
        }
    }

    //repeatするページ数(このページする×繰り返し回数)
    numRepPages = endIdx - startIdx;
    this.makePageRepetition(startIdx + 1, numRepPages, numUseFiles, prop, files);

    //this.qAnswers = new Array(this.qPages.length);
    this.qAnswers = [];
    for (i = 0; i < this.qPages.length; i += 1) {
        this.qAnswers[i] = [];
    }
    //alert(this.qPages.length);
};

// -------------------------------------
// 繰り返しページの作成
// -------------------------------------
quesMgr.makePageRepetition = function (startIdx, numRepPages, numUseFiles, prop, files) {
    let befPages, aftPages, amax, extPages,
        randFiles, i, j, fmax, r1, r2, tmp;

    if (files === undefined) { return; }

    //ページの追加
    befPages = this.qPages.slice(0, startIdx);
    aftPages = this.qPages.slice(startIdx + numRepPages);

    amax = numRepPages * numUseFiles; //ファイル数の繰り返しではなく，指定数の繰り返しとする
    extPages = [];
    for (i = 0; i < amax; i += 1) {
        for (j = startIdx; j < (startIdx + numRepPages); j += 1) {
            extPages.push(this.qPages[j]);
        }
    }

    this.qPages = befPages.concat(extPages, aftPages);

    //propの処理
    if (prop.match(/rand/)) {
        //ベースとなる配列の準備
        randFiles = [];
        fmax = files.length;
        for (i = 0; i < fmax; i += 1) {
            //for(j=0; j<rep; j+=1) {
            randFiles.push(files[i]);
            //}
        }

        //ファイル名をランダムな順序に
        fmax = randFiles.length;
        for (i = 0; i < fmax * 2; i += 1) {
            r1 = Math.round(Math.random() * (fmax - 1));
            r2 = Math.round(Math.random() * (fmax - 1));
            tmp = randFiles[r1];
            randFiles[r1] = randFiles[r2];
            randFiles[r2] = tmp;
        }

        randFiles = randFiles.slice(0, numUseFiles)

        //ファイル名管理オブジェクトを初期化
        this.repFileMgr.init(randFiles, startIdx, numRepPages);

        //テスト
        //console.log(this.repFileMgr.showSortedIndex());

    }
    //
    else if (prop.match(/fix/)) {
        //ファイル名管理オブジェクトを初期化
        files = files.slice(0, numUseFiles)
        this.repFileMgr.init(files, startIdx, numRepPages);
    }

};

// ----------------------------------------
// ファイル名管理オブジェクト (クロージャ)
// ----------------------------------------
quesMgr.repFileMgr = {
    init: function (files, startIdx, numRepPages) {
        let cnt = 0,
            startPage = startIdx,
            sorted = files.slice(0).sort();

        quesMgr.repFileMgr = {
            //カウンタを進める
            getFname: function () {
                let f = files[cnt];
                //console.log(f);
                cnt += 1;
                return f;
            },
            //表示するだけ
            showFname: function (num) {
                let id = num || cnt;
                return files[id];
            },
            //カウンタの値を表示
            showCounterVal: function () {
                return cnt;
            },
            //すべてのファイルを表示
            showAllFnames: function () {
                return files;
            },
            //繰り返しを開始したページ
            showStartPage: function () {
                return startPage;
            },
            //繰り返しの最後のページ
            showNumRepPages: function () {
                return numRepPages;
            },
            //ソートしたファイル名に合わせたインデックスを表示
            showSortedIndex: function () {
                let ary = [],
                    smax, fmax, i, j, tmp;

                smax = sorted.length;
                fmax = files.length;
                tmp = "";
                for (i = 0; i < smax; i += 1) {
                    if (tmp === sorted[i]) { continue; }
                    for (j = 0; j < fmax; j += 1) {
                        if (sorted[i] === files[j]) {
                            ary.push(j);
                            break;
                        }
                    }
                    tmp = sorted[i];
                }
                return ary;
            },
            //ソートしたファイル名を表示
            showSortedFnames: function () {
                return sorted;
            }
        };
    }
};

// -------------------------------------
// 質問内容のパース
// -------------------------------------
quesMgr.parseQuestions = function (ques) {
    let qtbl = document.getElementById("qTable");

    qtbl.innerHTML = "";
    this.pageButtonHidden = false;

    //qtbl.appendChild(this.makeHeader());

    document.body.scrollTop = 0;

    ques = ques.split("\n");
    for (let i = 0; i < ques.length; i += 1) {
        //コメントをスキップ
        if (ques[i].match(/^\/\//)) {
            continue;
        }

        let q = ques[i].replace(/\s+,\s+/g, ",").split(/,/);
        for (let j = 0; j < q.length; j += 1) {
            q[j] = q[j].replace(/^\s+/, "");
        }
        q[q.length - 1] = q[q.length - 1].replace(/\s+$/, "");
        if(q[0].match("selectList")){
            this.makeSelectionList(q);
        }
        else if (q[0].match("select2")) {
            this.makeSelection2(q);
        } else if (q[0].match("select")) {
            this.makeSelection(q);
        } else if (q[0].match("textAreaInline")) {
            this.makeTextAreaInline(q);
        } else if (q[0].match("textArea3")) {
            this.makeTextArea3(q);
        } else if (q[0].match("textArea2")) {
            this.makeTextArea2(q);
        } else if (q[0].match("textArea")) {
            this.makeTextArea(q);
        } else if (q[0].match("scaleExp")) {
            this.makeScaleExp(q);
        } else if (q[0].match("NasaTLX")) {
            this.makeNasaTLX();
        } else if (q[0].match("NasaPair")) {
            this.makeNasaPair();
        } else if (q[0].match("likertYoko")) {
            this.makeLikertYoko(q);
        }else if (q[0].match("likertExp7")) {
            this.makeLikertExp7(q);
        }else if (q[0].match("likertExp8")) {
            this.makeLikertExp8(q);
        }else if (q[0].match("likertExp6")) {
            this.makeLikertExp6(q);
        }else if (q[0].match("likertExp5")) {
            this.makeLikertExp5(q);
        }  else if (q[0].match("likertExp4")) {
            this.makeLikertExp4(q);
        }  else if (q[0].match("likertExp3")) {
            this.makeLikertExp3(q);
        } else if (q[0].match("likertExp2")) {
            this.makeLikertExp2(q);
        } else if (q[0].match("likertExp")) {
            this.makeLikertExp(q);
        }else if (q[0].match("likert2")) {
            this.makeLikert2(q);
        } else if (q[0].match("likert")) {
            this.makeLikert(q);
        } else if (q[0].match("link")) {
            this.makeLink(q);
        } else if (q[0].match("textSpan")) {
            this.makeTextSpan(q[1]);
        } else if (q[0].match("text3")) {
            this.makeText3(q);
        } else if (q[0].match("text2bf")) {
            this.makeText2bf(q);
        } else if (q[0].match("text2")) {
            this.makeText2(q);
        } else if (q[0].match("title")) {
            this.makeTitle(q[1]);
        } else if (q[0].match("text1")) {
            this.makeText(q[1]);
        } else if (q[0].match("checkbox2")) {
            this.makeCheckBox2(q);
        } else if (q[0].match("checkbox")) {
            this.makeCheckBox(q);
        } else if (q[0].match("radio")) {
            this.makeRadioButton(q);
        } else if (q[0].match("imagebox")) {
            this.makeImageBox(q);
        } else if (q[0].match("image")) {
            this.makeImage(q);
        } else if (q[0].match("audio")) {
            this.makeAudio(q);
        } else if (q[0].match("vspace")) {
            this.makeVspace(q);
        } else if (q[0].match("break")) {
            this.makeBreak();
        } else if (q[0].match("box")) {
            this.makeBox(q);
        } else if (q[0].match("hspace")) {
            this.makeHspace(q);
        } else if (q[0].match("video")) {
            this.makeVideo(q);
        } else if (q[0].match("unity")) {
            this.makeUnityFrame(q);
        }
         else if (q[0].match("iframeGidPre")) {
            this.makeIframeGidPre(q);
        } else if (q[0].match("iframeGid")) {
            this.makeIframeGid(q);
        } else if (q[0].match("iframe")) {
            this.makeIframe(q);
        } else if (q[0].match("sortableText")) {
            this.makeSortableText(q);
        } else if (q[0].match("sortableVideo")) {
            this.makeSortableVideo(q);
        } else if (q[0].match("writeInfoCookie")) {
            this.writeInfoCookie();
        } else if (q[0].match("checkImgmapFile")) {
            this.setCheckImgmapFile(q);
        } else if (q[0].match("asmeexit")) {
            this.asmEngExit();
        } else if (q[0].match("showcode")) {
            this.showCode();
        }else if (q[0].match("timeStamp")) {
            this.timeStamp();
        }else if (q[0].match("canvasSelectingImage")) {
            this.canvasSelectImage(q);
        }else if (q[0].match("getRandFileText")) {
            this.getRandFileText(q);
        }else if (q[0].match("nameCheck")) {
            this.makeNameCheck(q);
        }
    }

    this.makePageButtons();
    this.makeFooter();
    window.scrollTo(0, 0);
};

// -------------------------------------
// 縦方向のスペースの挿入
// -------------------------------------
quesMgr.makeVspace = function (param) {
    let val = param[1] || "40"
    $("<div class='vspace'></div>")
        //.css('height', val+"px")
        .css('margin-bottom', val + "px")
        .css('clear', 'left')
        //.css('float', 'none')
        //.css('display', 'block')
        //.css('display', 'flex')
        .appendTo('#qTable');
};

// -------------------------------------
// 横方向のスペースの挿入
// -------------------------------------
quesMgr.makeHspace = function (param) {
    let val = param[1] || "40"
    $("<div class='hspace'></div>")
        //.css('padding-left', val+"px")
        .css('margin-left', val + "px")
        .css('display', 'inline-block')
        //.text('__')
        //.css('display', 'flex')
        .appendTo('#qTable');
};

// -------------------------------------
// 横方向のスペースの挿入
// -------------------------------------
quesMgr.makeBreak = function () {
    $("<div></div>")
        .css('display', 'inline-block')
        .css('clear', 'left')
        .appendTo('#qTable');
};

// -----------------------------------------
// ID指定付きの画像タグ 位置決めに使う
// -----------------------------------------
quesMgr.makeImageBox = function (param) {
    let img_file = 'url(./conf/image/' + param[2].trim() + ')';

    $("<div></div>")
        .attr('id', param[1].trim())
        .css('position', 'relative')
        .css('clear', 'left')
        .css('background-image', img_file)
        .css('background-size', 'contain')
        .css('width', param[3] + 'px')
        .css('height', param[4] + 'px')
        .appendTo('#qTable');
};

// -------------------------------------
// 空のタグ(ID指定) 位置決めに使う
// -------------------------------------
quesMgr.makeBox = function (param) {
    $("<div></div>")
        .attr('id', param[1].trim())
        .css('position', 'relative')
        .css('clear', 'left')
        .css('width', param[2] + 'px')
        .css('height', param[3] + 'px')
        .appendTo('#qTable');
};

// -------------------------------------
// オーディオファイル再生
// -------------------------------------
quesMgr.makeAudio = function (param) {
    let that = this,
        fileName,
        params,
        attributes;

    if (param[1].match(/randFile/)) {
        fileName = "conf/sounds/" + this.repFileMgr.getFname();
    } else if (param[1].match(/\.php/)) { // phpファイルだった場合
        fileName = $.ajax({
            url: param[1],
            async: false
        }).responseText;
        this.providedFnames += "audio:" + fileName + ",";
        console.log(fileName);
    } else {
        fileName = param[1].replace(/^\s+/, "");
    }

    this.pageButtonHidden = true;
    this.pageLikertHidden = true;

    let qaid = "qaudio-" + fileName;
    let qaoid = "qaouter-" + fileName;
    let amsgid = "amsg-" + fileName;

    let elm = $('<div></div>')
        .attr("id", qaoid)
        .attr('align', 'center')
        .append($('<div></div>')
            .css("padding", "10px")
            .append(
                $("<audio></audio>")
                    .attr('id', qaid)
                    .attr('preload', "auto")
                    .css("margin-bottom", "0px")
                    .bind('click', function (e) { //領域クリックをキャンセル
                        e.preventDefault();
                        return;
                    })
                    .bind('ended', function () {
                        //alert("video end!");
                        $('#pageButton').css('display', 'block');
                        $('.likert').css('display', 'block');
                        let that = this; // <video>
                        let ctl = $(this).next(); //コントローラ
                        ctl.text("【もう一度再生する】").css("cursor", "pointer").bind("click", function () {
                            that.play();
                            $(this).text("再生中").css("cursor", "text").unbind("click");
                        });
                    })
                    .bind('pause', function (e) {
                        //this.play();
                    })
                    .append($('<source />')
                        .attr('src', fileName)
                        .attr('type', "audio/mp3"))
            )
            .append($("<div></div>")
                .attr("id", amsgid)
                .css("color", "#ffffff")
                .css("background-color", "#000000")
                .css("max-width", "640px")
                .css("height", "30px")
                .css("margin-top", "0px")
                .css("padding-top", "5px")
                .css("cursor", "pointer")
                .text("【再生する】")
                .bind("click", function () {
                    $(this).prev().get(0).play();
                    $(this).text("再生中").css("cursor", "text").unbind("click");
                })
            )
        )
        .css('margin-bottom', 40)
        .appendTo("#qTable")
};



// -------------------------------------
// ビデオの作成
// -------------------------------------
quesMgr.makeVideo = function (param) {
    let that = this,
        fileName,
        params,
        attributes;

    if (param[1].match(/randFile/)) {
        fileName = this.repFileMgr.getFname();
    } else if (param[1].match(/\.php/)) { // phpファイルだった場合
        fileName = $.ajax({
            url: param[1],
            async: false
        }).responseText;
        this.providedFnames += "video:" + fileName + ",";
        console.log(fileName);
    } else {
        fileName = param[1].replace(/^\s+/, "");
    }

    this.pageButtonHidden = true;

    let qvid = "qvideo-" + fileName;
    let qvoid = "qvouter-" + fileName;
    let vmsgid = "vmsg-" + fileName;

    let elm = $('<div></div>')
        .attr("id", qvoid)
        .attr('align', 'center')
        .append($('<div></div>')
            .css("max-width", "660px")
            .css("background-color", "#000000")
            .css("padding", "10px")
            .append(
                $("<video></video>")
                    .attr('id', qvid)
                    //.attr('controls', "")
                    //.attr('controlsList', "nodownload nofullscreen noremoteplayback")
                    .attr('disablePictureInPicture', "")
                    .attr('preload', "auto")
                    .attr('width', "640")
                    .css("margin-bottom", "0px")
                    .bind('click', function (e) { //領域クリックをキャンセル
                        e.preventDefault();
                        return;
                    })
                    .bind('ended', function () {
                        //alert("video end!");
                        $('#pageButton').css('display', 'block');
                        let that = this; // <video>
                        let ctl = $(this).next(); //コントローラ
                        ctl.text("【もう一度再生する】").css("cursor", "pointer").bind("click", function () {
                            that.play();
                            $(this).text("再生中").css("cursor", "text").unbind("click");
                        });
                    })
                    .bind('pause', function (e) {
                        //this.play();
                    })
                    .append($('<source />')
                        .attr('src', "conf/video/" + fileName)
                        .attr('type', "video/mp4"))
            )
            .append($("<div></div>")
                .attr("id", vmsgid)
                .css("color", "#ffffff")
                .css("background-color", "#000000")
                .css("max-width", "640px")
                .css("height", "30px")
                .css("margin-top", "0px")
                .css("padding-top", "5px")
                .css("cursor", "pointer")
                .text("【再生する】")
                .bind("click", function () {
                    $(this).prev().get(0).play();
                    $(this).text("再生中").css("cursor", "text").unbind("click");
                })
            )
        )
        .css('margin-bottom', 40)
        .appendTo("#qTable")
};

// -------------------------------------
// 画像の作成
//　2025-12/30 図のラベルを置けるように修正しました
// -------------------------------------
quesMgr.makeImage = function (param) {
    let fileName;
    if (param[1].match(/randFile|fixFile/)) {
        fileName = this.repFileMgr.getFname();
        this.charImages.push(this.removeExtensionAndDirectory(fileName));
        // console.log(this.charImages);
    } else if (param[1].match(/\.php/)) { // phpファイルだった場合
        fileName = $.ajax({
            url: param[1],
            async: false
        }).responseText;
        this.providedFnames += "image:" + fileName + ",";
        this.charImages.push(this.removeExtensionAndDirectory(fileName));
        // console.log(this.charImages);
    } else {
        fileName = param[1].trim();
    }

    let $container = $('<div>')
        .attr('class', param[2])
        .append($('<img>')
            .attr('src', './conf/image/' + fileName)
            .css('width', param[3])
        );

    // キャプション（ラベル）がある場合
    if (param[4] && param[4].trim() !== "") {
        $container.append($('<div>')
            .addClass('caption')
            .css({
                'text-align': 'center',
                'font-size': '1.0em',
                'margin-top': '10px',
                'color': '#333'
                // 'font-weight': 'bold'
            })
            .text(param[4])
        );
    }

    $container.appendTo('#qTable');
};
// -----------------------------------------
// CHECKBOX
// -----------------------------------------
quesMgr.makeCheckBox = function (param) {
    let that = this;

    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (param[2] && param[2].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let fid = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);
    let chkFlg = false;
    let chkOffImg = "css/images/checkboxOff.png";
    let chkOnImg = "css/images/checkboxOn.png";

    $('<div class="checkbox"></div>')
        .attr("id", fid)
        .append($('<img>')
            .attr('src', chkOffImg)
            .unbind()
            .bind('click', function () {
                if (chkFlg === false) {
                    that.qAnswers[that.crrPage][idx].ans = param[1];
                    $(this).attr("src", chkOnImg);
                    chkFlg = true;
                } else {
                    that.qAnswers[that.crrPage][idx].ans = "";
                    $(this).attr("src", chkOffImg);
                    chkFlg = false;
                }
            })
            .mouseover(function () {
                $(this)
                    .css('background-color', 'rgba(251, 152, 11, 1)')
                    .css('box-shadow', '0 5px 20px rgba(251, 152, 11, 1.0)')
            })
            .mouseout(function () {
                $(this)
                    .css('background-color', 'none')
                    .css('box-shadow', 'none')
            })
        )
        .append($('<span class="checkbox-msg"></span>')
            .html(param[1])
        )
        .appendTo("#qTable");
};

// -----------------------------------------
// CHECKBOX その2 親要素からの相対位置指定
// checkbox2, 親id, 記録する値, elective/essential, テキスト, x座標, y座標
// 例) checkbox2, imb3, 0.08, elective, , 280, 130
// -----------------------------------------
quesMgr.makeCheckBox2 = function (param) {
    let that = this;

    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (param[3] && param[3].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let fid = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);
    let chkFlg = false;
    let chkOffImg = "css/images/checkboxOff.png";
    let chkOnImg = "css/images/checkboxOn.png";

    $("<div></div>")
        .attr('id', fid)
        .attr('class', 'checkbox')
        .css('position', 'absolute')
        .css('left', param[5] + 'px')
        .css('top', param[6] + 'px')
        .append($('<img>')
            .attr('src', chkOffImg)
            .unbind()
            .bind('click', function () {
                if (chkFlg === false) {
                    that.qAnswers[that.crrPage][idx].ans = param[2].trim();
                    $(this).attr("src", chkOnImg);
                    chkFlg = true;
                } else {
                    that.qAnswers[that.crrPage][idx].ans = "";
                    $(this).attr("src", chkOffImg);
                    chkFlg = false;
                }
            })
            .mouseover(function () {
                $(this)
                    .css('background-color', 'rgba(251, 152, 11, 1)')
                    .css('box-shadow', '0 5px 20px rgba(251, 152, 11, 1.0)')
            })
            .mouseout(function () {
                $(this)
                    .css('background-color', 'none')
                    .css('box-shadow', 'none')
            })
        )
        .append($('<span></span>')
            .css("position", "relative")
            .css("left", "10px")
            .css("top", "-10px")
            .html(param[4])
        )
        .appendTo('#' + param[1].trim());
};

// ------------------------------------------
// テキストエリアの作成(インラインブロック)
// ------------------------------------------
quesMgr.makeTextAreaInline = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    that = this;

    if (param[4].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    $('<input type="text"></input>')
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .attr("value", "")
        .attr("type", param[1])
        .attr("placeholder", param[3])
        .css("width", param[2])
        .css("display", "inline-block")
        .unbind()
        .change(function () {
            let t = this.value;
            t = t.replace(/\"/g, "″");
            t = t.replace(/,/g, "，");
            t = t.replace(/\n+/g, "<br>");
            if (param[1] === "tel" || param[1] === "number") {
                t = that.zenkaku2float(t);
                if (isNaN(t) || t < 0) {
                    $('#noinput-title').text(that.systemMsg[that.lang].wrongInputTitle);
                    $('#noinput-body').text(that.systemMsg[that.lang].wrongInputBody);
                    $('#noinput-back').text(that.systemMsg[that.lang].wrongInputBack)
                    $('#noinput').modal('show');
                    this.value = "";
                    return;
                }
            }
            this.value = t;
            that.qAnswers[that.crrPage][idx].ans = t;
        })
        .appendTo("#qTable");
};

// -------------------------------------
// テキストエリアの作成
// -------------------------------------
quesMgr.makeTextArea = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    that = this;

    if (param[3].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    $("<textarea></textarea>")
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .attr("value", "")
        //.attr("cols", param[1])
        //.attr("rows", param[2])
        .css("position", "relative")
        //.css("left", "30px")
        .css("width", param[1])
        .css("height", param[2])
        //.css("margin-bottom", "30px")
        .unbind()
        .change(function () {
            let t = this.value;
            t = t.replace(/\"/g, "″");
            t = t.replace(/,/g, "，");
            t = t.replace(/\n+/g, "<br>");
            that.qAnswers[that.crrPage][idx].ans = t;
        })
        .appendTo("#qTable");
};

// --------------------------------------------
// テキストエリア その2 親ID指定＋相対座標指定
// --------------------------------------------
quesMgr.makeTextArea2 = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    that = this;

    if (param[2].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    $("<textarea></textarea>")
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .attr("value", "")
        .css('position', 'absolute')
        .css('left', param[3] + 'px')
        .css('top', param[4] + 'px')
        .css("width", param[5] + 'px')
        .css("height", param[6] + 'px')
        .css("border", "solid 2px #000000")
        .unbind()
        .change(function () {
            let t = this.value;
            t = t.replace(/\"/g, "″");
            t = t.replace(/,/g, "，");
            t = t.replace(/\n+/g, "<br>");
            that.qAnswers[that.crrPage][idx].ans = t;
        })
        .appendTo('#' + param[1].trim());
};


// --------------------------------------------
// テキストエリア その3 チェックあり
//配置については，textArea1に準拠


//textArea3, 300, 400, elective, count,500 =>500文字以上入れないと次へボタンが出ない
//textArea3, 200, 60, elective, password,hashpass,text=>id＋入力文字で合っていれば次へボタンが出る
// --------------------------------------------

quesMgr.makeTextArea3 = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    that = this;
    //次へボタンを隠す
    this.pageButtonHidden = true;

    if (param[3].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let infoText=$('<div class="wordcount"></div>')
    .append(param[4]=="count"?"0" + "/" + param[5] + "文字":param[6])
    .css("text-align", "left") 
    .appendTo("#qTable");



    let textArea =$('<textarea class="textarea"></textarea>')
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .attr("value","")
        .attr("readonly", false)
        .css("position", "relative")
        .css("width", param[1])
        .css("height", param[2]-30)
        .css("margin", "0 auto") // 中央寄せを設定
        .unbind()
        .change(function () {
            let t = this.value;
            t = t.replace(/\"/g, "″");
            t = t.replace(/,/g, "，");
            t = t.replace(/\n+/g, "<br>");
            that.qAnswers[that.crrPage][idx].ans = t;
            
        })
        .appendTo("#qTable");


        if (param[4] == "password") {
            // サイズ変更禁止
            textArea.css("resize", "none");
            // フォントサイズを2倍
            textArea.css("font-size", "200%");
            //中央寄せ
            textArea.css("text-align", "center");

        }

    textArea.on("input", function () {
        if (param[4] == "password") {
            let inputValue = this.value;
            
            // 半角英数字以外の文字を削除
            inputValue = inputValue.replace(/[^a-zA-Z0-9]/g, '');
            // 最低文字数を3文字に制限
            if(inputValue.length==3)
            {
                async_digestMessage(inputValue).then(function(hashHex){
                    
                    if(hashHex==param[5])
                    {
                        //表示
                        $('#pageButton').css('display', 'block');
                        //textAreaを入力不可に
                        textArea.attr("readonly", true);
                        infoText.text("次のページに進んでください");

                    }
                    else{
                        infoText.text("パスワードが違います");
                    }
            });}
            else if (inputValue.length > 3) {
                inputValue = inputValue.substring(0, 3); // 3文字未満の場合、最初の3文字だけ残す
                ///infoText.text(param[6]);
            }
            else{
                infoText.text(param[6]);
            }

            this.value = inputValue;
        }


        else if (param[4] == "count") {
            result = checkCountText(textArea.val(), param[5]);
            infoText.text(result.textCount + "/" + param[5] + "文字");

            if (result.isValid) {
                infoText.css({"color": "green" });
                $('#pageButton').css('display', 'block');
            } else {
                infoText.css({"color": "black" });
                $('#pageButton').css('display', 'none');
            }
        }
    });

    

    //与えられた文字列をハッシュ化する
    function async_digestMessage(message) {
        return new Promise(function(resolve){
        var msgUint8 = new TextEncoder("utf-8").encode(message);
        crypto.subtle.digest('SHA-256', msgUint8).then(
            function(hashBuffer){
                var hashArray = Array.from(new Uint8Array(hashBuffer));
                var hashHex = hashArray.map(function(b){return b.toString(16).padStart(2, '0')}).join('');
                return resolve(hashHex);
            });
        })
    }

    function checkPasswork(text){
        
    }
    //与えられた文字列数を改行コードなど無しで特定文字数に達しているか判定する
    function checkCountText(text, count){
        textCount=text.replace(/\r?\n/g, "").length;
        return { isValid: textCount >= count, textCount: textCount };
    }


};
// -------------------------------------
// SELECTの作成
// -------------------------------------
quesMgr.makeSelection = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    let that = this;

    let sel = $("<select></select>")
        .attr('name', param[1])
        .css('background-color', this.bgColor)
        .change(function () {
            that.qAnswers[that.crrPage][idx].ans = this.options[this.selectedIndex].value;
        });

    //sel.append($('<option value="" hidden></option>').text(param[3]));
    for (let i = 4; i < param.length; i += 1) {
        sel.append($('<option></option>').val(param[i]).html(param[i]));
    }

    $('<div class="select-item"><form></form></div>')
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .html(param[3] + "　")
        //.css('width', '400px')
        //.css('height', '50px')
        .append(sel)
        .appendTo("#qTable");
};

// -------------------------------------
// SELECT Listの作成
//ほかのSelectで選ばれた要素は選択できないようなSelectの集合体
//回答は文字列配列形式で["Aの選択肢","Bの選択肢",...]
// -------------------------------------

quesMgr.makeSelectionList = function (param) {
    this.qAnswers[this.crrPage].push("");//空の要素を追加(ここに選択された値が入る)
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    let that = this;

    let num=param[2];//作る必要のあるselectの数
    let ansList=[];//最終的にcsv形式で保存される回答
    let ansLockList=[];//各selectの選択番号
    let nameList=param[3].split("_");//選択肢の名前のリスト
    let appendItem=$('<div class="selecter-list"><form></form></div>')
    .css("background-color", this.bgColor)
    .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4));
    for(let i=0;i<num;i++){
        
        ansList.push("");
        ansLockList.push(0);//現在の選択番号
        let sel = $("<select></select>")
        .attr('name', param[1])
        .css('background-color', this.bgColor)
        .change(function () {
            ansList[i] = this.options[this.selectedIndex].value;
            ansLockList[i]=this.selectedIndex;
            if(ansLockList.includes(0)!=true){
                that.qAnswers[that.crrPage][idx].ans = ansList.join("_");//回答を一時保存　次に進めるようにする
            }else{
                that.qAnswers[that.crrPage][idx].ans = "";//回答を終了させない          
            }
            
            // 他の select で利用できない選択肢をロック
            for (let j = 0; j < num; j++) {
                    let otherSel = $(this).parent().find('select').eq(j);
                    let otherOptions = otherSel.find('option');
                    
                    otherOptions.prop('disabled', false);//一旦すべての選択肢をロック解除
                    
                    //ansLockListに入っている選択肢をロック
                    for (let k = 0; k < ansLockList.length; k++) {
                        //回答がまだない場合はロックしない
                        if(ansLockList[k] != 0){
                            //console.log(ansLockList[k]+"番目の選択肢をロックします",k);
                            otherOptions.eq(ansLockList[k]).prop('disabled', true);    
                        }

                    }
            }
            //console.log(ansLockList);
        });
        //質問を追加　j=4はparamの開始位置　param[4]
        for (let j = 4; j < param.length; j += 1) {
            sel.append($('<option></option>').val(param[j]).html(param[j]));
        }
        let textItem=$('<span class="selecter-list-name" style=margin-right:3px;>'+nameList[i]+'</span>')
        appendItem.append(textItem);
        appendItem.append(sel);
        let vpace=$('<div class="vspace" style="margin-bottom: 10px; clear: left;"></div>');
        appendItem.append(vpace);
        
    }

    appendItem.appendTo("#qTable");

    
    //sel.append($('<option value="" hidden></option>').text(param[3]));
    
}



// ----------------------------------------
// SELECTの作成 その2 (説明をリストに含む)
// ----------------------------------------
quesMgr.makeSelection2 = function (param) {
    this.qAnswers[this.crrPage].push("");
    let idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    let that = this;

    let sel = $("<select></select>")
        .attr('name', param[1])
        .css('background-color', this.bgColor)
        .change(function () {
            that.qAnswers[that.crrPage][idx].ans = this.options[this.selectedIndex].value;
        });

    sel.append($('<option value="" hidden></option>').html(param[3]));
    for (let i = 4; i < param.length; i += 1) {
        sel.append($('<option></option>').val(param[i]).html(param[i]));
    }

    $('<div class="select-item"><form></form></div>')
        .attr("id", "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        //.html(param[3] +  "　")
        //.css('width', '400px')
        //.css('height', '50px')
        .append(sel)
        .appendTo("#qTable");
};

// -------------------------------------
// NASA-TLXの作成
// -------------------------------------
quesMgr.makeNasaTLX = function () {
    this.flgNasaTLX = true;
    let divNTLX = document.createElement("div");

    let labels = [
        "精神的要求",
        "身体的要求",
        "時間的圧迫感",
        "作業達成度",
        "努力",
        "不満"
    ];

    for (let i = 0; i < 6; i += 1) {
        this.qAnswers[this.crrPage].push("");
        let idx = this.qAnswers[this.crrPage].length - 1;

        let label = document.createElement("div");
        label.id = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);
        label.innerHTML = labels[i];
        label.style.height = "30px";
        label.style.width = "140px";
        divNTLX.appendChild(label);
        divNTLX.appendChild(this.makeSliderTag(i));
    }

    let qtbl = document.getElementById("qTable");
    qtbl.appendChild(divNTLX);

    //スペーサ
    let div = document.createElement("div");
    div.style.height = "50px";
    qtbl.appendChild(div);

    //スライダの作成
    let that = this;
    for (let i = 0; i < 6; i += 1) {
        let slider = YAHOO.widget.Slider.getHorizSlider(
            "sliderbg-" + this.nf(i, 2), "sliderthum-" + this.nf(i, 2), 0, 299);
        slider.setValue(150, true);
        slider.num = i;
        this.qAnswers[this.crrPage][slider.num] = { ans: "", elec: false };
        slider.subscribe("change", function (oVal) {
            if (this.valueChangeSource !== 1) { return; }
            let nVal = Math.round(oVal / 3.0);
            that.qAnswers[that.crrPage][this.num].ans = nVal;
        });
    }

    this.makeNasaExpPanel();
};

// -------------------------------------
// NASA-TLXの一対比較(重み用)の作成
// -------------------------------------
quesMgr.makeNasaPair = function () {
    let divNPair = document.createElement("div");

    let labels = [
        "精神的要求",
        "身体的要求",
        "時間的圧迫感",
        "作業達成度",
        "努力",
        "不満"
    ];

    let tmp;
    for (let i = 0; i < 6; i += 1) {
        for (let j = i + 1; j < 6; j += 1) {
            this.qAnswers[this.crrPage].push("");
            let idx = this.qAnswers[this.crrPage].length - 1;
            let divPair = document.createElement("div");
            divPair.style.clear = "both";
            divPair.style.height = "30px";
            divPair.style.backgroundColor = this.bgColor;

            //if (j % 2 === 0) {
            if (i !== tmp) {
                let divLeft = this.createClickableDiv(labels[i], idx);
                let divRight = this.createClickableDiv(labels[j], idx);
                tmp = i;
            } else {
                let divLeft = this.createClickableDiv(labels[j], idx);
                let divRight = this.createClickableDiv(labels[i], idx);
                tmp = -1;
            }

            let divNum = document.createElement("div");
            divNum.id = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);
            divNum.innerHTML = (idx + 1) + ".";
            divNum.style.styleFloat = "left";
            divNum.style.cssFloat = "left";
            divNum.style.textAlign = "center";
            divNum.style.width = "30px";
            divNum.style.border = "solid 1px transparent";

            let divCenter = document.createElement("div");
            divCenter.innerHTML = "&hArr;";
            divCenter.style.styleFloat = "left";
            divCenter.style.cssFloat = "left";
            divCenter.style.textAlign = "center";
            divCenter.style.width = "35px";
            divCenter.style.border = "solid 1px transparent";

            divPair.appendChild(divNum);
            divPair.appendChild(divLeft);
            divPair.appendChild(divCenter);
            divPair.appendChild(divRight);

            divNPair.appendChild(divPair);
            divNPair.appendChild(document.createElement("br"));
        }
    }

    let qtbl = document.getElementById("qTable");
    qtbl.appendChild(divNPair);

    this.makeNasaExpPanel();
};

// -------------------------------------
// 用語説明パネル
// -------------------------------------
quesMgr.makeNasaExpPanel = function () {

    this.panel = new YAHOO.widget.Panel("exPanel", {
        width: this.panelWidth + "px",
        visible: true,
        //constraintoviewport: true,
        constraintoviewport: false,
        close: false,
        draggable: false,
        y: 50
    });

    let msg = "　【精神的要求】<br>" +
        "どの程度，精神的かつ知覚的活動が要求されましたか？" +
        "（例：思考，意思決定，計算，記憶，観察，検索など）" +
        "容易／困難，単純／複雑，寛大／過酷だったかを基準にしてください．" +
        "<br><hr>";

    msg += "　【身体的要求】<br>" +
        "どの程度，身体的活動が必要でしたか？" +
        "（例：押す，引く，回す，操作，活動するなど）" +
        "容易／困難，ゆっくり／きびきび，ゆるやか／努力を要する，" +
        "落ち着いていた／骨の折れるものだったかを基準にしてください．" +
        "<br><hr>";

    msg += "　【時間的切迫感】<br>" +
        "作業や要素作業の頻度や速さにどの程度，時間的圧迫感を感じましたか？" +
        "作業ペースはゆっくりしていて暇だったか，それとも急速で大変だったか．" +
        "<br><hr>";

    msg += "　【作業達成度】<br>" +
        "実験者によって設定された作業の達成目標の遂行について，" +
        "どの程度成功したと思いますか？" +
        "この目標達成における作業成績にどのくらい満足していますか？" +
        "<br><hr>";

    msg += "　【努力】<br>" +
        "あなたの作業達成レベルに到達するのにどのくらい一生懸命" +
        "（精神的および身体的に）作業を行わなければなりませんでしたか？" +
        "<br><hr>";

    msg += "　【不満】<br>" +
        "作業中どのくらい不安，落胆，いらいら，ストレス，不快感，" +
        "あるいは安心，喜び，満足，リラックス，自己満足を感じましたか？";

    let divMsg = document.createElement("div");
    divMsg.style.fontSize = "11 pt";
    divMsg.style.lineHeight = 1.3;
    divMsg.innerHTML = msg;

    this.panel.setHeader("説明");
    this.panel.setBody(divMsg);
    this.panel.render("qTable");
    this.resize();
};

// -------------------------------------
// クリック可能なエレメントの作成
// -------------------------------------
quesMgr.createClickableDiv = function (label, idx) {
    let cDiv = document.createElement("div");
    cDiv.style.styleFloat = "left";
    cDiv.style.cssFloat = "left";
    cDiv.style.textAlign = "center";
    cDiv.style.width = "100px";
    cDiv.innerHTML = label;
    cDiv.style.cursor = "pointer";
    cDiv.style.border = "solid 2px transparent";

    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    let that = this;
    cDiv.onclick = function () {
        let elms = this.parentNode.getElementsByTagName("div");
        for (let i = 0; i < elms.length; i += 1) {
            elms[i].style.border = "solid 2px transparent";
        }
        this.style.border = "solid 2px " + that.hlColor2;
        that.qAnswers[that.crrPage][idx].ans = this.innerHTML;
    };
    cDiv.onmouseover = function () {
        this.style.backgroundColor = that.hlColor1;
    }
    cDiv.onmouseout = function () {
        this.style.backgroundColor = this.parentNode.style.backgroundColor;
    }

    return cDiv;
};

// -------------------------------------
// スライダータグの生成
// -------------------------------------
quesMgr.makeSliderTag = function (idx) {
    let divSlider = document.createElement("div");

    let divLeft = document.createElement("div");
    divLeft.style.styleFloat = "left";
    divLeft.style.cssFloat = "left";
    divLeft.innerHTML = "低い";
    divLeft.style.width = "100px";
    divLeft.style.textAlign = "right";

    let divRight = document.createElement("div");
    divRight.style.styleFloat = "left";
    divRight.style.cssFloat = "left";
    divRight.innerHTML = "高い";
    divRight.style.textAlign = "left";
    //divRight.style.marginLeft = "20px";

    let divSliderBG = document.createElement("div");
    divSliderBG.id = "sliderbg-" + this.nf(idx, 2);
    divSliderBG.className = "yui-h-slider";
    divSliderBG.style.styleFloat = "left";
    divSliderBG.style.cssFloat = "left";
    divSliderBG.style.width = "316px";
    //divSliderBG.style.position = "relative";
    //divSliderBG.style.left = "10px";
    //divSliderBG.style.paddingLeft = "-10px";
    //divSliderBG.style.paddingRight = "10px";
    divSliderBG.style.height = "22px";
    divSliderBG.title = "Slider";
    divSliderBG.tabindex = "-1";
    divSliderBG.style.backgroundImage = "url(css/images/nasatlx-black3.png)";
    divSliderBG.style.backgroundPosition = "center center";
    divSliderBG.style.backgroundRepeat = "no-repeat";

    let divSliderThumb = document.createElement("div");
    divSliderThumb.id = "sliderthum-" + this.nf(idx, 2);
    divSliderThumb.className = "yui-slider-thumb";
    let sliderImg = document.createElement("img");
    //sliderImg.src = "http://yui.yahooapis.com/2.8.1/build/slider/assets/thumb-n.gif";
    sliderImg.src = "./yui/build/slider/assets/thumb-n.gif";
    divSliderThumb.appendChild(sliderImg);
    divSliderBG.appendChild(divSliderThumb);

    divSlider.style.height = "70px";
    divSlider.appendChild(divLeft);
    divSlider.appendChild(divSliderBG);
    divSlider.appendChild(divRight);
    divSlider.appendChild(document.createElement("br"));

    return divSlider;
};

// -------------------------------------------------
// リッカートスケール 横書き説明 一体化 左詰め配置
// 説明文字クリック版
// -------------------------------------------------
quesMgr.makeLikertYoko = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    //全体のDIV
    let div_main = $('<div class="scale likert"></div>')

    //テキストの表示+スペーサ
    $('<div class="text_s"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .html(param[4])
        .appendTo(div_main);

    //クリッカブルな文字の生成
    let div_nums = $('<div class="scale-label"></div>'),
        maxText = 0;
    for (let i = 0; i < num; i += 1) {
        if (maxText < param[i + 5].length) {
            maxText = param[i + 5].length;
        }
        $('<div class="scale-label-yoko-item"></div>')
            .html(param[i + 5])
            .attr('id', "l-" + (i + 1))
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_main);

    //横幅の指定
    $(div_nums).css('width', ((maxText + 1) * num) + 'em');

    // ----------- ライン -----------
    let divBars = $('<div class="scale-line"></div>')
    divBars.append('<div class="scale-part-r"></div>');
    //横幅の指定
    $(divBars).css('width', ((maxText + 1) * num) + 'em');
    for (let i = 0; i < num - 1; i += 1) {
        divBars.append('<div class="scale-part-b"></div>');
        divBars.append('<div class="scale-part-br"></div>');
    }
    divBars.append('<div class="scale-part-n"></div>');
    divBars.appendTo(div_main);

    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};
// ---------------------------------------------------
//type8 リッカートスケール 説明付き一体化 中央よせ　
//Exp6からの変更点
//左寄せを中央よせに変更
//竹内, 丸㔟(2025B4一同)がExp6を改変(20251217)
// ---------------------------------------------------
quesMgr.makeLikertExp8 = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let div_width = (32 * num) + "px";

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('height', '300px')
        .css('width', div_width) 
        //.css('min-width', div_width) // 古い設定
        .css('margin-top', '10px')
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('margin-bottom', '10px')

    
    function getTextWidth(text, font) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = font;
        return ctx.measureText(text).width;
    }
    if(param[5]=="randFile"){
        insertText = quesMgr.repFileMgr.nowText;
        text=param[4];
        text=text.replace("%randtext%",insertText);
    }
    else{
        text=param[4];
    }
    let text_left_margin = ((32 * num) - (getTextWidth(param[4], '12pt sans-serif')))/2 + 'px';    
    //テキスト部分
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .html(text)
        //左よせ
        //  ここで位置合わせを行います
        .css('width', 'max-content')  // テキストの長さに合わせて枠を広げる
        .css('max-width', '90vw')     // ただし画面幅の90%を超えたら改行する（画面外へのはみ出し防止）
        .css('white-space', 'nowrap') // 基本的に改行させない
        //.css('width', div_width)      // テキストボックスの幅をスケールと同じにする
        //.css('margin-left', 'auto')   // テキストボックス自体を中央に寄せる（スケールと重なるように）
        //.css('margin-right', 'auto')  // テキストボックス自体を中央に寄せる
        .css('text-align', 'left')    // ボックスの中で文字を左に寄せる
        //中央よせ
        //.css('text-align', 'center') 

        .css('font-weight', param[5] === 'bold' ? 'bold' : 'normal')
        .appendTo(div_main);

    //スケール部分
    let div_scale = $('<div></div>')
        .css('clear', 'both')
        .css('min-width', div_width)
        .css('width', div_width)
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('background-color', that.bgColor);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i ))//i+1で1~param[2]になる
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i )//i+1で1~param[2]になる
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_scale);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_scale);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_scale);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+6]);
    }
    

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '11.5px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }
    $('<div></div>')
        .css('clear', 'both')
        //.css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_scale);

    div_main.append(div_scale);
    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};
// ---------------------------------------------------
//type7 リッカートスケール 説明付き一体化 座標指定型
//ベースはlickert2
//説明文を.textではなく.htmlで追加することで<b>タグなどを利用できるよう変更
//getRandFileTextで得られるtextに対応
//吉原がExp2を改変(20240901)
// リッカートスケール 説明付き一体化 左詰め 座標指定
//likertExp7,0,0, 11, q1,,ロボットに<b>%randtext%</b>をさせる場合に<br><b>①の画像のような外見のロボット</b>を採用したいですか,randFile,全く採用したくない,,,,, どちらでもない,,,,,非常に採用したい 
//likertExp7,500,-320, 11, q2,,ロボットに<b>%randtext%</b>をさせる場合に<br><b>②の画像のような外見のロボット</b>を採用したいですか,randFile,全く採用したくない,,,,, どちらでもない,,,,,非常に採用したい 

// ---------------------------------------------------
quesMgr.makeLikertExp7 = function (param) {
    let x = (param[1]) + 'px';//x座標をどれだけ左上からずらすか
    let y = (param[2]) + 'px';//y座標をどれだけ左上からずらすか
    let num = Number(param[3]);//リッカートの分解能
    
    let elecFlg = param[5];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let text=param[6];
    if(param[7]=="randFile"){
        console.log("randFile");
        insertText = quesMgr.repFileMgr.nowText;
        text=param[6];
        text=text.replace("%randtext%",insertText);
    }
    


    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        //.css('left', x)
        //.css('top', y)
        .css('display', 'inline-block')
        .css('float', 'left')
        .css('height', '300px')
        .css('width', '50%')
        .css('margin-top', y)
        .css('margin-left', x)
        .css('margin-right', '20px')
        .css('margin-bottom', '10px');

    //テキストの表示+スペーサ
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .html(text)
        .css('font-size', '12pt')
        .append(
            $('<span></span>').css('margin', '20px')
        )
        .appendTo(div_main);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i + 1))
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i )//+1すると1から始まる
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_main);

    let lWidth = 360 + (32 * num) + "px";

    let divLikert = $('<div></div>')
        .css('clear', 'both')
        .css('height', '20px')
        //.css('width', lWidth)
        .css('background-color', that.bgColor)
        .append(div_main);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_main);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_main);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+8]);
        
    }
    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '12px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }

    $('<div></div>')
        .css('clear', 'both')
        .css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_main);

    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};

// ---------------------------------------------------
//type6 リッカートスケール 説明付き一体化 左よせ　
//説明文を.textではなく.htmlで追加することで<b>タグなどを利用できるよう変更
//getRandFileTextで得られるtextに対応
//likertExp6,段階数,id,必須回答フラグ,説明文,randFileFlag,リッカート補助説明,,,,~~
//例：likertExp6, 11, q1, essential,掃除,randFile, 全く任せたくない,,,,, どちらでもない,,,,, 非常に任せたい
//吉原がExp5を改変(20240830)
// ---------------------------------------------------
quesMgr.makeLikertExp6 = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let div_width = (32 * num) + "px";

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('height', '300px')
        .css('min-width', div_width)
        .css('margin-top', '10px')
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('margin-bottom', '10px')

    
    function getTextWidth(text, font) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = font;
        return ctx.measureText(text).width;
    }
    if(param[5]=="randFile"){
        insertText = quesMgr.repFileMgr.nowText;
        text=param[4];
        text=text.replace("%randtext%",insertText);
    }
    else{
        text=param[4];
    }
    let text_left_margin = ((32 * num) - (getTextWidth(param[4], '12pt sans-serif')))/2 + 'px';    
    //テキスト部分
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .html(text)
        //.css('margin-left', text_left_margin)
        //.css('font-size', '12pt')
        .css('text-align', 'left') 

        .css('font-weight', param[5] === 'bold' ? 'bold' : 'normal')
        .appendTo(div_main);

    //スケール部分
    let div_scale = $('<div></div>')
        .css('clear', 'both')
        .css('min-width', div_width)
        .css('width', div_width)
        //.css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('background-color', that.bgColor);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i ))//i+1で1~param[2]になる
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i )//i+1で1~param[2]になる
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_scale);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_scale);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_scale);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+6]);
    }
    

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '11.5px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }
    $('<div></div>')
        .css('clear', 'both')
        //.css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_scale);

    div_main.append(div_scale);
    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};
// ---------------------------------------------------
//type5 リッカートスケール 説明付き一体化 左よせ　
//説明文のみ，リッカートの中央にくるように配置
//説明文をBoldにすることも可能
//likertExp5,段階数,id,必須回答フラグ,説明文,boldフラグ,リッカート補助説明,,,,~~
//例：likertExp5, 11, q1, essential,掃除,bold, 全く任せたくない,,,,, どちらでもない,,,,, 非常に任せたい
//吉原がExp3を改変(20240807)
// ---------------------------------------------------
quesMgr.makeLikertExp5 = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let div_width = (32 * num) + "px";

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('height', '300px')
        .css('min-width', div_width)
        .css('margin-top', '10px')
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('margin-bottom', '10px')

    
    function getTextWidth(text, font) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = font;
        return ctx.measureText(text).width;
    }

    let text_left_margin = ((32 * num) - (getTextWidth(param[4], '12pt sans-serif')))/2 + 'px';    
    //テキスト部分
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .text(param[4])
        .css('margin-left', text_left_margin)
        .css('font-size', '12pt')
        .css('text-align', 'left') 

        .css('font-weight', param[5] === 'bold' ? 'bold' : 'normal')
        .appendTo(div_main);

    //スケール部分
    let div_scale = $('<div></div>')
        .css('clear', 'both')
        .css('min-width', div_width)
        .css('width', div_width)
        //.css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('background-color', that.bgColor);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i ))//i+1で1~param[2]になる
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i )//i+1で1~param[2]になる
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_scale);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_scale);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_scale);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+6]);
    }
    

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '11.5px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }
    $('<div></div>')
        .css('clear', 'both')
        //.css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_scale);

    div_main.append(div_scale);
    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};
// ---------------------------------------------------
// リッカートスケール 説明付き一体化 左よせ　吉原がExp3を改変
// ---------------------------------------------------
quesMgr.makeLikertExp4 = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let div_width = (32 * num) + "px";

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('height', '300px')
        .css('min-width', div_width)
        .css('margin-top', '10px')
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('margin-bottom', '10px');

    //テキスト部分
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .text(param[4])
        .css('font-size', '12pt')
        .css('text-align', 'left')
        .appendTo(div_main);

    //スケール部分
    let div_scale = $('<div></div>')
        .css('clear', 'both')
        .css('min-width', div_width)
        .css('width', div_width)
        //.css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('background-color', that.bgColor);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i ))//i+1で1~param[2]になる
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i )//i+1で1~param[2]になる
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_scale);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_scale);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_scale);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+5]);
    }
    

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '12px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }
    $('<div></div>')
        .css('clear', 'both')
        //.css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_scale);

    div_main.append(div_scale);
    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};
// ---------------------------------------------------
// リッカートスケール 説明付き一体化 センタリング
// ---------------------------------------------------
quesMgr.makeLikertExp3 = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let div_width = (32 * num) + "px";

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('height', '300px')
        .css('min-width', div_width)
        .css('margin-top', '10px')
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('margin-bottom', '10px');

    //テキスト部分
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .text(param[4])
        .css('font-size', '12pt')
        .css('text-align', 'center')
        .appendTo(div_main);

    //スケール部分
    let div_scale = $('<div></div>')
        .css('clear', 'both')
        .css('min-width', div_width)
        .css('width', div_width)
        .css('margin-left', 'auto')
        .css('margin-right', 'auto')
        .css('background-color', that.bgColor);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i + 1))
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i + 1)
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_scale);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_scale);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_scale);


    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+5]);
    }
   

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '12px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }
    $('<div></div>')
        .css('clear', 'both')
        //.css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_scale);

    div_main.append(div_scale);
    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};

// ---------------------------------------------------
// リッカートスケール 説明付き一体化 左詰め 座標指定
// ---------------------------------------------------
quesMgr.makeLikertExp2 = function (param) {
    let x = param[1] + 'px';
    let y = param[2] + 'px';
    let num = Number(param[3]);
    let name = param[4];
    let elecFlg = param[5];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .css('position', 'relative')
        .css('left', x)
        .css('top', y)
        .css('display', 'inline-block')
        .css('float', 'left')
        .css('height', '300px')
        .css('margin-top', '10px')
        .css('margin-left', '20px')
        .css('margin-right', '20px')
        .css('margin-bottom', '10px');

    //テキストの表示+スペーサ
    $('<div class="text_exp"></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .text(param[6])
        .css('font-size', '12pt')
        .append(
            $('<span></span>').css('margin', '20px')
        )
        .appendTo(div_main);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i + 1))
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '32px')
            .html(i + 1)
            .css('font-size', '16pt')
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_main);

    let lWidth = 360 + (32 * num) + "px";

    let divLikert = $('<div></div>')
        .css('clear', 'both')
        .css('height', '20px')
        //.css('width', lWidth)
        .css('background-color', that.bgColor)
        .append(div_main);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_main);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_main);


    // ----------- スケールの説明 -----------
    // ----------- スケールの説明 -----------
    let label;
    label = [];
    
    for (let i=0; i<num; i++){
        label.push(param[i+5]);
    }

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '12px')
            .css('padding-right', '20px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '12pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }

    $('<div></div>')
        .css('clear', 'both')
        .css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_main);

    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};

// ------------------------------------------
// リッカートスケール 説明付き一体化 左詰め
// ------------------------------------------
quesMgr.makeLikertExp = function (param) {
    let num = Number(param[1]);
    let name = param[2];
    let elecFlg = param[3];
    let idx, bgcolor;
    let that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    //全体のDIV
    let div_main = $('<div class="likert"></div>')
        .attr('id', 'scale')
        .css('display', 'inline-block')
        .css('float', 'left');

    //テキストの表示+スペーサ
    $('<div class="text_s"></div>')
        .text(param[4])
        .append(
            $('<span></span>').css('margin', '20px')
        )
        .appendTo(div_main);

    //クリッカブルな数字の生成
    let div_nums = $('<div></div>');
    for (let i = 0; i < num; i += 1) {
        $('<div></div>')
            .attr('id', "l-" + (i + 1))
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '2em')
            .html(i + 1)
            .css('cursor', 'pointer')
            .css('border', 'solid 2px transparent')
            .click(function () {
                let elms = $(this).parent().find('div');
                for (let i = 0; i < elms.length; i += 1) {
                    $(elms[i]).css('border', 'solid 2px transparent');
                }
                $(this)
                    .css('border', 'solid 2px ' + that.hlColor2)
                    .css('background-color', that.bgColor);
                that.qAnswers[that.crrPage][idx].ans = $(this).html();
            })
            .mouseover(function () {
                $(this).css('background-color', that.hlColor1);
            })
            .mouseout(function () {
                $(this).css('background-color', $(this).parent().css('background-color'));
            })
            .appendTo(div_nums);
    }
    div_nums.appendTo(div_main);

    let lWidth = 360 + (32 * num) + "px";

    let divLikert = $('<div></div>')
        .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(idx, 4))
        .css('clear', 'both')
        .css('height', '20px')
        //.css('width', lWidth)
        .css('background-color', that.bgColor)
        .append(div_main);

    // ----------- ライン -----------
    let divLine = $('<div></div>')
        .css('clear', 'both')
        //.css('width', lWidth)
        .appendTo(div_main);

    let divBars = $('<div></div>')
    for (let i = 0; i < num; i += 1) {
        for (let j = 0; j < 2; j += 1) {
            let divSC = $('<div></div>')
                .css('float', 'left')
                .css('text-align', 'center')
                .css('width', '16px')
                .css('height', '10px')
                .css('font-size', '0%')
            if ((i === 0 && j === 0) || (i === num - 1 && j === 1)) {
                divSC.css('border-bottom', 'solid 2px transparent');
            } else {
                divSC.css('border-bottom', 'solid 2px ' + that.fgColor);
            }
            if (j === 1) {
                divSC.css('border-left', 'solid 1px ' + that.fgColor);
            } else {
                divSC.css('border-right', 'solid 1px ' + that.fgColor);
            }
            divSC.css('color', that.bgColor)
                .html('_')
                .appendTo(divBars);
        }
    }
    divBars.appendTo(div_main);


    // ----------- スケールの説明 -----------
    let label;
    if (num === 7) {
        label = [
            param[5],
            param[6],
            param[7],
            param[8],
            param[9],
            param[10],
            param[11]
        ];
    } else if (num === 5) {
        label = [
            param[5],
            param[6],
            param[7],
            param[8],
            param[9]
        ];
    } else if (num === 4) {
        label = [
            param[5],
            param[6],
            param[7],
            param[8]
        ];
    } else if (num === 3) {
        label = [
            param[5],
            param[6],
            param[7]
        ];
    } else if (num === 2) {
        label = [
            param[5],
            param[6],
        ];
    }

    let div_txt = $('<div></div>')
        .css('color', that.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = that.bgColor;
        } else {
            fg_color = that.fgColor;
        }
        $('<div></div>')
            .css('float', 'left')
            .css('text-align', 'center')
            .css('width', '1em')
            .css('line-height', '1.05em')
            .css('padding-left', '12px')
            .css('padding-right', '11px')
            .css('margin-top', '4px')
            .css('letter-spacing', "-1px")
            .css('font-size', '8pt')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }

    $('<div></div>')
        .css('clear', 'both')
        .css('margin', '0 auto')
        .css('border', '0px')
        //.css('width', l_width)
        .css('height', '100px')
        .append(div_txt)
        .appendTo(div_main);

    div_main.appendTo('#qTable');

    if (this.pageLikertHidden === true) {
        div_main.css('display', 'none');
    }
};

// ----------------------------------------
// リッカートスケール その2 左右の文字なし
// ----------------------------------------
quesMgr.makeLikert2 = function (param) {
    let num = Number(param[1]),
        name = param[2],
        elecFlg = param[3],
        idx, bgcolor,
        that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let likertElm = $('.likert-template')
        .clone()
        .attr('class', 'likert')
        .attr('id', "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .appendTo('#qTable');

    // ----------- 項目名 -----------
    $(likertElm).children('.likert-left-part').remove();
    $(likertElm).children('.likert-right-part').remove();

    // ----------- 数字 -----------
    for (let i = 0; i < num; i += 1) {
        $(likertElm)
            .children('.likert-number-part')
            .append($('<div class="likert-num">' + (i + 1) + '</div>')
                .click(function () {
                    let elms = $(this).parent().find('div');
                    for (let i = 0; i < elms.length; i += 1) {
                        $(elms[i]).css('border', 'solid 2px transparent');
                    }
                    $(this)
                        .css('border', 'solid 2px ' + that.hlColor2)
                        .css('background-color', that.bgColor);
                    that.qAnswers[that.crrPage][idx].ans = $(this).html();
                })
                .mouseover(function () {
                    $(this).css('background-color', that.hlColor1);
                })
                .mouseout(function () {
                    $(this).css('background-color', $(this).parent().css('background-color'));
                })
            );
    }

    // ----------- ライン -----------
    $(likertElm).children('.likert-scale-part')
        .css('margin-left', '0px')
        .append('<div class="scale-part-num-r"></div>');
    for (let i = 0; i < num - 1; i += 1) {
        $(likertElm).children('.likert-scale-part')
            .append('<div class="scale-part-num-b"></div>');
        $(likertElm).children('.likert-scale-part')
            .append('<div class="scale-part-num-br"></div>');
    }
    $(likertElm).children('.likert-scale-part')
        .append('<div class="scale-part-num-n"></div>');

};

// ----------------------------------------------------------
// リッカートスケールの作成 (左右に項目名配置可，ほぼ中央配置)
// SD法と呼んだ方がよいかも
// ----------------------------------------------------------
quesMgr.makeLikert = function (param) {
    let num = Number(param[1]),
        name = param[2],
        leftText = param[3] || '',
        rightText = param[4] || '',
        elecFlg = param[5],
        idx, bgcolor,
        that = this;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (elecFlg.match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    let likertElm = $('.likert-template')
        .clone()
        .attr('class', 'likert')
        .attr('id', "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4))
        .appendTo('#qTable');

    // ----------- 項目名 -----------
    $(likertElm).children('.likert-left-part')
        .text(param[3]);
    $(likertElm).children('.likert-right-part')
        .text(param[4]);

    // ----------- 数字 -----------
    for (let i = 0; i < num; i += 1) {
        $(likertElm)
            .children('.likert-number-part')
            .append($('<div class="likert-num">' + (i + 1) + '</div>')
                .click(function () {
                    let elms = $(this).parent().find('div');
                    for (let i = 0; i < elms.length; i += 1) {
                        $(elms[i]).css('border', 'solid 2px transparent');
                    }
                    $(this).css('border', 'solid 2px ' + that.hlColor2);
                    that.qAnswers[that.crrPage][idx].ans = $(this).html();
                })
                .mouseover(function () {
                    $(this).css('background-color', that.hlColor1);
                })
                .mouseout(function () {
                    $(this).css('background-color', $(this).parent().css('background-color'));
                })
            );
    }

    // ----------- ライン -----------
    $(likertElm).children('.likert-scale-part')
        .append('<div class="scale-part-num-r"></div>');
    for (let i = 0; i < num - 1; i += 1) {
        $(likertElm).children('.likert-scale-part')
            .append('<div class="scale-part-num-b"></div>');
        $(likertElm).children('.likert-scale-part')
            .append('<div class="scale-part-num-br"></div>');
    }
    $(likertElm).children('.likert-scale-part')
        .append('<div class="scale-part-num-n"></div>');

};

// -------------------------------------
// ページボタンの作成
// -------------------------------------
quesMgr.makePageButtons = function () {
    if (this.qPages.length === 1) { return; }

    //次のページボタン
    let that = this;
    if (this.crrPage < (this.qPages.length - 2)) {
        $('<div class="pageButtonArea"></div>')
            .append($('<button id="pageButton" type="button" class="btn btn-primary btn-lg btn-block"></button>')
                .text(this.systemMsg[this.lang].buttonForward)
                .bind('click', function () {
                    if (that.qAnswers[that.crrPage].length > 0 || that.checkImgmapFile) {
                        that.checkAnswers();
                    } else {
                        that.crrPage += 1;
                        that.parseQuestions(that.qPages[that.crrPage]);
                    }
                })
            ).appendTo('#qTable');

        if (this.pageButtonHidden === true) {
            //nxtButton.style.visibility = "hidden";
            $('#pageButton').css('display', 'none');
        }
    }
    //終了ボタン
    else if (this.crrPage === (this.qPages.length - 2)) {
        $('<div class="sendButtonArea"></div>')
            .append($('<button id="sendButton" type="button" class="btn btn-primary btn-lg btn-block"></button>')
                .text(this.systemMsg[this.lang].sendButtonMsg)
                .bind('click', function () {
                    that.checkAnswers();
                })
            ).appendTo('#qTable');
        this.isFinished = true;
    }
};

// -------------------------------------
// ページタイトルの生成
// -------------------------------------
quesMgr.makeTitle = function (text) {
    $('<div class="textTitle"></div>')
        .html(text)
        .appendTo('#qTable');
};

// -------------------------------------
// 説明文章の生成
// -------------------------------------
quesMgr.makeText = function (text) {
    $('<div class="text"></div>')
        .html(text)
        .appendTo('#qTable');
};

// -------------------------------------
// 説明文章の生成
// -------------------------------------
quesMgr.makeTextSpan = function (text) {
    $('<span class="text-span"></span>')
        .html(text)
        .appendTo('#qTable');
};

// -------------------------------------
// 説明文章の生成(その2・太文字)
// -------------------------------------
quesMgr.makeText2bf = function (param) {
    let dval = param[2] || "block"

    //alert(param + "," + dval);

    $('<div class="text_s_bf"></div>')
        .html(param[1])
        .appendTo('#qTable');
};

// -------------------------------------
// 説明文章の生成(その2)
// -------------------------------------
quesMgr.makeText2 = function (param) {
    let dval = param[2] || "block"
    if(param[2]=="randFile"){
        insertText = quesMgr.repFileMgr.nowText;
        text=param[1];
        text=text.replace("%randtext%",insertText);
            
    }else{
        text=param[1];
    }


    //alert(param + "," + dval);

    $('<div class="text_s"></div>')
        .html(text)
        //.css('margin-bottom', 12)
        ///.css('margin-left', 12)
        .appendTo('#qTable');
};

// ----------------------------------------------
// リンクの生成
// ----------------------------------------------
quesMgr.makeLink = function (param) {
    $('<a></a>')
        .text(param[1])
        .attr('href', param[2])
        .html(param[1])
        .css('display', 'block')
        .css('text-align', 'left')
        .css('vertical-align', 'top')
        .css('clear', 'left')
        .appendTo('#qTable');
};

// ----------------------------------------------
// 説明文章の生成(その3) 親id指定かつ相対座標指定
// ----------------------------------------------
quesMgr.makeText3 = function (param) {
    $('<div class="text_s"></div>')
        .css('position', 'absolute')
        .css('left', param[2] + 'px')
        .css('top', param[3] + 'px')
        .text(param[4])
        .appendTo('#' + param[1].trim());
};

// -------------------------------------
// スケールの説明
// -------------------------------------
quesMgr.makeScaleExp = function (param) {
    let num = Number(param[1]);
    let l_width = 360 + (32 * num) + "px";

    let label;
    if (num === 7) {
        label = [
            param[2],
            param[3],
            param[4],
            param[5],
            param[6],
            param[7],
            param[8]
        ];
    } else if (num === 5) {
        label = [
            param[2],
            param[3],
            param[4],
            param[5],
            param[6]
        ];
    } else if (num === 4) {
        label = [
            param[2],
            param[3],
            param[4],
            param[5]
        ];
    } else if (num === 3) {
        label = [
            param[2],
            param[3],
            param[4]
        ];
    }

    let div_txt = $('<div></div>')
        .css('color', this.fgColor);

    for (let i = 0; i < num; i += 1) {
        let lbl = label[i] || '_'
        let fg_color;
        if (lbl === '_') {
            fg_color = this.bgColor;
        } else {
            fg_color = this.fgColor;
        }
        $('<div class="scale-exp-label"></div>')
            .css('color', fg_color)
            .html(lbl)
            .appendTo(div_txt);
    }

    $('<div class="scale-exp"></div>')
        .append(div_txt)
        .appendTo('#qTable');
};

// -------------------------------------
// データの保存
// -------------------------------------
quesMgr.saveData = function () {
    let i, j, k, idx, sidx, eidx, files, p, max;
    let sp = 0;
    let numRepPages = 0;
    let sIndex = [];
    let jAns = "";

    //PHPでファイル名を供給した場合のファイル名と順序の情報を追加
    jAns += this.providedFnames;

    //繰り返しページが存在していた場合
    if (this.repFileMgr.showStartPage) {
        sp = this.repFileMgr.showStartPage();
        numRepPages = this.repFileMgr.showNumRepPages();
        sIndex = this.repFileMgr.showSortedIndex();
    }

    // (1) 繰り返しページ以外 (前半)
    for (i = 0; i < sp; i += 1) {
        for (j = 0; j < this.qAnswers[i].length; j += 1) {
            jAns += this.qAnswers[i][j].ans
            jAns += ",";
            //console.log(i + ": " + jAns);
        }
    }

    // (2) 繰り返しページ
    //console.log("numRepPages: " + numRepPages)
    //console.log("sIndex.length: " + sIndex.length)
    if (sIndex.length > 0) {
        for (i = 0; i < sIndex.length; i += 1) {
            for (p = sp; p < (sp + numRepPages); p += 1) {
                idx = sIndex[i] * numRepPages + p;
                for (j = 0; j < this.qAnswers[idx].length; j += 1) {
                    jAns += this.qAnswers[idx][j].ans
                    jAns += ",";
                    //console.log(jAns);
                }
            }
        }
    }

    // (3) 繰り返しページ以外 (後半)
    for (i = (sp + sIndex.length * numRepPages); i < this.qAnswers.length; i += 1) {
        for (j = 0; j < this.qAnswers[i].length; j += 1) {
            jAns += this.qAnswers[i][j].ans
            jAns += ",";
        }
    }

    //繰り返しページが存在していた場合
    if (this.repFileMgr.showStartPage) {
        // (4) ファイル提示順序
        files = this.repFileMgr.showAllFnames();
        jAns += "provided: ";
        for (i = 0; i < files.length; i += 1) {
            jAns += files[i];
            jAns += " ";
        }
        jAns += ",";

        // (5) ファイル保存順序
        sortfiles = this.repFileMgr.showSortedFnames();
        jAns += "saved: ";
        for (i = 0; i < sortfiles.length; i += 1) {
            jAns += sortfiles[i];
            jAns += " ";
        }
    }

    //alert(jAns);

    let cDate = new Date();
    let tYear = cDate.getFullYear();
    let tMonth = this.nf(cDate.getMonth() + 1, 2);
    let tDate = this.nf(cDate.getDate(), 2);
    let tHour = this.nf(cDate.getHours(), 2);
    let tMin = this.nf(cDate.getMinutes(), 2);
    let tSec = this.nf(cDate.getSeconds(), 2);
    let endDate = tYear + "-" + tMonth + "-" + tDate + "_" +
        tHour + "-" + tMin + "-" + tSec;

    this.writeCookie("status", "finish", 60);

    let ret = $.ajax({
        //url: proxyURL + "addComment.php?",
        //url: "saveQuesData.php?",
        type: "POST",
        url: "php/saveQuesDataPost.php",
        data: {
            gid: this.gid,
            lang: this.lang,
            startDate: this.startDate,
            endDate: endDate,
            val: jAns,
        },
        async: false,
        complete: function () {
        }
    }).responseText;

    // ---------------------------------------------------------
    // ここからデバッグ用のログを追加
    // ---------------------------------------------------------
    
    //console.log("window.timeLogData の中身:", window.timeLogData);

    //iframeの中からログデータを探し出す処理
    //ブラウザの共通領域からデータを取り出す
    let targetLogData = localStorage.getItem('chatTimeLog');
    // データの中身をコンソールで確認（デバッグ用）
    console.log("LocalStorageの中身:", targetLogData);

    // ---------------------------------------------------------
    // 2. ★追加: タイムログの保存 (saveTestTimelogPost.php へ)
    // ---------------------------------------------------------
    if (targetLogData && targetLogData.length > 0) {
        console.log("データが存在するため、saveTestTimelogPost.php へ送信を試みます");

        $.ajax({
            type: "POST",
            url: "php/saveTestTimelogPost.php", // ★新しく作成するPHP
            data: {
                gid: this.gid,       // ファイル名用のID
                lang: this.lang,
                startDate: this.startDate,
                endDate: endDate,
                timeLog: targetLogData // 記録したログデータ
            },
            async: false, // これも確実に保存するために同期通信にします
            success: function(res) {
                //console.log("Time log saved: " + response);
                console.log("成功: " + res);
                // 保存できたらゴミを残さないように消去
                localStorage.removeItem('chatTimeLog');
            },
            error: function(err) {
                //console.error("Time log save failed.");
                console.log("失敗: 通信エラー発生");
            }
        }).responseText;
    }else {
        console.error("【原因判明】 window.timeLogData が空または未定義のため、送信がスキップされました。");
    }

    // =========================================================
    // 3. ★追加: スクロールログの保存 (saveScrollLogPost.php へ)
    // =========================================================
    let targetScrollData = localStorage.getItem('chatScrollLog');

    // ★追加: データが null (未操作) の場合は、空文字にしておく
    if (!targetScrollData) {
        targetScrollData = "";
    }

        console.log("データが存在するため、saveScrollLogPost.php へ送信を試みます");

        $.ajax({
            type: "POST",
            // スクロールログ保存用に新しいPHPファイルを作成してください
            url: "php/saveScrollLogPost.php", 
            data: {
                gid: this.gid,
                lang: this.lang,
                startDate: this.startDate,
                endDate: endDate,
                // PHP側では $_POST['scrollLog'] で受け取ります
                scrollLog: targetScrollData 
            },
            async: false, // 確実に保存するために同期通信
            success: function(res) {
                console.log("ScrollLog 成功: " + res);
                // 保存できたら削除
                localStorage.removeItem('chatScrollLog');
            },
            error: function(err) {
                console.log("ScrollLog 失敗: 通信エラー発生");
            }
        }).responseText;
    

};

// -------------------------------------
// 回答データのチェック (グループID判定版)
// -------------------------------------
quesMgr.checkAnswers = function () {
    let elm,
        flgEmptyReqAnswer = false,
        flgEmptyElecAnswer = false,
        that = this,
        msg = "",
        isEmpty;

    // --- イメージマップのチェック (変更なし) ---
    if (this.checkImgmapFile) {
        isEmpty = false;
        for (let i = 0; i < this.checkImgmapTypes.length; i += 1) {
            if (this.isEmptyMapData(this.gid, this.checkImgmapTypes[i])) {
                isEmpty = true;
                break;
            }
        }
        if (isEmpty) {
            $('#noinput-title').text(this.systemMsg[this.lang].noMapDataTitle);
            $('#noinput-body').text(this.systemMsg[this.lang].noMapDataBody);
            $('#noinput-back').text(this.systemMsg[this.lang].noInputBack)
            $('#noinput').modal('show');
            return;
        }
    }

    // --- nameCheck用のグループ集計マップを作成 ---
    // どのグループIDで、いくつ「1(選択済み)」があるかを数える
    let groupCounts = {};
    for (let i = 0; i < this.qAnswers[this.crrPage].length; i++) {
        let item = this.qAnswers[this.crrPage][i];
        if (item.groupID) { // nameCheckの項目なら
            if (!groupCounts[item.groupID]) {
                groupCounts[item.groupID] = 0;
            }
            if (item.ans === "1") {
                groupCounts[item.groupID]++;
            }
        }
    }

    // --- 回答チェックループ ---
    for (let i = 0; i < this.qAnswers[this.crrPage].length; i += 1) {
        let item = this.qAnswers[this.crrPage][i];
        elm = $('#P' + that.nf(that.crrPage, 2) + "_Q" + that.nf(i, 4));
        
        // 色のリセット
        if (item.groupID) {
            // nameCheck: 未選択なら枠線をグレーに戻す
            if (item.ans === "0") {
                elm.css('border-color', '#ccc').css('background-color', '#fff');
            }
        } else {
            elm.css('background-color', that.bgColor);
        }

        // --- パターンA: 空文字チェック (通常) ---
        if (item.ans === "") {
            if (!item.elec) {
                flgEmptyReqAnswer = true;
                elm.css('background-color', that.hlColor3);
                if (elm.attr('class') === 'checkbox')
                    elm.css('border-bottom', 'solid 3px rgba(251,152,11,1.0)');
            } else if (item.elec) {
                flgEmptyElecAnswer = true;
                elm.css('background-color', that.hlColor4);
                if (elm.attr('class') === 'checkbox')
                    elm.css('border-bottom', 'solid 3px rgba(152,251,11,1.0)');
            }
        }
        
        // --- パターンB: nameCheck (groupID持ち) のチェック ---
        else if (item.groupID) {
            // 必須回答 かつ グループ内の選択数が0の場合 -> エラー
            if (!item.elec && groupCounts[item.groupID] === 0) {
                flgEmptyReqAnswer = true;
                // エラー強調表示 (枠線を赤っぽく)
                elm.css('border-color', 'rgba(251,152,11,1.0)');
                elm.css('background-color', '#fff0e0');
            }
        }
    }

    // --- 結果による分岐 ---
    if (flgEmptyReqAnswer) {
        $('#noinput-title').text(this.systemMsg[this.lang].noInputTitle);
        $('#noinput-body').text(this.systemMsg[this.lang].noInputBody);
        $('#noinput-back').text(this.systemMsg[this.lang].noInputBack)
        $('#noinput').modal('show');
    } else {
        msg = this.systemMsg[this.lang].confirmation;

        if (this.flgNasaTLX) {
            msg += '<span class="ui-icon ui-icon-alert" style="float:left; margin:6px 6px 0px 0;"></span>';
            msg += "動かしていないスライダがあります<br>";
        }
        if (this.flgNoMovedSortableText) {
            msg += '<span class="ui-icon ui-icon-alert" style="float:left; margin:6px 6px 0px 0;"></span>';
            msg += "項目が移動されていません<br>";
        }
        if (flgEmptyElecAnswer) {
            msg += '<span class="ui-icon ui-icon-alert" style="float:left; margin:6px 6px 0px 0;"></span>';
            msg += this.systemMsg[this.lang].noInputElective;
        }

        $('#confirm-body').html(msg);
        $('#confirm-title').text(this.systemMsg[this.lang].confirmTitle);
        $('#confirm-back').text(this.systemMsg[this.lang].confirmBack);
        $('#confirm-forward')
            .text(this.systemMsg[this.lang].confirmForward)
            .unbind().bind('click', function () {
                if (that.flgNasaTLX && flgEmptyReqAnswer) {
                    for (let i = 0; i < that.qAnswers[that.crrPage].length; i += 1) {
                        if (that.qAnswers[that.crrPage][i].ans === "") {
                            that.qAnswers[that.crrPage][i].ans = 50;
                        }
                    }
                    that.flgNasaTLX = false;
                }
                if (that.crrPage === (that.qPages.length - 2)) {
                    that.saveData();
                    window.onbeforeunload = null;
                    if (that.asm !== "") {
                        location.replace("https://enq4.dstyleweb.com/orca/EQ61284459?id=" + that.asm);
                    } else if (that.asme1 !== "" && that.asme2 !== "") {
                        location.replace("https://dkr1.ssisurveys.com/projects/end?rst=1&basic=67297&psid=" + that.asme1 + "&pid=" + that.asme2);
                    }
                }
                that.flgNoMovedSortableText = false;
                that.crrPage += 1;
                that.parseQuestions(that.qPages[that.crrPage]);
                $('#confirm-dialog').modal('hide');
            });
        $('#confirm-dialog').modal('show');
    }
};

// -------------------------------------
// ウィンドウサイズの取得
// -------------------------------------
quesMgr.getWinSize = function () {
    let ua = navigator.userAgent;
    let w, h;
    let nHit = ua.indexOf("MSIE");
    let bIE = (nHit >= 0);
    let bVer6 = (bIE && ua.substr(nHit + 5, 1) === "6");
    let bStd = (document.compatMode && document.compatMode === "CSS1Compat");

    // 標準モードかどうか
    if (bIE) {
        if (bVer6 && bStd) {
            w = document.documentElement.clientWidth;
            h = document.documentElement.clientHeight;
        } else {
            w = document.body.clientWidth;
            h = document.body.clientHeight;
        }
    } else {
        w = window.innerWidth;
        h = window.innerHeight;
    }

    return { width: w, height: h };
};

// -------------------------------------
// クッキーオンオフチェック
// -------------------------------------
quesMgr.checkCookie = function () {
    let key = "test";
    this.writeCookie(key, 1, 1);
    if (this.readCookie(key) == "") {
        return false;
    } else {
        this.clearCookie(key);
        return true;
    }
};

// -------------------------------------
// ラジオボタンの作成
// -------------------------------------
quesMgr.makeRadioButton = function (param) {
    let idx, values, max, i,
        that = this, idval;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    values = param[1].replace(/\"/g, "");
    values = values.replace(/^ /, "").split(/ +/);

    idval = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);

    $('<div></div>')
        .attr('id', idval)
        .css('position', 'relative')
        .css('float', 'left')
        .css('display', param[2])
        .appendTo('#qTable');

    max = values.length;
    for (i = 0; i < max; i += 1) {
        $("<div></div>")
            .append(
                $("<input type='radio'></input>")
                    .attr('name', 'nm' + idval)
                    //.attr('type', 'radio')
                    .attr('value', values[i])
                    .unbind()
                    .bind('click', function () {
                        //alert($(this).attr('value'));
                        that.qAnswers[that.crrPage][idx].ans = $(this).attr('value');
                    }))
            .css('position', 'absolute')
            .css('width', '200px')
            .css('left', (i * 200 + 50) + "px")
            .append(values[i])
            //.appendTo('#qTable');
            .appendTo('#' + idval);
    }
};


// -------------------------------------
// Unity(WebGl)の埋め込み
//第１引数でBuildファイルの場所を指定
//第２引数でpreloadか否かを指定
//preloadの場合、qTableに挿入せずに、Bodyに挿入する
// -------------------------------------
quesMgr.makeUnityFrame = function (param) {
    //この時点でクッキーにgidが登録されている前提の挙動のため注意
    //.csv -> writeInfoCookie

    //現在のdiv構造にpreコンテナが存在するなら
    // //class名で検索
    // if($('.'+param[1]).length){


    // if ($('#unity-container').length) {
    if ($('.' + param[1]).length) {
        //qTableにunity-containerを移動
        $('.' + param[1]).appendTo('#qTable');
        //display:noneを解除
        $('.' + param[1]).css('display', 'block');
    }
    //ないなら作る
    else {
        // Create unity-container
        var unityContainer = $("<div></div>")
            .attr('id', 'unity-container')
            .addClass(param[1]);

        // Create unity-canvas
        var unityCanvas = $("<canvas></canvas>")
            .attr('id', 'unity-canvas-'+param[1])


        // Create unity-loading-bar
        var unityLoadingBar = $("<div></div>")
            .attr('id', 'unity-loading-bar');

        // Create unity-logo
        var unityLogo = $("<div></div>")
            .attr('id', 'unity-logo');

        // Create unity-progress-bar-empty
        var unityProgressBarEmpty = $("<div></div>")
            .attr('id', 'unity-progress-bar-empty');

        // Create unity-progress-bar-full
        var unityProgressBarFull = $("<div></div>")
            .attr('id', 'unity-progress-bar-full');

        unityProgressBarEmpty.append(unityProgressBarFull);

        unityLoadingBar.append(unityLogo, unityProgressBarEmpty);

        // Create unity-warning
        var unityWarning = $("<div></div>")
            .attr('id', 'unity-warning');

        // // Create unity-footer
        // var unityFooter = $("<div></div>")
        // .attr('id', 'unity-footer');

        // // Create unity-webgl-logo
        // var unityWebGLLogo = $("<div></div>")
        // .attr('id', 'unity-webgl-logo');

        // // Create unity-fullscreen-button
        // var unityFullscreenButton = $("<div></div>")
        // .attr('id', 'unity-fullscreen-button');

        // // Create unity-build-title
        // var unityBuildTitle = $("<div></div>")
        // .attr('id', 'unity-build-title')
        // .text('object test');

        //unityFooter.append(unityWebGLLogo, unityFullscreenButton, unityBuildTitle);

        // Append all elements to unity-container
        unityContainer.append(unityCanvas, unityLoadingBar, unityWarning);

        // Append unity-container to a specific div
        
        
        if (param[2] === "preload") {
            //UnityContainerを非表示に
            unityContainer.css('display', 'none');
            //bodyに挿入
            unityContainer.appendTo('body');
        }
        else{
            unityContainer.appendTo('#qTable');
        }


        //ここからUnity側の処理
        var container = document.querySelector("#unity-container");
        var canvas = document.querySelector("#unity-canvas-"+param[1]);
        var loadingBar = document.querySelector("#unity-loading-bar");
        var progressBarFull = document.querySelector("#unity-progress-bar-full");
        var fullscreenButton = document.querySelector("#unity-fullscreen-button");
        var warningBanner = document.querySelector("#unity-warning");

        // Shows a temporary message banner/ribbon for a few seconds, or
        // a permanent error message on top of the canvas if type=='error'.
        // If type=='warning', a yellow highlight color is used.
        // Modify or remove this function to customize the visually presented
        // way that non-critical warnings and error messages are presented to the
        // user.
        function unityShowBanner(msg, type) {
            function updateBannerVisibility() {
                warningBanner.style.display = warningBanner.children.length ? 'block' : 'none';
            }
            var div = document.createElement('div');
            div.innerHTML = msg;
            warningBanner.appendChild(div);
            if (type == 'error') div.style = 'background: red; padding: 10px;';
            else {
                if (type == 'warning') div.style = 'background: yellow; padding: 10px;';
                setTimeout(function () {
                    warningBanner.removeChild(div);
                    updateBannerVisibility();
                }, 5000);
            }
            updateBannerVisibility();
        }

        var buildUrl = param[1]+"/Build";
        var loaderUrl = buildUrl + "/"+param[1]+".loader.js";
        var config = {
            dataUrl: buildUrl + "/"+param[1]+".data.unityweb",
            frameworkUrl: buildUrl +"/"+param[1]+".framework.js.unityweb",
            codeUrl: buildUrl + "/"+param[1]+".wasm.unityweb",
            streamingAssetsUrl: "StreamingAssets",
            companyName: "DefaultCompany",
            productName: "object test",
            productVersion: "0.1",
            showBanner: unityShowBanner,
        };

        // By default Unity keeps WebGL canvas render target size matched with
        // the DOM size of the canvas element (scaled by window.devicePixelRatio)
        // Set this to false if you want to decouple this synchronization from
        // happening inside the engine, and you would instead like to size up
        // the canvas DOM size and WebGL render target sizes yourself.
        // config.matchWebGLToCanvasSize = false;

        if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
            // Mobile device style: fill the whole browser client area with the game canvas:

            var meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, height=device-height, initial-scale=1.0, user-scalable=no, shrink-to-fit=yes';
            document.getElementsByTagName('head')[0].appendChild(meta);
            container.className = "unity-mobile";
            canvas.className = "unity-mobile";

            // To lower canvas resolution on mobile devices to gain some
            // performance, uncomment the following line:
            // config.devicePixelRatio = 1;

            unityShowBanner('WebGL builds are not supported on mobile devices.');
        } else {
            // Desktop style: Render the game canvas in a window that can be maximized to fullscreen:

            //canvas.style.width = "960px";
            //canvas.style.height = "600px";
        }

        loadingBar.style.display = "block";

        var script = document.createElement("script");
        script.src = loaderUrl;
        script.onload = () => {
            createUnityInstance(canvas, config, (progress) => {
                progressBarFull.style.width = 100 * progress + "%";
            }).then((unityInstance) => {
                loadingBar.style.display = "none";
                // fullscreenButton.onclick = () => {
                //     unityInstance.SetFullscreen(1);
                // };
            }).catch((message) => {
                alert(message);
            });
        };
        document.body.appendChild(script);
    }
};

// -------------------------------------
// 
// -------------------------------------
quesMgr.makeWaitTrigger = function (param) {
    this.pageButtonHidden = true;
};


// -------------------------------------
// インラインフレームの作成
// gidと指定の値をハッシュパラメータとして渡す
// -------------------------------------
quesMgr.makeIframeGid = function (param) {
    let that = this, fileName, customId = param[2] || "",
        urlParam = param[3] || "";

    if (param[1].match(/randFile|fixFile/)) {

        fileName = this.repFileMgr.getFname();
    } else if (param[1].match(/\.php/)) { // phpファイルだった場合
        fileName = $.ajax({
            url: param[1],
            async: false
        }).responseText;
        this.providedFnames += "iframegid:" + fileName + ",";
        console.log(fileName);
    } else {
        fileName = param[1].replace(/^\s+/, "");
    }

    if (customId !== "") {
        customId = customId.replace(/^\s+/, "");
        customId = "-" + customId;
    }

    this.pageButtonHidden = true;

    let div = $("<div></div>")
        .attr('align', 'center')
        .appendTo('#qTable');

    $("<iframe></iframe>")
        .attr('align', 'middle')
        .attr('src', "./" + fileName + urlParam + "#" + this.gid + customId)
        .attr('scrolling', 'no')
        .appendTo(div)

    div.appendTo('#qTable');

    // ★追加: チャットシステムからの終了メッセージを受け取るリスナー
    window.addEventListener('message', function (event) {
        // セキュリティのため、本来はevent.originを確認することが望ましいですが、
        // 同一ドメインやローカル環境での動作を優先してここではチェックを省略、または必要に応じて追加してください。
        
        if (event.data && event.data.type === 'show_completion_code') {
            // ボタンを表示する
            $('#pageButton').show();
            // quesMgrの内部状態も更新しておく（もしページ遷移判定に使っている場合）
            that.pageButtonHidden = false;
            console.log("Chat completed. Next button shown.");
        }
    });
};

// -------------------------------------
// インラインフレームの作成
// -------------------------------------
quesMgr.makeIframe = function (param) {
    let that = this,
        fileName;

    if (param[1].match(/randFile|fixFile/)) {
        console.log(this.repFileMgr);
        fileName = this.repFileMgr.getFname();
    } else if (param[1].match(/\.php/)) { // phpファイルだった場合
        fileName = $.ajax({
            url: param[1],
            async: false
        }).responseText;
        this.providedFnames += "iframe:" + fileName + ",";
        console.log(fileName);
    } else {
        fileName = param[1].replace(/^\s+/, "");
    }

    $("<div id='ifr' align='center'></div>")
        .appendTo('#qTable');

    $("<iframe></iframe>")
        .attr('align', 'middle')
        .attr('src', "./" + fileName + "/index.html")
        .attr('scrolling', 'no')
        .appendTo('#ifr');
};

// -------------------------------------
// ソータブルテキストの作成
// -------------------------------------
quesMgr.makeSortableText = function (param) {
    let idx, values, max, i,
        that = this, idval, defAns;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (param[3].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    values = param[1].replace(/\"/g, "");
    values = values.replace(/^ /, "").split(/ +/);

    idval = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);

    $("<div></div>")
        .appendTo('#qTable')
        .css('height', '130px')
        .css('width', '400px')
        .css('padding', '10px')
        .css('margin-left', '200px')
        .css('margin-right', 'auto')
        .css('background', '#F0F0F0')
        .append($("<ul></ul>")
            .attr('id', 'sortRank')
            .css('list-style-type', 'none')
            .css('margin', '0')
            .css('width', '65px')
            .css('float', 'left'))
        .append($("<ul></ul>")
            .attr('id', idval)
            .css('margin', '0')
            .css('list-style-type', 'none')
            .css('width', '250px')
            .css('float', 'left')
            .css('padding', 0));

    max = values.length;
    for (i = 0; i < max; i += 1) {
        $("<li align='left'></li>")
            .attr('class', 'ui-state-default')
            .css('margin', '0px 3px 3px 3px')
            .css('padding', '10px')
            .css('height', '20px')
            .css('font-size', '20px')
            //.append("<span>" + (i+1) + "位</span>")
            .append((i + 1) + "位")
            .appendTo('#sortRank');

        $("<li align='left'></li>")
            .attr('class', 'ui-state-default')
            .css('margin', '0 3px 3px 3px')
            .css('padding', '10px')
            .css('height', '20px')
            .css('font-size', '20px')
            .css('cursor', 'pointer')
            //	.append($("<span></span>")
            //		.attr('class', 'ui-icon ui-icon-arrowthick-2-n-s')
            //		.css('margin-left', '-1.3em')
            //	       )
            .append("<span>" + values[i] + "</span>")
            .appendTo('#' + idval);
    }

    $("#" + idval).sortable({
        placeholder: "ui-state-highlight",
        update: function (event, ui) {
            let itemOrder = param[2] + " ";
            //console.log(jQuery.makeArray($("li span:last-child")));
            jQuery.each($("li span:last-child"), function () {
                itemOrder += $(this).html() + " ";
            });
            //console.log('"' + itemOrder + '"');
            that.qAnswers[that.crrPage][idx].ans = itemOrder;
            that.flgNoMovedSortableText = false;
            //console.log($("li span:last-child"));
        }
    });
    $("#" + idval).disableSelection();

    //デフォルトの回答を設定
    this.flgNoMovedSortableText = true;
    defAns = param[2] + " ";
    jQuery.each($("li span:last-child"), function () {
        defAns += $(this).html() + " ";
    });
    this.qAnswers[that.crrPage][idx].ans = defAns;
};

// -------------------------------------
// ビデオファイルの埋め込み
// -------------------------------------
quesMgr.embedVideo = function (id, fileName) {
    let num = 0;

    jwplayer(id).setup({
        flashplayer: "conf/player.swf",
        file: "conf/video/" + fileName,
        volume: 100,
        width: 200,
        height: 150,
        'controlbar.position': "none",
        //'controlbar.position': "bottom",
        'bufferlength': 40,
        icons: false,
        events: {
            onReady: function () {
                let that = this;
                $('#btn-' + id)
                    .unbind()
                    .bind('click', function () { that.play(); })

                $('#btn-' + id)
                    .html('<img src="css/images/play.png">');

                $('#msg-' + id)
                    .html("(再生回数：" + num + "回)");
            },
            onComplete: function () {
                //alert("ビデオが終了しました．");
                //$('#pageButton').css('visibility', 'visible');
                //$('#msg-'+id).html("ビデオが終了しました．【もう一度再生する】");
                $('#msg-' + id).html("(再生回数:" + num + "回)");

                $('#btn-' + id)
                    .html('<img src="css/images/play.png">');
            },
            /*
                   onBufferFull: function() {
                   $('#pageButton').css('visibility', 'visible');
                   },
                   */
            /*
                   onBuffer: function() {
        //alert('しばらくお待ちください．');
                $('#vmsg').html("しばらくお待ちください．");
                },
                */
            onBufferChange: function () {
                $('#msg-' + id).html("しばらくお待ちください (" +
                    Math.floor(this.getBuffer()) + "％)");
            },
            onPlay: function () {
                num += 1;
                //this.getState();
                $('#msg-' + id).html("再生中");
                $('#btn-' + id)
                    .html('<img src="css/images/pause.png">');
            },
            onPause: function () {
                $('#msg-' + id).html("一時停止中");
                $('#btn-' + id)
                    .html('<img src="css/images/play.png">');
                //$('#vmsg').html("クリックして再生してください．");
                //停止しないようにする
                //this.play();
            }
        }
    });
};

// -------------------------------------
// クッキーへの情報書き込み
// -------------------------------------
quesMgr.writeInfoCookie = function () {
    //gidの書き込み
    this.writeCookie("gid", this.gid, 60);

    //開始時刻のクッキー書き込み
    this.writeCookie("startDate", this.startDate, 60);
};

// -------------------------------------
// ランダムな文字列生成
// -------------------------------------
quesMgr.randobet = function (n, b) {
    b = b || '';
    let a = 'abcdefghijklmnopqrstuvwxyz'
        + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        + '0123456789'
        + b;
    a = a.split('');
    let s = '';
    for (let i = 0; i < n; i++) {
        s += a[Math.floor(Math.random() * a.length)];
    }
    return s;
};

// -------------------------------------
// 開始日時
// -------------------------------------
quesMgr.makeStartDate = function () {
    let cDate = new Date();
    let tYear = cDate.getFullYear();
    let tMonth = this.nf(cDate.getMonth() + 1, 2);
    let tDate = this.nf(cDate.getDate(), 2);
    let tHour = this.nf(cDate.getHours(), 2);
    let tMin = this.nf(cDate.getMinutes(), 2);
    let tSec = this.nf(cDate.getSeconds(), 2);
    let startDate = tYear + "-" + tMonth + "-" + tDate + "_" +
        tHour + "-" + tMin + "-" + tSec;

    return startDate;
};

// -------------------------------------
// ソータブルビデオの作成
// -------------------------------------
quesMgr.makeSortableVideo = function (param) {
    let idx, values, max, i, j, x, y, x0, y0, mod,
        that = this, idval, defAns,
        cols = 3, elm,
        pv, ph, w, hh, hv, my, f, h0, w0, mr, taw,
        r1, r2, tmp;

    this.qAnswers[this.crrPage].push("");
    idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };

    if (param[3].match("elective")) {
        this.qAnswers[this.crrPage][idx].elec = true;
    }

    values = param[1].replace(/\"/g, "");
    values = values.replace(/^ /, "").split(/ +/);

    idval = "P" + this.nf(this.crrPage, 2) + "_Q" + this.nf(idx, 4);

    $("<div></div>")
        .attr('id', 'sortableFrame')
        .appendTo('#qTable')
        .css('width', '700px')
        .css('height', '800px')
        .css('padding', '10px')
        .css('margin-left', '80px')
        .css('margin-right', 'auto')
        .append($("<ul></ul>")
            .attr('id', idval)
            .addClass('sortable-item')
            .css('margin', '0')
            .css('list-style-type', 'none')
            .css('width', '700px')
            .css('padding', 0));

    // ================= ラベル部分 =================
    x0 = $('#sortableFrame').position().left;
    y0 = $('#sortableFrame').position().top;
    x0 += 90, y0 += 15;
    x = x0, y = y0;

    //配置の設定
    ph = 5, pv = 10, w = 210, hh = 20, hv = 220;
    my = 0, mr = 5, mt = 35, f = 14, taw = (w - 2 * pv);

    // ---- .ui-state-highlight-video の CSS設定値 ----
    // width: (w + (ph+1)*2)
    // height: (hv + (pv+1)*2) )
    // margin-top: mt
    //

    //if(jQuery.browser.msie){
    if (!$.support.noCloneChecked) {
        //w += (ph+1) * 2;
        w += (ph + 1) * 2;
        hh += (ph + 1) * 2 + 1;
        my = -((ph + 1) * 2);
    }

    max = values.length;
    for (i = 0; i < max; i += 1) {
        $('#sortableFrame')
            .append($("<div>" + (i + 1) + "位</div>")
                .attr('class', 'ui-state-default')
                .attr('align', 'center')
                .css('position', "absolute")
                .css('padding', ph)
                .css('left', x)
                .css('top', y)
                .css('width', w)
                .css('height', hh)
                .css('font-size', '16px')
            );

        x += (w + (ph + 1) * 2 + mr + my);
        if ((i + 1) % cols == 0) {
            x = x0;
            y += hv + ((pv + 1) * 2) + mt;
        }
    }

    // ================= アイテムをランダムな順に =================
    max = values.length;
    for (i = 0; i < max * 2; i += 1) {
        r1 = Math.round(Math.random() * (max - 1));
        r2 = Math.round(Math.random() * (max - 1));
        tmp = values[r1];
        values[r1] = values[r2];
        values[r2] = tmp;
    }

    // ================= ソートアイテム部分 =================
    w -= pv;
    //if(jQuery.browser.msie){
    if (!$.support.noCloneChecked) {
        w += pv;
        mt += 4;
    }

    max = values.length;
    for (i = 0; i < max; i += 1) {
        $("<li align='left'></li>")
            .attr('class', 'ui-state-default')
            .css('width', w + 'px')
            .css('height', hv + 'px')
            .css('margin-top', mt + 'px')
            .css('margin-right', mr + 'px')
            .css('padding', pv + 'px')
            .css('font-size', f + 'px')
            .css('cursor', 'pointer')
            .css('float', 'left')
            .append("映像" + (i + 1) +
                "<span id='video" + i + "'></span>")
            .append("<span id='btn-video" + i + "'></span>")
            .append("<span id='msg-video" + i + "'></span>")
            .append($('<textarea></textarea>')
                .css('width', taw + 'px')
                .css('height', (f * 2) + 'px')
                .css('overflow', 'hidden')
                .unbind()
                .bind('click', function () {
                    $(this).focus();
                })
                .bind('focusout', function () {
                    let userMemo = [], itemOrder, i;
                    jQuery.each($("li textarea"), function () {
                        userMemo.push($(this).attr('value'));
                    });

                    itemOrder = param[2] + " ";
                    i = 0;
                    jQuery.each($("li span:last-child"), function () {
                        itemOrder += $(this).attr('id') + "(" + userMemo[i] + ") ";
                        i += 1;
                    });
                    //console.log('"' + itemOrder + '"');
                    that.qAnswers[that.crrPage][idx].ans = itemOrder;
                }))
            .append("<span id='" + values[i] + "'></span>")
            .appendTo('#' + idval);

        $('#btn-video' + i)
            .css('position', 'asboslute')
            .css('margin-top', '10px')
            .append('<img src="css/images/play.png">');

        $('#msg-video' + i)
            .css('font-weight', 'normal')
            .css('font-size', '12px')
            .css('margin-left', '10px');

        this.embedVideo("video" + i, values[i]);
    }

    $("#" + idval).sortable({
        placeholder: "ui-state-highlight-video",
        update: function (event, ui) {
            let userMemo = [], itemOrder, i;
            jQuery.each($("li textarea"), function () {
                userMemo.push($(this).attr('value'));
            });

            itemOrder = param[2] + " ";
            i = 0;
            //console.log(jQuery.makeArray($("li span:last-child")));
            jQuery.each($("li span:last-child"), function () {
                //itemOrder += $(this).html() + " ";
                itemOrder += $(this).attr('id') + "(" + userMemo[i] + ") ";
                i += 1;
            });
            //console.log('"' + itemOrder + '"');

            that.qAnswers[that.crrPage][idx].ans = itemOrder;
            that.flgNoMovedSortableText = false;
            //console.log($("li span:last-child"));
        }
    });
    $("#" + idval).disableSelection();

    //デフォルトの回答を設定
    this.flgNoMovedSortableText = true;
    defAns = param[2] + " ";
    jQuery.each($("li span:last-child"), function () {
        //defAns += $(this).html() + " ";
        defAns += $(this).attr('id') + " ";
    });
    this.qAnswers[that.crrPage][idx].ans = defAns;
};


//ハッシュパラメータとURLパラメータの取得
quesMgr.checkURLparams = function () {
    //URLパラメータの取得
    let params = (new URL(document.location)).searchParams;

    //言語
    this.lang = params.get('lang') || "ja"; //なければnullが入るが，ここでは"ja"を設定

    //エリアCODE
    this.code = params.get('code') || ""; //なければnullが入るが，ここでは""を設定
    if (this.code.length > 0) {
        this.code += "_";
    }

    //デバッグモード(newpage命令のスキップ)
    this.debug = params.get('debug') || "off";

    //アスマークID・日本語版
    this.asm = params.get('asm') || "";
    if (this.asm.length > 0) {
        this.code = "asm_" + this.code;
    }

    //アスマークID・英語版
    this.asme1 = params.get('asme1') || "";
    this.asme2 = params.get('asme2') || "";
    if (this.asme1.length > 0 && this.asme1.length > 0) {
        this.code = "asme_" + this.code;
    }

    //回数チェック実施の有無 (デフォルトオフ）
    this.isAgain = params.get('again') || "off";
    if (this.debug === "on") {  //デバッグモードのときは回数チェックしない
        this.isAgain = "on";
    }
};

//全角から半角数字への変換
quesMgr.zenkaku2float = function (str) {
    return str.replace(/[Ａ-Ｚａ-ｚ０-９]/g, function (s) {
        return parseFloat(String.fromCharCode(s.charCodeAt(0) - 0xFEE0));
    });
};

// アスマーク(日本語版)入力受け付け状態の読み込み
quesMgr.isASMClosed = function () {
    let req = new XMLHttpRequest(), flg;
    req.addEventListener("readystatechange", () => {
        if (req.readyState === 4 && req.status === 200) {
            if (req.responseText !== "") {
                let data = JSON.parse(req.responseText);
                if (data.status === "closed") {
                    flg = true;
                } else {
                    flg = false;
                }
            } else {
                flg = false
            }
        }
    });
    req.open("POST", "php/loadASMstatus.php", false);
    req.setRequestHeader('content-type', 'application/json');
    req.send();
    return flg;
};

// アスマーク(英語版)入力受け付け状態の読み込み
quesMgr.isASMEnClosed = function () {
    let req = new XMLHttpRequest(), flg;
    req.addEventListener("readystatechange", () => {
        if (req.readyState === 4 && req.status === 200) {
            if (req.responseText !== "") {
                let data = JSON.parse(req.responseText);
                if (data.status === "closed") {
                    flg = true;
                } else {
                    flg = false;
                }
            } else {
                flg = false
            }
        }
    });
    req.open("POST", "php/loadASMEnstatus.php", false);
    req.setRequestHeader('content-type', 'application/json');
    req.send();
    return flg;
};

//地図情報の入力チェック設定
quesMgr.setCheckImgmapFile = function (param) {
    this.checkImgmapFile = true;
    this.checkImgmapTypes = [];
    for (let i = 1; i < param.length; i += 1) {
        this.checkImgmapTypes.push(param[i]);
    }
};

// 地図データが空かどうかのチェック
quesMgr.isEmptyMapData = function (pid, param) {
    let ret = $.ajax({
        type: "POST",
        url: "php/isEmptyMapData.php",
        data: { pid: pid, param: param },
        async: false
    }).responseText;

    if (ret === "true") {
        return true;
    } else if (ret === "false") {
        return false;
    }
};

// アスマーク用・同意しない場合
quesMgr.asmEngExit = function () {
    if (this.asme1 === "" || this.asme2 === "") { return; }

    let exitUrl = "https://dkr1.ssisurveys.com/projects/end?rst=2&psid=" + this.asme1 + "&pid=" + this.asme2;

    $('<a class="text_s"></a>')
        .attr('href', exitUrl)
        .text("If you disagree, click here to exit.")
        .css('display', 'block')
        .css('text-align', 'left')
        .css('vertical-align', 'top')
        .css('clear', 'left')
        .appendTo('#qTable');
};

// コードの表示
quesMgr.showCode = function () {
    if (this.code === "" || this.debug === "on") { return; }
    if (this.crrPage !== this.qPages.length - 1) { return; }

    $('<div id="ccode" class="text_s_bf"></div>')
        .appendTo('#qTable');

    fetch('php/getCompletionCode.php', {
        method: "post",
        body: JSON.stringify({ code: this.code }),
        headers: { "Content-Type": "application/json" }
    }).then((res) => {
        if (res.status !== 200) {
            throw new Error("system error.");
        }
        return res.text();
    }).then((text) => {
        $('#ccode').text(text);
    }).catch((e) => {
        console.log(e.message);
    }).finally(() => {
        // console.log("fetch ok.");
    });

    this.setTextSelect();
};


// -------------------------------------
// ページ移行したタイミングのタイムスタンプ
//特定のページでの時間を確認したい場合、そのページと次のページの２回呼び出しが必要
// -------------------------------------
quesMgr.timeStamp = function () {
    this.qAnswers[this.crrPage].push("");//アンサー配列のページ番目に空文字列を追加
    let idx = this.qAnswers[this.crrPage].length - 1;//あとからその番号を参照できるように最終番号を取得
    

    let cDate = new Date();
    let tYear = cDate.getFullYear();
    let tMonth = this.nf(cDate.getMonth() + 1, 2);
    let tDate = this.nf(cDate.getDate(), 2);
    let tHour = this.nf(cDate.getHours(), 2);
    let tMin = this.nf(cDate.getMinutes(), 2);
    let tSec = this.nf(cDate.getSeconds(), 2);
    let endDate = tYear + "-" + tMonth + "-" + tDate + "_" +
        tHour + "-" + tMin + "-" + tSec;
        
    this.qAnswers[this.crrPage][idx] = { ans: endDate, elec: false };
    
};



// -------------------------------------
//20240809 お盆前に微妙に時間があり，yswrが作成
//画像を選択してランキング化することのできるCanvasを作成
//第1引数でclass名を指定
//第2引数でcanvasの幅(px)を指定 
//第3引数でcanvasの高さ(px)を指定
//第4引数で画像のURLを指定 画像枚数はsplitで分割することで確定
// -------------------------------------
quesMgr.canvasSelectImage = function (param) {
    this.qAnswers[this.crrPage].push("");
    const idx = this.qAnswers[this.crrPage].length - 1;
    this.qAnswers[this.crrPage][idx] = { ans: "", elec: false };
    



    const width=param[2];
    const height=param[3];
    const images = param[4].split(' ');
    const imageCache = {}; // 画像キャッシュ
    // 選択中の画像のインデックス
    let selectedImageIndex = -1;
    let now_mouse_select_index = -1;

    let before_x = 0;
    let before_y = 0;
    let img_show_list = [];
    for (let i = 0; i < images.length; i++) {
      img_show_list.push(-1);
    }
     // 画像をロードする関数
  function loadImage(src) {
    if (imageCache[src]) {
      return Promise.resolve(imageCache[src]);
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        imageCache[src] = img;
        resolve(img);
      };
      img.onerror = reject;
      img.src = src;
    });
  }
  
    // 画像をキャンバスに描画する関数
    async function drawImage(img, x, y) {
        ctx.drawImage(img, x, y, canvas.width / images.length, canvas.height/2);
  
    }
    async function drawImageByMouseDown(img, x, y) {
        img_width = canvas.width / images.length;
        img_height = canvas.height;
        ctx.drawImage(img, x - img_width / 2, y - img_height / 4, canvas.width / (images.length)*0.8, canvas.height*0.4);
        
  
    }

    // 画像を並べる関数
    async function arrangeImages() {
        // 画像をロードする
        const loadedImages = await Promise.all(images.map(loadImage));
     //上部分
        let x = 0;
        for (let i = 0; i < images.length; i++) {
          await drawImage(loadedImages[i], x, 0);
          if(i === selectedImageIndex){
            //半透明の四角を描画
            ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
            ctx.fillRect(x, 0, canvas.width / images.length, canvas.height/2);
          }
          x += canvas.width / images.length;
        }
  
        //下部分
        x=0;
        for (let i = 0; i < images.length; i++) {
          if(img_show_list[i] !== -1){
            await drawImage(loadedImages[img_show_list[i]], x, canvas.height/2);
          }
          else{
            ctx.fillStyle = "rgba(200, 200, 200, 1)";
            ctx.fillRect(x, canvas.height/2, canvas.width / images.length, canvas.height);
          }
          if(i=== now_mouse_select_index){
            //半透明の四角を描画
            ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
            ctx.fillRect(x, canvas.height/2, canvas.width / images.length, canvas.height);
          }
          x += canvas.width / images.length;
        }

        if(selectedImageIndex !== -1){
            drawImageByMouseDown(loadedImages[selectedImageIndex], before_x, before_y);
        }

    }
    $('<canvas></canvas>')
        .attr('class',param[1]) 
        .attr('width',width)
        .attr('height',height)
        .appendTo('#qTable');
    
    const canvas = document.querySelector('.'+param[1]);
    const ctx = canvas.getContext("2d");

    // マウスダウン時の処理
    canvas.addEventListener("mousedown", async (event) => {
        // マウスダウンした位置を取得
        const x = event.offsetX;
        const y = event.offsetY;
        if(y>canvas.height/2){
          return;
        }
        // マウスダウンした画像のインデックスを取得
        selectedImageIndex = Math.floor(x / (canvas.width / images.length));
  
        // 選択中の画像を記録
        if (selectedImageIndex >= 0 && selectedImageIndex < images.length) {
          // 画像をロードする
          const img = await loadImage(images[selectedImageIndex]);
          // 画像を新しい位置に描画する
          await drawImageByMouseDown(img, x, y);
        }
      });
  
      // マウスアップ時の処理
      canvas.addEventListener("mouseup", async (event) => {
        
        
  
        //離した際のマウス位置を確認
        const x = event.offsetX;
        const y = event.offsetY;
  
        // マウスダウンした画像のインデックスを取得
        const imageIndex = Math.floor(x / (canvas.width / images.length));
  
        if(selectedImageIndex !== -1){
          //すでにリストの中にselectedImageIndexがある場合
          if(img_show_list.includes(selectedImageIndex)){
            //リストの中のselectedImageIndexの位置を変更 もともとの場所には-1を入れる
            index=img_show_list.indexOf(selectedImageIndex);
            img_show_list[index] = -1;
          }
          img_show_list[now_mouse_select_index] = selectedImageIndex;
          
          
        }
        
        // 選択中の画像のインデックスをリセット
        selectedImageIndex = -1;
        now_mouse_select_index = -1;
        //0.1秒後に画像を再描画
        requestAnimationFrame(arrangeImages);
  
      });
      canvas.addEventListener("mouseout", () => {
        // 選択中の画像のインデックスをリセット
        selectedImageIndex = -1;
        now_mouse_select_index = -1;
        //0.1秒後に画像を再描画
        requestAnimationFrame(arrangeImages);
      });
  
      // マウスオーバー時の処理
      canvas.addEventListener("mousemove", async (event) => {
        //console.log("move");
        // マウスオーバーした位置を取得
        const x = event.offsetX;
        const y = event.offsetY;
        if (selectedImageIndex === -1) {
          //requestAnimationFrame(arrangeImages);
          return;
        }
        //前回からの移動距離が3未満であれば移動しない
        //描画コストを抑えるため
        if (Math.abs(before_x - x) < 3 && Math.abs(before_y - y) < 3) {
          return;
        }
        
        before_x = x;
        before_y = y;
  
        // マウスオーバーした画像のインデックスを取得
        if(y<canvas.height/2){
          now_mouse_select_index=-1;
          return;
        }
        now_mouse_select_index = Math.floor(x / (canvas.width / images.length));
  
        // マウスダウン中の場合のみ、画像を移動する
        // 画像をロードする
        const img = await loadImage(images[selectedImageIndex]);
        // 画像を新しい位置に描画する
        requestAnimationFrame(arrangeImages);
  
      });
  
      // 画像を並べて描画
      requestAnimationFrame(arrangeImages);
    
};


// -------------------------------------
//特定のページでの時間を確認したい場合、そのページと次のページの２回呼び出しが必要
// -------------------------------------
quesMgr.timeStamp = function () {
    this.qAnswers[this.crrPage].push("");//アンサー配列のページ番目に空文字列を追加
    let idx = this.qAnswers[this.crrPage].length - 1;//あとからその番号を参照できるように最終番号を取得
    

    let cDate = new Date();
    let tYear = cDate.getFullYear();
    let tMonth = this.nf(cDate.getMonth() + 1, 2);
    let tDate = this.nf(cDate.getDate(), 2);
    let tHour = this.nf(cDate.getHours(), 2);
    let tMin = this.nf(cDate.getMinutes(), 2);
    let tSec = this.nf(cDate.getSeconds(), 2);
    let endDate = tYear + "-" + tMonth + "-" + tDate + "_" +
        tHour + "-" + tMin + "-" + tSec;
        
    this.qAnswers[this.crrPage][idx] = { ans: endDate, elec: false };
    
};


// -------------------------------------
// repeatStart命令で得られる文字列を1ページ内で複数回利用したい場合の保管場所
// -------------------------------------

quesMgr.getRandFileText = function (param) {
    quesMgr.repFileMgr.nowText=this.repFileMgr.getFname();
    console.log(this.repFileMgr.nowText);
}

// -------------------------------------
// 名前（テキスト）選択チェックボックスの作成 (グループID対応版)
// -------------------------------------
quesMgr.makeNameCheck = function (param) {
    let that = this;

    let isMulti = (param[0] && String(param[0]).toLowerCase().indexOf("multi") !== -1);
    let baseW = parseInt(param[4]) || 150;
    let nameList = param.slice(5);

    // グループIDを生成 (この質問グループを一意に識別するため)
    let groupID = "group_" + this.crrPage + "_" + this.qAnswers[this.crrPage].length;

    let validCount = 0;
    for (let i = 0; i < nameList.length; i++) {
        if ((nameList[i] || "").trim() !== "") validCount++;
    }

    let isElective = false;
    if (param[2] && String(param[2]).toLowerCase().indexOf("elective") !== -1) {
        isElective = true;
    }

    let startIdx = this.qAnswers[this.crrPage].length;

    for (let i = 0; i < validCount; i++) {
        this.qAnswers[this.crrPage].push("");
        let idx = this.qAnswers[this.crrPage].length - 1;
        // ★重要: ans="0", そして groupID をデータに追加して連携させる
        this.qAnswers[this.crrPage][idx] = { ans: "0", elec: isElective, groupID: groupID };
    }

    if (param[3] && String(param[3]).trim() !== "") {
        $('<div class="text_s"></div>')
            .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(startIdx, 4) + "_desc")
            .html(param[3])
            .appendTo("#qTable");
    }

    let container = $('<div class="name-check-container"></div>')
        // グループIDをDOMにも持たせる
        .attr('data-group-id', groupID)
        .css('display', 'flex').css('flex-wrap', 'wrap').css('gap', '15px')
        .css('justify-content', 'center').css('margin-top', '10px').css('margin-bottom', '30px');

    let chkOffImg = "css/images/checkboxOff.png";
    let chkOnImg = "css/images/checkboxOn.png";
    let itemIndexCounter = 0;

    for (let i = 0; i < nameList.length; i++) {
        let textVal = (nameList[i] || "").trim();
        if (textVal === "") continue;

        let currentAnsIdx = startIdx + itemIndexCounter;
        itemIndexCounter++;

        let frame = $('<div></div>')
            .attr('id', "P" + that.nf(that.crrPage, 2) + "_Q" + that.nf(currentAnsIdx, 4))
            .css('width', baseW + 'px').css('min-height', '80px').css('padding', '10px')
            .css('border', '2px solid #ccc').css('border-radius', '8px').css('cursor', 'pointer')
            .css('display', 'flex').css('flex-direction', 'column')
            .css('align-items', 'center').css('justify-content', 'space-between')
            .css('background-color', '#fff').css('transition', 'all 0.2s').css('box-sizing', 'border-box')
            .attr('data-idx', currentAnsIdx);

        let textContainer = $('<div></div>')
            .css('width', '100%').css('flex-grow', '1').css('display', 'flex')
            .css('align-items', 'center').css('justify-content', 'center').css('text-align', 'center')
            .css('font-weight', 'bold').css('font-size', '16px').css('word-break', 'break-word')
            .css('margin-bottom', '10px').html(textVal);

        let chkEl = $('<img>').attr('class', 'chk-icon').attr('src', chkOffImg)
            .css('width', '24px').css('height', '24px').css('flex-shrink', '0');

        frame.append(textContainer).append(chkEl);

        frame.click(function () {
            let myAnsIdx = parseInt($(this).attr('data-idx'));
            let isSelected = ($(this).attr('data-selected') === 'true');

            if (isMulti) {
                if (isSelected) {
                    $(this).attr('data-selected', 'false');
                    $(this).css('border-color', '#ccc').css('background-color', '#fff');
                    $(this).find('.chk-icon').attr('src', chkOffImg);
                    that.qAnswers[that.crrPage][myAnsIdx].ans = "0";
                } else {
                    $(this).attr('data-selected', 'true');
                    $(this).css('border-color', that.hlColor2).css('background-color', '#fff8e1');
                    $(this).find('.chk-icon').attr('src', chkOnImg);
                    that.qAnswers[that.crrPage][myAnsIdx].ans = "1";
                }
            } else {
                $(this).parent().children().each(function () {
                    let otherAnsIdx = parseInt($(this).attr('data-idx'));
                    $(this).css('border-color', '#ccc').css('background-color', '#fff');
                    $(this).find('.chk-icon').attr('src', chkOffImg);
                    $(this).attr('data-selected', 'false');
                    if (!isNaN(otherAnsIdx)) that.qAnswers[that.crrPage][otherAnsIdx].ans = "0";
                });
                $(this).css('border-color', that.hlColor2).css('background-color', '#fff8e1');
                $(this).find('.chk-icon').attr('src', chkOnImg);
                $(this).attr('data-selected', 'true');
                that.qAnswers[that.crrPage][myAnsIdx].ans = "1";
            }
        });

        frame.hover(
            function () { if ($(this).attr('data-selected') !== 'true') $(this).css('border-color', '#999').css('background-color', '#f9f9f9'); },
            function () { if ($(this).attr('data-selected') !== 'true') $(this).css('border-color', '#ccc').css('background-color', '#fff'); }
        );
        container.append(frame);
    }
    container.appendTo('#qTable');
};