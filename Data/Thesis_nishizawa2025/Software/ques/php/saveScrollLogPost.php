<?php
// saveScrollLogPost.php 

ini_set('display_errors', 1);
error_reporting(E_ALL);

$gid = isset($_POST['gid']) ? $_POST['gid'] : 'unknown';
$lang = isset($_POST['lang']) ? $_POST['lang'] : 'unknown';
$startDate = isset($_POST['startDate']) ? $_POST['startDate'] : '';
$endDate = isset($_POST['endDate']) ? $_POST['endDate'] : '';
$scrollLog = isset($_POST['scrollLog']) ? $_POST['scrollLog'] : '';

// GIDの安全対策
$safeGid = preg_replace('/[^a-zA-Z0-9_-]/', '', $gid);

if (empty($safeGid)) {
    $safeGid = "unknown_user";
}

// ファイル名
$fname = "../save/quesData/scrolllog/scrolllog_" . $safeGid . ".csv";

$fp = fopen($fname, 'a+');

if (!$fp) {
    http_response_code(500);
    echo "Error: Could not open file ($fname)";
    exit;
}

if (flock($fp, LOCK_EX)) {
    clearstatcache();
    
    // 1. ファイルが空（新規作成）ならヘッダーを書き込む
    if (filesize($fname) === 0) {
        fwrite($fp, "\xEF\xBB\xBF"); // BOM
        
        // 
        $header = "ユーザーID,開始日時,終了日時,条件,経過時間(秒),イベントタイプ,Y座標,スクロール率,表示メッセージID,シナリオID\n";
        fwrite($fp, $header);
    }

    if (empty($scrollLog)) {
        // ---------------------------------------------------------
        // パターンA: スクロール操作が一度もなかった場合
        // ---------------------------------------------------------
        //
        // 列構成: GID, Start, End, Time, Event, Y, %, MsgID, ScenarioID
        $noScrollRow = "$gid,$startDate,$endDate,-,スクロールなし,-,-,-,-\n";
        
        fwrite($fp, $noScrollRow);
        
    } else {
        // ---------------------------------------------------------
        // パターンB: 通常のログがある場合
        // ---------------------------------------------------------
        $lines = preg_split("/\r\n|\n|\r/", $scrollLog);
        foreach ($lines as $line) {
            if (trim($line) === "") {
                continue;
            }
            // JS側のヘッダー行はスキップ
            if (strpos($line, "Time(sec)") !== false) {
                continue;
            }

            $csvRow = "$gid,$startDate,$endDate,$line\n";
            fwrite($fp, $csvRow);
        }
    }
    
    flock($fp, LOCK_UN);
} else {
    echo "Error: Could not lock file";
}

fclose($fp);
echo "Success";
?>