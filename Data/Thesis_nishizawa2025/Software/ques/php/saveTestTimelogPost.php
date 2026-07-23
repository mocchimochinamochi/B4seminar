<?php
// saveTestTimelogPost.php (ユーザー別ファイル保存版)

ini_set('display_errors', 1);
error_reporting(E_ALL);

$gid = isset($_POST['gid']) ? $_POST['gid'] : 'unknown';
$lang = isset($_POST['lang']) ? $_POST['lang'] : 'unknown';
$startDate = isset($_POST['startDate']) ? $_POST['startDate'] : '';
$endDate = isset($_POST['endDate']) ? $_POST['endDate'] : '';
$timeLog = isset($_POST['timeLog']) ? $_POST['timeLog'] : '';

if (empty($timeLog)) {
    echo "Error: Empty data";
    exit;
}

// GID（ユーザーID）をファイル名に使うための安全対策
// 英数字、ハイフン、アンダースコア以外を削除して、ディレクトリトラバーサル攻撃を防ぐ
$safeGid = preg_replace('/[^a-zA-Z0-9_-]/', '', $gid);

// GIDが空（不正）だった場合のフォールバック
if (empty($safeGid)) {
    $safeGid = "unknown_user";
}

// ファイル名にGIDを含める
// 例: ../save/quesData/timelog_wzceI1FRS4.csv
$fname = "../save/quesData/timelog/timelog_" . $safeGid . ".csv";

// ファイルを開く (a+ なので、同ユーザーが複数回保存した場合は追記されます)
$fp = fopen($fname, 'a+');

if (!$fp) {
    echo "Error: Could not open file ($fname)";
    exit;
}

if (flock($fp, LOCK_EX)) {
    // ファイルサイズが0（新規作成時）ならヘッダーを書き込む
    clearstatcache(); // ファイルサイズのキャッシュをクリア
    if (filesize($fname) === 0) {
        // Excel等で文字化けしないようにBOM(Byte Order Mark)をつける
        fwrite($fp, "\xEF\xBB\xBF"); 
        
        // ヘッダー行 (PHPで付与する情報 + JSからの情報)
        $header = "ユーザーID,開始日時,終了日時,条件,経過時間(秒),イベント,シナリオID,ステップ数\n";
        fwrite($fp, $header);
    }

    $lines = preg_split("/\r\n|\n|\r/", $timeLog);
    foreach ($lines as $line) {
        // JS側から送られてくるヘッダー行や空行はスキップ
        if (trim($line) === "" || strpos($line, "Interval") !== false || strpos($line, "ItoE_name") !== false) {
            continue;
        }
        // CSV行の作成
        $csvRow = "$gid,$startDate,$endDate,$line\n";
        fwrite($fp, $csvRow);
    }
    flock($fp, LOCK_UN);
} else {
    echo "Error: Could not lock file";
}

fclose($fp);
echo "Success";
?>