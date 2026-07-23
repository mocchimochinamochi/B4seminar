<?php
$stimulus = array(
  'conf/iframe/chatItoI_name_js/index.html',
  'conf/iframe/chatItoE_name_js/index.html',
  'conf/iframe/chatItoI_noname_js/index.html',
  'conf/iframe/chatItoE_noname_js/index.html'
);
//カウンター用いて分岐
$fname = "./counter/counter.dat";

$maxNum = count($stimulus) -1;

$fp = fopen($fname, 'r');
$str = fgets($fp);
$num = 0;
if ($str != '') {
   $num = intval($str);
}
fclose($fp);

echo $stimulus[$num];

$fp = fopen($fname, 'w');
flock($fp, LOCK_EX);
$wnum = $num + 1;

if ($wnum > $maxNum) {
   $wnum = 0;
}

if (!fwrite($fp, $wnum) ) {
   echo "error.";
   fclose($fp);
   exit;
}

fclose($fp);
?>