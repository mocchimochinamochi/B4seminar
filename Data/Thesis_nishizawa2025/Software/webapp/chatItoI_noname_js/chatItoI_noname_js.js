/*
 * Chat Simulation p5.js Sketch (Final Fix Version 8 - Human Icon & Tail)
 * * 修正内容:
 * - ベース: ユーザー提供の Version 8
 * - 変更点:
 * 1. エージェントのアイコンを「黒い円」から「人型（上半身）」のシルエットに変更
 * 2. エージェントのメッセージ枠に「しっぽ（三角形）」を追加して吹き出しに変更
 * ★統合済: 禁則処理と厳密な高さ計算
 */

// --- クラス定義 ---

class ChatMessage {
  constructor(isUser, text, img) {
    this.isUser = isUser;
    this.text = text;
    this.img = img;
  }
}

// --- グローバル変数 ---

let scenario = [];
let history = [];
let scenarioIndex = 0;
let currentScenarioID = 0;
let allSessionsCompleted = false;

let myFont;
let dummyImage;
let chatImages = [];

let imgDisplayWidth = 270;

// UIレイアウト用変数
let currentScrollY = 20;
let targetScrollY = 20;

let contentHeight = 0;

const MARGIN = 20;
const ICON_SIZE = 40;

// 吹き出し内の余白
const BUBBLE_PADDING = 10;

// 吹き出しの幅
const BUBBLE_WIDTH = 230;

let currentInputHeight = 80;
const MIN_INPUT_HEIGHT = 80;

const NEXT_BUTTON_AREA_HEIGHT = 100;

// サイドバーの幅
const SIDEBAR_WIDTH = 60;

// スクロールバーの設定
const SCROLLBAR_WIDTH = 10;
let isDraggingScrollbar = false;

// エージェントの返答遅延用変数
let isWaitingForAgent = false;
let nextResponseTime = 0;
const AGENT_DELAY_MS = 2000;

// ユーザーの入力演出用変数
let isUserTyping = false;
let userTypingEndTime = 0;
let userTypingStartTime = 0;
const USER_TYPING_MS = 1500; 
const START_DELAY_MS = 100; // 100ms
let isReadyToStartTyping = false; 

// リセット時の待機処理用変数
let isResetting = false;
let resetStartTime = 0;
let isHoveringResetIcon = false; 

// 完了メッセージ送信管理フラグ
let completionMessageSent = false;

// 新規作成後のウェイト用変数
let isLoadingNextScenario = false;
let loadingStartTime = 0;
const LOADING_DELAY_MS = 1000; 

// 演出用変数
let sendButtonHighlightStart = -1000;
let newChatIconHighlightStart = -1000;
const HIGHLIGHT_DURATION = 300;

// テキストサイズと行間
const TEXT_SIZE = 16;
const TEXT_LEADING = 22;

// ★追加: 行頭に来てはいけない文字リスト（禁則文字）
const PROHIBITED_START_CHARS = ["。", "、", "」", "』", "）", "}", "]", "！", "？", "!", "?", "…", "～"];

// ログ用の時間計測変数
let lastLogTime = 0;
let timeLogData = ""; // ログ蓄積用

// スクロールログ用の変数
let scrollLogData = ""; 
let lastScrollLogTime = 0; // スクロールログの間引き用
const SCROLL_LOG_INTERVAL = 200; // 200msごとに記録
let scenarioStartTime = 0; // シナリオ開始時刻を記録する変数

// 仮想マウスカーソル用の変数
let vCursorX = 0;
let vCursorY = 0;
let iconHighlightTriggered = false;


// --- 関数定義 ---

function preload() {
  for (let i = 0; i < 8; i++) {
    let filename = "";
    if (i < 4) {
      filename = "chat1_" + (i + 1) + ".jpg";
    } else {
      filename = "chat2_" + (i - 3) + ".jpg";
    }
    chatImages[i] = loadImage(filename);
  }
}

function setup() {
  createCanvas(500, 600);

  // textWrap(CHAR); // ★削除: 自前で禁則処理を行うため削除

  textFont("Yu Gothic UI");
  textStyle(BOLD);
  textSize(TEXT_SIZE);

  dummyImage = createImage(200, 150);
  dummyImage.loadPixels();
  for (let i = 0; i < dummyImage.width * dummyImage.height; i++) {
    let r = random(100, 200);
    let g = random(100, 200);
    let b = 255;
    let idx = i * 4;
    dummyImage.pixels[idx] = r;
    dummyImage.pixels[idx + 1] = g;
    dummyImage.pixels[idx + 2] = b;
    dummyImage.pixels[idx + 3] = 255;
  }
  dummyImage.updatePixels();

  for (let i = 0; i < 8; i++) {
    if (chatImages[i] && chatImages[i].width > 1) {
      let newHeight = int(chatImages[i].height * (imgDisplayWidth / chatImages[i].width));
      chatImages[i].resize(imgDisplayWidth, newHeight);
    } else {
      chatImages[i] = dummyImage;
    }
  }

  // 初期カーソル位置
  let inputAreaWidth = width - SIDEBAR_WIDTH;
  vCursorX = SIDEBAR_WIDTH + inputAreaWidth - MARGIN - 80 + 40;
  vCursorY = height - NEXT_BUTTON_AREA_HEIGHT - 40;

  // メインログ初期化
  timeLogData = "Interval(sec), Event, ScenarioID, StepIndex\n";
  window.timeLogData = timeLogData;
  localStorage.setItem('chatTimeLog', timeLogData);

  // スクロールログ初期化
  scrollLogData = "Time(sec), EventType, RawY, ScrollPercent, VisibleMsgID, ScenarioID\n";
  // localStorage.setItem('scrollLog', scrollLogData);

  lastLogTime = millis();
  //console.log("Interval(sec), Event, ScenarioID, StepIndex");
  
  // アプリ起動時に時間をセット
  scenarioStartTime = millis();

  setupScenario();
}

function setupScenario() {
  scenario = [];

  switch (currentScenarioID) {
    case 0: // INFJ1
      scenario.push(new ChatMessage(false, "画像生成のお手伝いをしますので、作りたいイメージの概要を教えてください。", null));
      scenario.push(new ChatMessage(true, "トランプのクラブとスペードのカードを2枚ずつ描いてください。", null));
      scenario.push(new ChatMessage(false, "はい、クラブとスペードのトランプカードを、それぞれ2枚ずつ描写します。", chatImages[0]));
      scenario.push(new ChatMessage(true, "左から2番目の数字をQに変更してください。", null));
      scenario.push(new ChatMessage(false, "承知しました。左から2番目のカードの数字をQに変更します。", chatImages[1]));
      scenario.push(new ChatMessage(true, "QではなくKに変更されています。数字をKからQに変更してください。", null));
      scenario.push(new ChatMessage(false, "申し訳ありません。私の不手際でした。左から2番目のカードの数字をKからQに変更します。", chatImages[2]));
      scenario.push(new ChatMessage(true, "変更されていません。左から2番目のカードをQに変更してください。", null));
      scenario.push(new ChatMessage(false, "大変申し訳ありません。意図した変更が反映されていなかったようですね。改めて、左から2番目のカードをQに変更します。", chatImages[3]));
      break;

    case 1: // INFJ
      scenario.push(new ChatMessage(false, "画像生成のお手伝いをしますので、作りたいイメージの概要を教えてください。", null));
      scenario.push(new ChatMessage(true, "トランプのクラブとスペードのカードを2枚ずつ描いてください。", null));
      scenario.push(new ChatMessage(false, "はい、承知しました。クラブとスペードのトランプカードを、それぞれ2枚ずつ描きます。", chatImages[4]));
      scenario.push(new ChatMessage(true, "スペードが1枚しか書かれていません。スペードを2枚にしてください。", null));
      scenario.push(new ChatMessage(false, "申し訳ありません。スペードが1枚しか描かれていませんでしたね。スペードを2枚にして描きます。", chatImages[5]));
      scenario.push(new ChatMessage(true, "スペードが1枚のままで修正されていません。スペードを2枚にしてください。", null));
      scenario.push(new ChatMessage(false, "大変申し訳ありません。まだスペードが1枚のままでしたね。ご迷惑をおかけして申し訳ありません。今度こそ、スペードを2枚にして描きます。", chatImages[6]));
      scenario.push(new ChatMessage(true, "修正されていません。1番右のカードをスペードに変更してください。", null));
      scenario.push(new ChatMessage(false, "大変申し訳ありません。ご指摘ありがとうございます。右端のカードをスペードに変更し、スペードが2枚になるように改めて描きます。", chatImages[7]));
      break;
  }

  if (scenario.length > 0) {
    if (!scenario[0].isUser) {
      isWaitingForAgent = true;
      nextResponseTime = millis() + AGENT_DELAY_MS;
      updateContentHeight();
      scrollToBottom();
    } else {
      isReadyToStartTyping = true;
    }
  }
}

function draw() {
  background(220);

  if (isLoadingNextScenario) {
    if (millis() - loadingStartTime > LOADING_DELAY_MS) {
      setupScenario();
      isLoadingNextScenario = false;
    }
  }

  if (allSessionsCompleted && !isResetting && !completionMessageSent) {
      window.parent.postMessage({ type: 'show_completion_code' }, '*');
      completionMessageSent = true;
  }

  updateUserTyping(); 
  updateAgentResponse();
  calculateInputAreaHeight();
  updateContentHeight();

  // --- チャットエリア ---
  push();
  let viewableHeight = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
  let maxScroll = MARGIN;
  let minScroll = viewableHeight - contentHeight - MARGIN - 20;
  if (minScroll > MARGIN) minScroll = MARGIN;

  if (!isDraggingScrollbar) {
    targetScrollY = constrain(targetScrollY, minScroll, maxScroll);
    currentScrollY = lerp(currentScrollY, targetScrollY, 0.1);
    if (abs(currentScrollY - targetScrollY) < 0.1) {
      currentScrollY = targetScrollY;
    }
  } else {
    currentScrollY = constrain(currentScrollY, minScroll, maxScroll);
    targetScrollY = currentScrollY;
  }

  translate(SIDEBAR_WIDTH, currentScrollY);

  let currentY = 0;
  for (let msg of history) {
    currentY += drawMessage(msg, currentY) + 20;
  }

  if (isWaitingForAgent) {
    drawTypingIndicator(currentY, false);
  }
    
  pop();

  // --- UI ---
  drawInputArea();
  drawSidebar();
  drawNextButtonArea();
  drawScrollbar();

  updateVirtualCursor();
  drawVirtualCursor();
}

// 人型アイコン描画用関数
function drawAgentIcon(x, y, size, col) {
  push();
  translate(x, y);
  fill(col);
  noStroke();
  // 頭 (円)
  ellipse(size/2, size * 0.35, size * 0.4);
  // 体 (半円)
  arc(size/2, size, size * 0.8, size * 0.55, PI, TWO_PI);
  pop();
}

// ★追加: 禁則処理（追い出し処理）を含んだ行分割関数
function splitTextToLines(str, maxWidth) {
  let finalLines = [];
   
  let rawLines = str.replace(/\r/g, "").split('\n');

  push();
  textSize(TEXT_SIZE); 
   
  for(let i=0; i<rawLines.length; i++) {
    let rawLine = rawLines[i];
    
    if(rawLine.length === 0) {
      finalLines.push("");
      continue;
    }
    
    let currentLine = "";
    let currentW = 0;

    for (let j = 0; j < rawLine.length; j++) {
      let c = rawLine.charAt(j);
      let charW = textWidth(c);
      
      if (currentW + charW > maxWidth) {
        let isProhibited = false;
        for (let k = 0; k < PROHIBITED_START_CHARS.length; k++) {
            if (c === PROHIBITED_START_CHARS[k]) {
                isProhibited = true;
                break;
            }
        }

        if (isProhibited && currentLine.length > 0) {
            let lastChar = currentLine.slice(-1);
            let prevLine = currentLine.slice(0, -1);
            
            finalLines.push(prevLine); 
            currentLine = lastChar + c; 
            currentW = textWidth(currentLine);
        } else {
            finalLines.push(currentLine);
            currentLine = c;
            currentW = charW;
        }
      } else {
        currentLine += c;
        currentW += charW;
      }
    }
    finalLines.push(currentLine);
  }
   
  pop();
  return finalLines;
}

// ★修正: テキストの高さ計算を厳密に（splitTextToLinesを使用）
function calculateTextHeight(str, specificWidth) {
  let lines = splitTextToLines(str, specificWidth);
  if (lines.length === 0) return 0;
  // (行数 * 行間) から (行間と文字サイズの差分 = 最後の行の下の余白) を引く
  return (lines.length * TEXT_LEADING) - (TEXT_LEADING - TEXT_SIZE);
}

function drawMessage(msg, y) {
  let h = 0;
  let bubblePadding = BUBBLE_PADDING;
  let chatAreaWidth = width - SIDEBAR_WIDTH;

  textSize(TEXT_SIZE);
  textLeading(TEXT_LEADING);

  if (msg.isUser) {
    fill(100, 180, 255);
    noStroke();
    rectMode(CORNER);

    // ★修正: 禁則処理付きで行分割
    let lines = splitTextToLines(msg.text, BUBBLE_WIDTH - bubblePadding * 2);
    let textH = (lines.length > 0) ? (lines.length * TEXT_LEADING) - (TEXT_LEADING - TEXT_SIZE) : 0;
    h = int(textH) + bubblePadding * 2;

    rect(chatAreaWidth - MARGIN - BUBBLE_WIDTH - SCROLLBAR_WIDTH, y, BUBBLE_WIDTH, h, 10);
    fill(255);
    textAlign(LEFT, TOP);
    
    // ★修正: 分割された行をループで描画
    let ly = y + bubblePadding;
    for (let lineStr of lines) {
      text(lineStr, chatAreaWidth - MARGIN - BUBBLE_WIDTH - SCROLLBAR_WIDTH + bubblePadding, ly);
      ly += TEXT_LEADING;
    }

  } else {
    // --- エージェント側 ---
    
    drawAgentIcon(MARGIN, y, ICON_SIZE, color(0));

    // メッセージコンテンツの座標
    let contentX = MARGIN * 2 + ICON_SIZE;
    let currentContentY = y;
    let startY = y;

    if (msg.text.length > 0) {
      fill(255);
      noStroke();

      // ★修正: 禁則処理付きで行分割
      let lines = splitTextToLines(msg.text, BUBBLE_WIDTH - bubblePadding * 2);
      let textH = (lines.length > 0) ? (lines.length * TEXT_LEADING) - (TEXT_LEADING - TEXT_SIZE) : 0;
      let textBoxH = int(textH) + bubblePadding * 2;
      
      // 吹き出しのしっぽ (三角形)
      triangle(contentX, currentContentY + 10, 
               contentX, currentContentY + 20, 
               contentX - 8, currentContentY + 15);

      rect(contentX, currentContentY, BUBBLE_WIDTH, textBoxH, 10);
      
      fill(0);
      textAlign(LEFT, TOP);
      
      // ★修正: 分割された行をループで描画
      let ly = currentContentY + bubblePadding;
      for (let lineStr of lines) {
        text(lineStr, contentX + bubblePadding, ly);
        ly += TEXT_LEADING;
      }
      currentContentY += textBoxH + 10;
    }
    if (msg.img != null) {
      image(msg.img, contentX, currentContentY);
      currentContentY += msg.img.height;
    }
    h = currentContentY - startY;
  }
  return h;
}

function startUserTyping() {
  isUserTyping = true;
  userTypingStartTime = millis(); 
  userTypingEndTime = millis() + START_DELAY_MS + USER_TYPING_MS;
}

function updateUserTyping() {
  if (isUserTyping && millis() > userTypingEndTime) {
    isUserTyping = false; 
  }
}

function updateVirtualCursor() {
  let inputAreaWidth = width - SIDEBAR_WIDTH;
  
  let btnHeight = 40; 
  let btnY_top = (height - NEXT_BUTTON_AREA_HEIGHT - currentInputHeight) + MARGIN + (currentInputHeight - MARGIN * 2 - btnHeight);
  let btnY_center = btnY_top + btnHeight / 2;
  
  let btnX = SIDEBAR_WIDTH + inputAreaWidth - MARGIN - 80 + 40; 
  let btnY = btnY_center;

  let iconX = SIDEBAR_WIDTH / 2; 
  let iconY = 70 + 34 / 2;

  let targetX = btnX;
  let targetY = btnY;
  
  let lerpAmt = 0.1;

  if (isResetting) {
    targetX = iconX;
    targetY = iconY;
    
    lerpAmt = 0.05;
    
    if (dist(vCursorX, vCursorY, iconX, iconY) < 2) {
      isHoveringResetIcon = true;
      if (!iconHighlightTriggered) {
         newChatIconHighlightStart = millis();
         iconHighlightTriggered = true;
      }
    } else {
      isHoveringResetIcon = false;
    }
  } else {
    isHoveringResetIcon = false;
  }

  vCursorX = lerp(vCursorX, targetX, lerpAmt);
  vCursorY = lerp(vCursorY, targetY, lerpAmt);
}

function drawVirtualCursor() {
  push();
  translate(vCursorX, vCursorY);
   
  fill(255); 
  stroke(0); 
  strokeWeight(1.5);
   
  beginShape();
  vertex(0, 0);
  vertex(0, 20);
  vertex(5, 15);
  vertex(12, 24);
  vertex(16, 21);
  vertex(9, 12);
  vertex(16, 12);
  endShape(CLOSE);
   
  pop();
}

function drawTypingIndicator(y, isUser) {
  let contentX;
  let w = 70;
  let h = 40;
   
  drawAgentIcon(MARGIN, y, ICON_SIZE, color(0));

  contentX = MARGIN * 2 + ICON_SIZE;
  fill(255); 
  noStroke();
  
  // 吹き出しのしっぽ
  triangle(contentX, y + 10, 
           contentX, y + 20, 
           contentX - 8, y + 15);

  rect(contentX, y, w, h, 10);

  fill(150); 
   
  let dotSize = 8;
  let startX = contentX + 20;
  let startY = y + h / 2;

  let time = millis() * 0.01;

  for (let i = 0; i < 3; i++) {
    let offset = sin(time + i * 1.5) * 3;
    ellipse(startX + i * 15, startY + offset, dotSize, dotSize);
  }
}

function drawScrollbar() {
  let viewableHeight = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
  let totalHeight = contentHeight + MARGIN * 2 + 20;

  if (totalHeight <= viewableHeight) return;

  let ratio = viewableHeight / totalHeight;
  let barHeight = viewableHeight * ratio;

  if (barHeight < 30) barHeight = 30;

  let maxScroll = MARGIN;
  let minScroll = viewableHeight - contentHeight - MARGIN - 20;
  let scrollRange = maxScroll - minScroll;

  let scrollRatio = (maxScroll - currentScrollY) / scrollRange;

  let barY = scrollRatio * (viewableHeight - barHeight);
  let scrollbarX = width - SCROLLBAR_WIDTH;

  fill(200, 100);
  noStroke();
  rect(scrollbarX, 0, SCROLLBAR_WIDTH, viewableHeight);

  if (isDraggingScrollbar) {
    fill(100, 180, 255);
  } else {
    fill(150);
  }
  rect(scrollbarX, barY, SCROLLBAR_WIDTH, barHeight, 5);
}

function drawSidebar() {
  let sidebarHeight = height - NEXT_BUTTON_AREA_HEIGHT;

  fill(255);
  noStroke();
  rect(0, 0, SIDEBAR_WIDTH, sidebarHeight);

  stroke(200);
  strokeWeight(1);
  line(SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, sidebarHeight);

  let burgerX = SIDEBAR_WIDTH / 2 - 15;
  let burgerY = 20;
  let burgerW = 30;
  let lineGap = 8;

  stroke(50);
  strokeWeight(3);
  line(burgerX, burgerY, burgerX + burgerW, burgerY);
  line(burgerX, burgerY + lineGap, burgerX + burgerW, burgerY + lineGap);
  line(burgerX, burgerY + lineGap * 2, burgerX + burgerW, burgerY + lineGap * 2);

  let iconSize = 34;
  let iconX = SIDEBAR_WIDTH / 2 - iconSize / 2;
  let iconY = 70;

  let isHighlighted = (millis() - newChatIconHighlightStart < HIGHLIGHT_DURATION) || isHoveringResetIcon;

  if (isHighlighted) {
    fill(100, 180, 255);
    stroke(100, 180, 255);
  } else {
    noFill();
    stroke(50);
  }

  strokeWeight(2);
  rect(iconX, iconY, iconSize, iconSize, 5);

  if (isHighlighted) {
    stroke(255);
  } else {
    stroke(50);
  }
  let centerIconX = iconX + iconSize / 2;
  let centerIconY = iconY + iconSize / 2;
  let plusSize = 10;
  line(centerIconX - plusSize, centerIconY, centerIconX + plusSize, centerIconY);
  line(centerIconX, centerIconY - plusSize, centerIconX, centerIconY + plusSize);

  if (isHoveringResetIcon) {
    let tooltipW = 120; 
    let tooltipH = 65; 
    let tooltipX = max(0, (SIDEBAR_WIDTH - tooltipW) / 2);
    let tooltipY = iconY + iconSize + 5;
    
    fill(255);
    stroke(0); 
    strokeWeight(1);
    rect(tooltipX, tooltipY, tooltipW, tooltipH, 4);

    noStroke();
    fill(0);
    textSize(24); 
    textAlign(CENTER, CENTER);
    
    let centerX = tooltipX + tooltipW / 2;

    text("チャットの", centerX, tooltipY + tooltipH / 2 - 12);
    text("新規作成", centerX, tooltipY + tooltipH / 2 + 12);
    
    textSize(TEXT_SIZE);
  }

  noStroke();
}

function updateContentHeight() {
  let totalH = 0;
  for (let msg of history) {
    totalH += calcMessageHeight(msg) + 15;
  }

  if (isWaitingForAgent) {
    totalH += 60;
  }

  contentHeight = totalH;
}

function calcMessageHeight(msg) {
  let h = 0;
  let bubblePadding = BUBBLE_PADDING;

  if (msg.isUser) {
    // ★修正: 禁則処理対応の高さ計算
    let textH = calculateTextHeight(msg.text, BUBBLE_WIDTH - bubblePadding * 2);
    h = int(textH) + bubblePadding * 2;
  } else {
    if (msg.text.length > 0) {
      // ★修正: 禁則処理対応の高さ計算
      let textH = calculateTextHeight(msg.text, BUBBLE_WIDTH - bubblePadding * 2);
      let textBoxH = int(textH) + bubblePadding * 2;
      h += textBoxH;
      if (msg.img != null) h += 10;
    }
    if (msg.img != null) {
      h += msg.img.height;
    }
  }
  return h;
}

function mouseWheel(event) {
  let e = event.delta;
  let scrollStep = (e > 0) ? 1 : -1;
  let scrollSpeed = 20;
  targetScrollY -= scrollStep * scrollSpeed;

  logScrollAction("Wheel");

  return false;
}

function scrollToBottom() {
  let viewableHeight = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
  let minScroll = viewableHeight - contentHeight - MARGIN - 20;
  if (minScroll > MARGIN) minScroll = MARGIN;

  targetScrollY = minScroll;
}

function updateAgentResponse() {
  if (isWaitingForAgent && millis() > nextResponseTime) {
    if (scenarioIndex < scenario.length) {
      let nextMsg = scenario[scenarioIndex];
      if (!nextMsg.isUser) {
        history.push(nextMsg);
        scenarioIndex++;

        isWaitingForAgent = false;
        updateContentHeight();
        scrollToBottom();

        if (scenarioIndex < scenario.length && scenario[scenarioIndex].isUser) {
          isReadyToStartTyping = true;
        } else if (scenarioIndex < scenario.length && !scenario[scenarioIndex].isUser) {
          nextResponseTime = millis() + AGENT_DELAY_MS;
          isWaitingForAgent = true;
          updateContentHeight();
          scrollToBottom();
        } else {
          isWaitingForAgent = false;
        }
      } else {
        isWaitingForAgent = false;
      }
    } else {
      isWaitingForAgent = false;
    }
  }
}

function calculateInputAreaHeight() {
  let nextText = "";
  if (!isWaitingForAgent && !isUserTyping && !isResetting && !isReadyToStartTyping && scenarioIndex < scenario.length) {
    let nextMsg = scenario[scenarioIndex];
    if (nextMsg.isUser) {
      nextText = nextMsg.text;
    }
  }
  let chatAreaWidth = width - SIDEBAR_WIDTH;
  let textBoxWidth = chatAreaWidth - MARGIN * 3 - 80;

  // ★修正: 禁則処理対応の高さ計算
  let txtH = calculateTextHeight(nextText, textBoxWidth - 20);

  currentInputHeight = max(MIN_INPUT_HEIGHT, txtH + 20 + MARGIN * 2);
}

function drawInputArea() {
  let areaY = height - NEXT_BUTTON_AREA_HEIGHT - currentInputHeight;

  let inputAreaX = SIDEBAR_WIDTH;
  let inputAreaWidth = width - SIDEBAR_WIDTH;

  fill(200);
  noStroke();
  rect(inputAreaX, areaY, inputAreaWidth, currentInputHeight);

  let boxHeight = currentInputHeight - MARGIN * 2;
  let boxWidth = inputAreaWidth - MARGIN * 3 - 80;

  fill(255);
  rect(inputAreaX + MARGIN, areaY + MARGIN, boxWidth, boxHeight);

  let isScenarioFinished = (scenarioIndex >= scenario.length && !isWaitingForAgent && !isUserTyping && !isReadyToStartTyping && !isLoadingNextScenario);
  let isHighlighted = (millis() - sendButtonHighlightStart < HIGHLIGHT_DURATION);

  if (allSessionsCompleted && !isResetting) {
    fill(100);
  } else if (isHighlighted) {
    fill(100, 180, 255);
  } else if (isWaitingForAgent || isUserTyping || isResetting || isReadyToStartTyping || isLoadingNextScenario) {
    fill(150); 
  } else {
    fill(50);
  }

  let btnHeight = 40;
  let btnY = (areaY + MARGIN + boxHeight) - btnHeight;

  rect(inputAreaX + inputAreaWidth - MARGIN - 80, btnY, 80, btnHeight, 5);

  fill(255);
  textAlign(CENTER, CENTER);
  textSize(TEXT_SIZE);

  if (allSessionsCompleted) {
    text("終了", inputAreaX + inputAreaWidth - MARGIN - 40, btnY + btnHeight / 2);
  } else {
    text("送信", inputAreaX + inputAreaWidth - MARGIN - 40, btnY + btnHeight / 2);
  }

  if (allSessionsCompleted && !isResetting) {
    fill(50);
    textAlign(LEFT, CENTER);
    text("...", inputAreaX + MARGIN + 10, areaY + currentInputHeight / 2);
  } else if (isUserTyping) {
    if (millis() - userTypingStartTime > START_DELAY_MS) {
      fill(150); 
      textAlign(LEFT, TOP);
      text("入力中です", inputAreaX + MARGIN + 10, areaY + MARGIN + 10);
       
      let startX = inputAreaX + MARGIN + 10 + textWidth("入力中です") + 5;
      let startY = areaY + MARGIN + 15; 
      let time = millis() * 0.01;
      for (let i = 0; i < 3; i++) {
        let offset = sin(time + i * 1.5) * 3;
        ellipse(startX + i * 10, startY + offset, 4, 4);
      }
    }

  } else if (!isWaitingForAgent && !isResetting && !isScenarioFinished && !isReadyToStartTyping && !isLoadingNextScenario) {
    let nextMsg = scenario[scenarioIndex];
    if (nextMsg.isUser) {
      fill(0);
      textAlign(LEFT, TOP);
      textLeading(TEXT_LEADING);
      
      // ★修正: 入力エリア内も禁則処理付き描画
      let lines = splitTextToLines(nextMsg.text, boxWidth - 20);
      let ly = areaY + MARGIN + 10;
      for(let lineStr of lines) {
        text(lineStr, inputAreaX + MARGIN + 10, ly);
        ly += TEXT_LEADING;
      }
    }
  }
}

function drawNextButtonArea() {
  let areaY = height - NEXT_BUTTON_AREA_HEIGHT;

  fill(50);
  noStroke();
  rect(0, areaY, width, NEXT_BUTTON_AREA_HEIGHT);

  let isHover = false;
  let centerX = 0 + width / 2;
  if (mouseX > centerX - 100 && mouseX < centerX + 100 &&
      mouseY > areaY + 20 && mouseY < areaY + 20 + 60) {
    isHover = true;
  }

  if (isHover) {
    fill(255, 100, 100);
  } else {
    fill(200, 50, 50);
  }
  
  if ((isResetting && !isHoveringResetIcon) || isLoadingNextScenario || isWaitingForAgent || isUserTyping) {
     fill(100);
  }

  centerX = width / 2;
  rect(centerX - 100, areaY + 20, 200, 60, 10);

  fill(255);
  textSize(24);
  textAlign(CENTER, CENTER);
   
  if (allSessionsCompleted && !isResetting) {
    text("最初に戻る", centerX, areaY + 20 + 30);
  } else {
    text("次へ", centerX, areaY + 20 + 30);
  }
  textSize(TEXT_SIZE);
}

function logUserAction(eventName) {
  let currentTime = millis();
  let interval = (currentTime - lastLogTime) / 1000.0;
  let evt = eventName || "Next Pressed";
  
  let logLine = "ItoI_noname, " + nf(interval, 0, 3) + ", " + evt + ", Scenario:" + currentScenarioID + ", Step:" + scenarioIndex;
  
  //console.log(logLine);
  if (typeof timeLogData === 'undefined') timeLogData = ""; 
  timeLogData += logLine + "\n";
  window.timeLogData = timeLogData;
  localStorage.setItem('chatTimeLog', timeLogData);
  
  lastLogTime = currentTime;
}

function getVisibleMessageIndexes() {
  let visibleIndices = [];
  
  let viewTop = -currentScrollY; 
  let viewBottom = viewTop + (height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT);
  
  let msgY = 0; 
  
  for (let i = 0; i < history.length; i++) {
    let msgH = calcMessageHeight(history[i]); 
    let msgTop = msgY;
    let msgBottom = msgY + msgH;
    
    if (msgBottom > viewTop && msgTop < viewBottom) {
      visibleIndices.push(i);
    }
    
    msgY += msgH + 20; 
  }
  
  if (visibleIndices.length === 0) return "None";
  return visibleIndices.join("-"); 
}

function logScrollAction(eventType) {
  let now = millis();
  
  if (eventType === "Drag") {
    if (now - lastScrollLogTime < SCROLL_LOG_INTERVAL) return;
  }
  
  lastScrollLogTime = now;
  
  let formattedTime = nf((now - scenarioStartTime) / 1000.0, 0, 3);
  let scrollYVal = nf(currentScrollY, 0, 1);
  
  let viewableH = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
  let maxScroll = contentHeight + MARGIN * 2 + 20 - viewableH;
  let scrollPercent = 0;
  
  if (maxScroll > 0) {
    let currentAbs = abs(currentScrollY - MARGIN); 
    scrollPercent = constrain(currentAbs / maxScroll, 0, 1);
  }
  let percentStr = nf(scrollPercent, 0, 2); 
  
  let visibleMsg = getVisibleMessageIndexes();
  
  let lineData = "ItoI_noname" + ", " + formattedTime + ", " + eventType + ", " + scrollYVal + ", " + percentStr + ", " + visibleMsg + ", " + currentScenarioID + "\n";
  
  scrollLogData += lineData;
  localStorage.setItem('chatScrollLog', scrollLogData);
}

function mousePressed() {
  let viewableHeight = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
  if (mouseX > width - SCROLLBAR_WIDTH && mouseY < viewableHeight) {
    isDraggingScrollbar = true;
    return;
  }

  if (isWaitingForAgent || isUserTyping || (isResetting && !isHoveringResetIcon) || isLoadingNextScenario) return;

  let nextBtnAreaY = height - NEXT_BUTTON_AREA_HEIGHT;

  let centerX = width / 2;
  if (mouseX > centerX - 100 && mouseX < centerX + 100 &&
      mouseY > nextBtnAreaY + 20 && mouseY < nextBtnAreaY + 20 + 60) {

    let eventName = "Next Pressed";
    if (isHoveringResetIcon) {
        eventName = "Create New Chat Pressed";
    } else if (allSessionsCompleted && !isResetting) {
        eventName = "End/Reset Pressed";
    } else if (isReadyToStartTyping) {
        eventName = "Start Typing Pressed";
    } else {
        if (scenarioIndex < scenario.length) {
            if (scenario[scenarioIndex].isUser) eventName = "Send Message Pressed";
            else eventName = "Next Message Pressed";
        }
    }
    
    logUserAction(eventName);

    if (isHoveringResetIcon) {
       proceedToNextScenario(); 
       return;
    }

    if (allSessionsCompleted && !isResetting) {
      fullReset();
      return; 
    }

    if (isReadyToStartTyping) {
       startUserTyping();
       isReadyToStartTyping = false;
       return;
    }

    if (scenarioIndex >= scenario.length) {
      resetConversation();
    } else {
      sendButtonHighlightStart = millis();
      advanceScenario();
    }
  }
}

function mouseDragged() {
  if (isDraggingScrollbar) {
    let viewableHeight = height - currentInputHeight - NEXT_BUTTON_AREA_HEIGHT;
    let totalHeight = contentHeight + MARGIN * 2 + 20;

    if (totalHeight <= viewableHeight) return;

    let ratio = constrain(mouseY / viewableHeight, 0, 1);

    let maxScroll = MARGIN;
    let minScroll = viewableHeight - contentHeight - MARGIN - 20;
    let scrollRange = maxScroll - minScroll;

    let newY = maxScroll - (ratio * scrollRange);
    currentScrollY = newY;
    targetScrollY = newY;
    
    logScrollAction("Drag");
  }
}

function mouseReleased() {
  if (isDraggingScrollbar) {
    logScrollAction("DragEnd");
  }
  isDraggingScrollbar = false;
}

function resetConversation() {
  // history = []; 
  isWaitingForAgent = false;
  isUserTyping = false; 

  if (currentScenarioID >= 1) {
    allSessionsCompleted = true;
  } else {
    isResetting = true;
    resetStartTime = millis();
  }
}

function proceedToNextScenario() {
  history = [];
  isLoadingNextScenario = true;
  loadingStartTime = millis();

  currentScenarioID++;
  scenarioIndex = 0;
  currentScrollY = MARGIN;
  targetScrollY = MARGIN;
  lastLogTime = millis();
  
  isResetting = false;
  isHoveringResetIcon = false; 
  iconHighlightTriggered = false;
}

function fullReset() {
  currentScenarioID = 0;
  scenarioIndex = 0;
    // ★追加: リセット時にも時間をリセット
  scenarioStartTime = millis();
  history = [];
  allSessionsCompleted = false;
  isResetting = false;
  isWaitingForAgent = false;
  isUserTyping = false;
  isReadyToStartTyping = false;
  isHoveringResetIcon = false;
  isLoadingNextScenario = false;
  // ★追加: フラグもリセット
  completionMessageSent = false;
    
  currentScrollY = MARGIN;
  targetScrollY = MARGIN;
  contentHeight = 0;
    
  let currentTime = millis();
  let logLine = "System, Reset Performed, -, -";
  //console.log(logLine);
    
  if (typeof timeLogData === 'undefined') timeLogData = ""; 
  timeLogData += logLine + "\n";
  window.timeLogData = timeLogData;
  localStorage.setItem('chatTimeLog', timeLogData);
    
  lastLogTime = currentTime;
    
  setupScenario();
}

function advanceScenario() {
  if (scenarioIndex >= scenario.length) return;
  let currentMsg = scenario[scenarioIndex];

  if (currentMsg.isUser) {
    history.push(currentMsg);
    scenarioIndex++;

    isWaitingForAgent = true;
    updateContentHeight();
    scrollToBottom();

    if (scenarioIndex < scenario.length && !scenario[scenarioIndex].isUser) {
      nextResponseTime = millis() + AGENT_DELAY_MS;
      isWaitingForAgent = true; 
    } else if (scenarioIndex < scenario.length && scenario[scenarioIndex].isUser) {
      isReadyToStartTyping = true;
      isWaitingForAgent = false;
    } else {
      isWaitingForAgent = false;
    }
    updateContentHeight();
    scrollToBottom();
  }
}