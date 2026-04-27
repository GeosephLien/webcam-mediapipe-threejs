# VRM Face Capture Demo

這個頁面會做以下流程：

1. 讀取 webcam 影像
2. 用 MediaPipe Face Landmarker 取得臉部 blendshapes 與頭部姿態
3. 把數值寫到 VRM 的 ARKit 命名 morph targets
4. 如果模型沒有對應 morph targets，退回到 VRM 預設表情
5. 用 three.js 即時更新角色

## 啟動

在專案目錄執行：

```powershell
python -m http.server 8000
```

然後開啟：

```text
http://localhost:8000
```

## 模型位置

目前固定載入：

`./vrms/Iris_z19t_max.vrm`

## 備註

- 第一次啟動時瀏覽器會要求攝影機權限。
- 頁面需要連到 CDN 與 MediaPipe 模型網址，所以要有網路。
- 如果你的 VRM 本身已經帶 ARKit 命名 blendshapes，表情會更完整。
