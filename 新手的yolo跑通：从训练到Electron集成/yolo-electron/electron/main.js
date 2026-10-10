const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

let backendProcess = null;
let mainWindow = null; // ← 【关键】声明全局变量，防止窗口被垃圾回收

// 启动 Python 后端
function startBackend() {
    const backendPath = app.isPackaged
        ? path.join(process.resourcesPath, 'backend', 'server.exe')
        : 'C:\\Users\\LEGION\\yolo-env\\Scripts\\python.exe';

    const args = app.isPackaged
        ? []
        : [path.join(__dirname, '..', 'backend', 'server.py')];

    backendProcess = spawn(backendPath, args, {
        stdio: 'inherit',
        windowsHide: true
    });

    backendProcess.on('error', (err) => {
        console.error('[backend] 启动失败:', err);
    });
}

// 轮询等待后端就绪
function waitForBackend(retries = 60) {
    return new Promise((resolve, reject) => {
        const check = (n) => {
            http.get('http://127.0.0.1:8000/health', (res) => {
                if (res.statusCode === 200) resolve();
                else if (n > 0) setTimeout(() => check(n - 1), 500);
                else reject(new Error('Backend timeout'));
            }).on('error', () => {
                if (n > 0) setTimeout(() => check(n - 1), 500);
                else reject(new Error('Backend not reachable'));
            });
        };
        check(retries);
    });
}

app.whenReady().then(async () => {
    console.log('正在启动后端...');
    startBackend();

    try {
        await waitForBackend();
        console.log('后端已就绪，准备创建窗口...');
    } catch (e) {
        console.error('后端启动失败:', e);
    }

    // 【关键】赋值给全局变量 mainWindow，而不是用 const win
    mainWindow = new BrowserWindow({
        width: 900,
        height: 700,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'index.html'))
       .then(() => {
           console.log('前端页面加载成功，显示窗口');
           mainWindow.show(); // 确保窗口显示
       })
       .catch((err) => {
           console.error('加载 index.html 失败:', err);
       });

    // 监听窗口被关闭的事件，方便排查
    mainWindow.on('closed', () => {
        console.log('窗口被关闭了');
        mainWindow = null;
    });
});

app.on('window-all-closed', () => {
    if (backendProcess) backendProcess.kill();
    app.quit();
});