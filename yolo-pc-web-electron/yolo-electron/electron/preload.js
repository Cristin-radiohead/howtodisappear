const { contextBridge } = require('electron');

// 暴露 API 地址给前端
contextBridge.exposeInMainWorld('api', {
    baseUrl: 'http://127.0.0.1:8000'
});